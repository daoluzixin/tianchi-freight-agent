"""规则化偏好解析器：LLM 不可用时的 fallback。

通过正则表达式和关键词匹配，从偏好原文中提取结构化约束。
不依赖 LLM，零网络调用，确保在任何环境下都能正确识别关键偏好。

设计原则：
  - 宁可漏分类为 custom（后续每步 LLM 评估），也不能错误分类导致硬约束误判
  - 对高罚分约束（go_home、special_cargo、family_event）优先匹配
  - 所有正则基于 10 个司机的实际偏好文本设计，但保持泛化能力
"""

from __future__ import annotations

import re
import logging
from typing import Any

# ---------------------------------------------------------------------------
# 中文数字 → 阿拉伯数字 转换工具
# ---------------------------------------------------------------------------
_CN_DIGIT_MAP = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT_MAP = {"十": 10, "百": 100, "千": 1000}


def _cn_to_number(text: str) -> int | None:
    """将简单中文数字字符串转为整数，失败返回 None。

    支持：一百五十 → 150, 二十 → 20, 八 → 8, 十二 → 12 等。
    """
    if not text:
        return None
    # 如果已经是纯数字直接返回
    if text.isdigit():
        return int(text)
    result = 0
    current = 0
    for ch in text:
        if ch in _CN_DIGIT_MAP:
            current = _CN_DIGIT_MAP[ch]
        elif ch in _CN_UNIT_MAP:
            unit = _CN_UNIT_MAP[ch]
            if current == 0:
                current = 1  # "十二" → 1*10+2
            result += current * unit
            current = 0
        else:
            return None  # 无法识别的字符
    result += current
    return result if result > 0 else None


from agent.config.preference_parser import (
    ParsedPreferences,
    RestConstraint,
    QuietWindowConstraint,
    ForbiddenCategoryConstraint,
    MaxDistanceConstraint,
    MaxOrdersConstraint,
    FirstOrderDeadlineConstraint,
    OffDaysConstraint,
    GeoFenceConstraint,
    ForbiddenZoneConstraint,
    GoHomeConstraint,
    SpecialCargoConstraint,
    FamilyEventConstraint,
    VisitTargetConstraint,
    CustomConstraint,
)

logger = logging.getLogger("agent.rule_based_parser")


def _safe_cap_value(val: Any) -> float | None:
    """安全转换 penalty_cap 值。"""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ===========================================================================
# 各类型解析函数
# ===========================================================================

def _try_parse_family_event(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析家事事件：含'家中急事'或'临时约定·家事'标记。"""
    if "家中急事" not in content and "家事" not in content:
        return False

    # 提取触发时间：YYYY年M月D日 HH:MM
    trigger_match = re.search(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})", content
    )
    if not trigger_match:
        return False

    trigger_time = (
        f"{trigger_match.group(1)}-{int(trigger_match.group(2)):02d}-"
        f"{int(trigger_match.group(3)):02d} {int(trigger_match.group(4)):02d}:"
        f"{trigger_match.group(5)}:00"
    )

    # 提取配偶接驳点坐标：须先到（lat，lng）接上配偶
    spouse_match = re.search(
        r"(?:到|至)[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)].*?接.*?配偶", content
    )
    waypoints = []
    if spouse_match:
        wp_lat = float(spouse_match.group(1))
        wp_lng = float(spouse_match.group(2))
        # 提取原地停留时间
        wait_match = re.search(r"停留不少于\s*(\d+)\s*分钟", content)
        wait_min = int(wait_match.group(1)) if wait_match else 10
        waypoints.append({"lat": wp_lat, "lng": wp_lng, "wait_minutes": wait_min})

    # 提取老家坐标：返回老家（lat，lng）
    home_match = re.search(
        r"(?:返回|回)老家[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)]", content
    )
    home_lat = float(home_match.group(1)) if home_match else 0.0
    home_lng = float(home_match.group(2)) if home_match else 0.0

    # 提取 home_deadline：须在 YYYY年M月D日HH:MM 前进家门
    deadline_match = re.search(
        r"须在\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})\s*前进家门",
        content
    )
    home_deadline = ""
    if deadline_match:
        home_deadline = (
            f"{deadline_match.group(1)}-{int(deadline_match.group(2)):02d}-"
            f"{int(deadline_match.group(3)):02d} {int(deadline_match.group(4)):02d}:"
            f"{deadline_match.group(5)}:00"
        )

    # 提取 stay_until：至少待到 YYYY年M月D日HH:MM
    stay_match = re.search(
        r"(?:至少)?待到\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})",
        content
    )
    stay_until = ""
    if stay_match:
        stay_until = (
            f"{stay_match.group(1)}-{int(stay_match.group(2)):02d}-"
            f"{int(stay_match.group(3)):02d} {int(stay_match.group(4)):02d}:"
            f"{stay_match.group(5)}:00"
        )

    # 提取每分钟罚金："每迟到或不在家1分钟罚5元"
    per_min_match = re.search(r"每.*?1分钟罚\s*(\d+)\s*元", content)
    penalty_per_minute = float(per_min_match.group(1)) if per_min_match else 5.0

    # 提取一次性罚金
    once_match = re.search(r"一次性罚\s*(\d+)\s*元", content)
    penalty_once = float(once_match.group(1)) if once_match else penalty_amount

    result.family_events.append(FamilyEventConstraint(
        trigger_time=trigger_time,
        waypoints=waypoints,
        home_lat=home_lat,
        home_lng=home_lng,
        home_deadline=home_deadline,
        stay_until=stay_until,
        penalty_per_minute_late=penalty_per_minute,
        penalty_once_if_failed=penalty_once,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed family_event trigger=%s home=(%.2f,%.2f)",
                trigger_time, home_lat, home_lng)
    return True


def _try_parse_special_cargo(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析特殊货源：含'熟货'/'指定货源'/'临时约定'关键词 + 货源编号。"""
    if "熟货" not in content and "指定货源" not in content and "临时约定" not in content:
        return False

    # 匹配货源编号
    id_match = re.search(r"(?:编号|货源)\s*(\d{4,})", content)
    if not id_match:
        return False

    cargo_id = id_match.group(1)

    # 提取装货地坐标
    loc_match = re.search(
        r"装货(?:地|点)[：:].*?[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)]", content
    )
    pickup_lat = float(loc_match.group(1)) if loc_match else 0.0
    pickup_lng = float(loc_match.group(2)) if loc_match else 0.0

    # 提取上架时间
    time_match = re.search(
        r"上架时间[：:]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", content
    )
    available_from = time_match.group(1) if time_match else ""

    # 提取经济损失金额
    loss_match = re.search(r"(?:经济损失|损失)\s*(\d+)\s*元", content)
    penalty_if_missed = float(loss_match.group(1)) if loss_match else penalty_amount

    result.special_cargos.append(SpecialCargoConstraint(
        cargo_id=cargo_id,
        available_from=available_from,
        pickup_lat=pickup_lat,
        pickup_lng=pickup_lng,
        penalty_if_missed=penalty_if_missed,
    ))
    logger.info("Rule-based: parsed special_cargo id=%s from=%s", cargo_id, available_from)
    return True


def _try_parse_go_home(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析回家约束：含'X点前'+'自家位置/回家/到家'+'坐标'。"""
    # 模式1："每天23点前车辆须在自家位置（23.12，113.28）一公里内"
    match = re.search(
        r"(\d{1,2})\s*点前.*?(?:自家|回家|到家|在家).*?[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)]",
        content
    )
    if match:
        deadline_hour = int(match.group(1))
        home_lat = float(match.group(2))
        home_lng = float(match.group(3))
    else:
        # 模式2："家在（lat，lng），每天晚上23点前必须把车开回家附近一公里内"
        # 坐标在前，deadline在后
        match2 = re.search(
            r"家在?[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)].*?(\d{1,2})\s*点前.*?(?:回家|到家|开回)",
            content
        )
        if not match2:
            # 模式3：更宽松 — 含"回家"且有坐标和时间
            match2 = re.search(
                r"家.*?[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)].*?(\d{1,2})\s*点",
                content
            )
            if match2 and not any(kw in content for kw in ["回家", "到家", "开回", "在家"]):
                match2 = None
        if not match2:
            return False
        home_lat = float(match2.group(1))
        home_lng = float(match2.group(2))
        deadline_hour = int(match2.group(3))

    # 提取半径（默认 1km）
    radius_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|km|千米)内", content)
    radius_km = float(radius_match.group(1)) if radius_match else 1.0

    # 检查是否有安静窗口（"当天23点至次日8点不接单"）
    quiet_match = re.search(
        r"(\d{1,2})\s*点(?:至|到)次日\s*(\d{1,2})\s*点", content
    )
    quiet_start = deadline_hour
    quiet_end = 8
    if quiet_match:
        quiet_start = int(quiet_match.group(1))
        quiet_end = int(quiet_match.group(2))

    result.go_home.append(GoHomeConstraint(
        home_lat=home_lat,
        home_lng=home_lng,
        deadline_hour=deadline_hour,
        quiet_start_hour=quiet_start,
        quiet_end_hour=quiet_end,
        radius_km=radius_km,
        penalty_per_day=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed go_home deadline=%dh home=(%.2f,%.2f)",
                deadline_hour, home_lat, home_lng)
    return True


def _try_parse_visit_target(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析月度到访目标：至少N个不同自然日到过（lat，lng）。支持多点巡回。"""

    # ---- 多点巡回模式 ----
    # "分别到过以下三个地点各至少3天：A点（22.95，113.25）一公里内、B点（23.10，113.45）一公里内..."
    multi_match = re.search(
        r"(?:分别|各自)?.*?(?:到过|到达|前往).*?(?:以下|如下)?.*?(?:各)?(?:至少)?\s*(\d+)\s*(?:个)?(?:不同)?\s*(?:天|日|自然日)",
        content
    )
    if multi_match:
        # 提取所有坐标点：（lat，lng）后面可能跟半径
        points = re.findall(
            r"[（(]\s*([\d.]+)\s*[，,]\s*([\d.]+)\s*[）)]\s*(?:附近)?\s*(?:(\d+(?:\.\d+)?)\s*(?:公里|km|千米)内)?",
            content
        )
        if len(points) >= 2:
            min_days = int(multi_match.group(1))
            for pt in points:
                target_lat = float(pt[0])
                target_lng = float(pt[1])
                radius_km = float(pt[2]) if pt[2] else 1.0
                result.visit_targets.append(VisitTargetConstraint(
                    target_lat=target_lat,
                    target_lng=target_lng,
                    radius_km=radius_km,
                    min_days=min_days,
                    penalty_once=penalty_amount,
                    penalty_cap=penalty_cap,
                ))
                logger.info("Rule-based: parsed visit_target (multi) (%.2f,%.2f) min_days=%d r=%.1fkm",
                            target_lat, target_lng, min_days, radius_km)
            return True

    # ---- 单点模式 ----
    # 模式1："N个不同自然日到过（lat，lng）"
    match = re.search(
        r"(\d+)\s*个.*?自然日.*?到(?:过|达).*?[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)]",
        content
    )
    if not match:
        # 模式2："每月N天到（lat，lng）附近" / "N天到...打卡"
        match = re.search(
            r"(?:每月)?\s*(\d+)\s*(?:个)?(?:不同)?\s*(?:天|日).*?(?:到|去|前往).*?[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)]",
            content
        )
    if not match:
        # 模式3：坐标在前 — "到（lat，lng）附近...N天"
        match3 = re.search(
            r"(?:到|去|前往).*?[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)].*?(\d+)\s*(?:个)?(?:不同)?\s*(?:天|日|自然日)",
            content
        )
        if match3:
            min_days = int(match3.group(3))
            target_lat = float(match3.group(1))
            target_lng = float(match3.group(2))
            radius_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|km|千米)内", content)
            radius_km = float(radius_match.group(1)) if radius_match else 1.0
            result.visit_targets.append(VisitTargetConstraint(
                target_lat=target_lat,
                target_lng=target_lng,
                radius_km=radius_km,
                min_days=min_days,
                penalty_once=penalty_amount,
                penalty_cap=penalty_cap,
            ))
            logger.info("Rule-based: parsed visit_target (%.2f,%.2f) min_days=%d",
                        target_lat, target_lng, min_days)
            return True
    if not match:
        return False

    min_days = int(match.group(1))
    target_lat = float(match.group(2))
    target_lng = float(match.group(3))

    radius_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|km|千米)内", content)
    radius_km = float(radius_match.group(1)) if radius_match else 1.0

    result.visit_targets.append(VisitTargetConstraint(
        target_lat=target_lat,
        target_lng=target_lng,
        radius_km=radius_km,
        min_days=min_days,
        penalty_once=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed visit_target (%.2f,%.2f) min_days=%d",
                target_lat, target_lng, min_days)
    return True


def _try_parse_quiet_window(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析禁止活动时段：N点至M点不接单不空驶。"""
    # 排除 go_home 类型（已在前面匹配）
    if any(kw in content for kw in ["回家", "到家", "自家位置", "进家门"]):
        return False

    start_hour: int | None = None
    start_minute: int = 0
    end_hour: int | None = None
    end_minute: int = 0

    # 模式1："每天凌晨2点至5点不接单" / "每晚23点至次日早6点" / "凌晨0点到早上8点"
    # 支持"N点半"表达：捕获可选的"半"/"一刻"/"三刻"后缀
    match = re.search(
        r"(?:每[天日晚]|凌晨)?\s*(\d{1,2})\s*点\s*(半|一刻|三刻)?\s*(?:至|到)\s*(?:次日)?(?:早上|早|晨|凌晨)?\s*(\d{1,2})\s*点\s*(半|一刻|三刻)?",
        content
    )
    if not match:
        # 模式2："中午12点至下午1点"
        match = re.search(
            r"(?:中午|下午|上午)?\s*(\d{1,2})\s*点\s*(半|一刻|三刻)?\s*(?:至|到)\s*(?:下午|上午|晚上)?\s*(\d{1,2})\s*点\s*(半|一刻|三刻)?",
            content
        )
    if match:
        start_hour = int(match.group(1))
        _start_suffix = match.group(2)
        if _start_suffix == "半":
            start_minute = 30
        elif _start_suffix == "一刻":
            start_minute = 15
        elif _start_suffix == "三刻":
            start_minute = 45
        end_hour = int(match.group(3))
        _end_suffix = match.group(4)
        if _end_suffix == "半":
            end_minute = 30
        elif _end_suffix == "一刻":
            end_minute = 15
        elif _end_suffix == "三刻":
            end_minute = 45

        # 修正 AM/PM 歧义：根据上下文中的"下午"/"晚上"/"中午"等前缀
        # 例如 "中午12点至下午1点" → start=12, end=13（而非 end=1）
        matched_text = match.group(0)
        # 检查 end_hour 前面是否有"下午"/"午后"标记
        if end_hour < 12 and re.search(r"(?:下午|午后)\s*" + str(end_hour) + r"\s*点", content):
            end_hour += 12
        # 检查 start_hour 前面是否有"下午"/"午后"标记
        if start_hour < 12 and re.search(r"(?:下午|午后)\s*" + str(start_hour) + r"\s*点", content):
            start_hour += 12
        # 检查"晚上"标记
        if end_hour < 12 and re.search(r"晚上\s*" + str(end_hour) + r"\s*点", content):
            end_hour += 12
        if start_hour < 12 and re.search(r"晚上\s*" + str(start_hour) + r"\s*点", content):
            start_hour += 12
        # 通用合理性修正：如果 start >= 12 且 end < 12 且差距很大（非跨天场景），
        # 且文本不含"次日"/"第二天"等跨天标记，则 end 可能是 PM
        if (start_hour >= 12 and end_hour < 12 and end_hour > 0
                and "次日" not in content and "第二天" not in content
                and (start_hour - end_hour) > 6):
            end_hour += 12

    # 模式3：HH:MM 格式 — "02:00至05:00" / "夜间02:00至05:00时段内"
    if start_hour is None:
        match_hm = re.search(
            r"(\d{1,2}):(\d{2})\s*(?:至|到|-|~)\s*(\d{1,2}):(\d{2})",
            content
        )
        if match_hm:
            start_hour = int(match_hm.group(1))
            start_minute = int(match_hm.group(2))
            end_hour = int(match_hm.group(3))
            end_minute = int(match_hm.group(4))

    # 模式4：描述性 — "晚上11点一到就收工了，一直到第二天早上6点之前"
    if start_hour is None:
        match_desc = re.search(
            r"(?:晚上|深夜|夜里)?\s*(\d{1,2})\s*点.*?(?:收工|不干|不接|不允许).*?(?:第二天|次日)?.*?(?:早上|凌晨|早)?\s*(\d{1,2})\s*点",
            content
        )
        if match_desc:
            start_hour = int(match_desc.group(1))
            end_hour = int(match_desc.group(2))

    # 模式5："之后...之前" — "深夜11点之后到凌晨4点之前不允许接任何订单"
    if start_hour is None:
        match_range = re.search(
            r"(\d{1,2})\s*点\s*(?:之后|以后|过后).*?(?:到|至).*?(\d{1,2})\s*点\s*(?:之前|以前|前)",
            content
        )
        if match_range:
            start_hour = int(match_range.group(1))
            end_hour = int(match_range.group(2))

    if start_hour is None or end_hour is None:
        return False

    # 需要确认是安静窗口（含"不接单"/"不空"/"歇脚"/"吃饭"等关键词）
    if not any(kw in content for kw in [
        "不接单", "不空", "歇脚", "吃饭", "不出车",
        "禁止承接", "禁止", "不允许", "收工", "不接",
        "不接任何", "不跑", "拒绝一切", "拒绝接单",
        "固定休息", "休息时间",
    ]):
        return False

    result.quiet_windows.append(QuietWindowConstraint(
        start_hour=start_hour,
        start_minute=start_minute,
        end_hour=end_hour,
        end_minute=end_minute,
        penalty_per_day=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed quiet_window %02d:%02d-%02d:%02d",
                start_hour, start_minute, end_hour, end_minute)
    return True


# 已知合法品类名称集合（用于无书名号时的品类识别）
_KNOWN_CATEGORIES = {
    "化工塑料", "煤炭矿产", "蔬菜", "鲜活水产", "快递快运", "搬家",
    "食品饮料", "建材", "家电", "家具", "日用百货", "粮食",
    "设备", "金属", "木材", "纺织", "农资", "危化品",
    "冷链", "生鲜", "水果", "药品", "电子产品", "汽配",
    "小件散货",
}


def _try_parse_forbidden_category(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析禁止品类：含'不接/不拉...品类为「X」'。"""
    # 需要确认与品类相关（扩展关键词列表）
    has_category_signal = any(kw in content for kw in [
        "品类", "不接", "不拉", "不碰", "不干", "不要派",
        "不做", "不运", "不送", "少派", "类的货", "类的单",
        "这类", "那种货",
    ])
    if not has_category_signal:
        return False

    # 策略1：提取书名号内的品类名
    categories = re.findall(r"[「『](.*?)[」』]", content)

    # 策略2：无书名号时，尝试从已知品类集合中匹配
    if not categories:
        found = []
        for cat in _KNOWN_CATEGORIES:
            if cat in content:
                found.append(cat)
        # 也尝试匹配 "XX品类" / "XX类" 模式（无书名号）
        cat_pattern_matches = re.findall(
            r"([\u4e00-\u9fa5]{2,6})(?:品类|类的|这类|那类)", content
        )
        for m in cat_pattern_matches:
            if m not in found:
                found.append(m)
        categories = found

    if not categories:
        return False

    # 判断软/硬约束
    is_soft = any(kw in content for kw in [
        "尽量不", "尽量少", "少接", "少拉", "尽量给我少", "少派",
    ])

    result.forbidden_categories.append(ForbiddenCategoryConstraint(
        categories=categories,
        is_soft=is_soft,
        penalty_per_order=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed forbidden_categories %s (soft=%s)", categories, is_soft)
    return True


def _try_parse_rest_constraint(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析连续休息约束：含'连续停车休息N小时'。"""
    # 模式1："连续停车熄火休息满8小时" / "连着停车歇满4小时"
    match = re.search(
        r"(?:连续|连着).*?(?:停车|休息|熄火|歇).*?(\d+)\s*(?:小时|个小时|h)",
        content
    )
    if not match:
        # 模式2："每天至少有一段连着停车歇满4小时"
        match = re.search(
            r"(\d+)\s*(?:小时|个小时|h).*?(?:连续|连着).*?(?:停车|休息|歇)",
            content
        )
    if not match:
        # 模式3："连续睡眠时间" / "连续休息时间" — 没有"停车"关键词
        match = re.search(
            r"(?:连续).*?(?:睡眠|休息|休整).*?(?:时间)?.*?(\d+)\s*(?:小时|个小时|h)",
            content
        )
    if not match:
        # 模式4："N个小时以上的连续睡眠"
        match = re.search(
            r"(\d+)\s*(?:小时|个小时|h).*?(?:以上)?.*?(?:连续).*?(?:睡眠|休息|休整)",
            content
        )
    if not match:
        # 模式5："歇够/歇满N个钟头" — "钟头"作为小时单位
        match = re.search(
            r"(?:停下来|停车)?.*?(?:歇够|歇满|歇上|休息够|休息满)\s*(\d+)\s*(?:个)?(?:钟头|小时|个小时|h)",
            content
        )
    if not match:
        # 模式6："N个钟头" 在含有休息语义的上下文中
        if any(kw in content for kw in ["歇", "休息", "睡", "停下来"]):
            match = re.search(
                r"(\d+)\s*(?:个)?钟头",
                content
            )
    if not match:
        return False

    min_hours = float(match.group(1))

    # 检查是否仅限平日/工作日
    weekday_only = any(kw in content for kw in ["平日", "工作日", "周一到周五"])

    result.rest_constraints.append(RestConstraint(
        min_hours=min_hours,
        weekday_only=weekday_only,
        penalty_per_day=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed rest_constraint min_hours=%.0f weekday_only=%s",
                min_hours, weekday_only)
    return True


def _try_parse_off_days(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析月度 off-day：含'N个整天不接单'或'周末不接单'。"""
    min_days: int | None = None

    # 模式0（周末限制）："周六周日全天不接单" / "周末不出车" / "每周六日不接单"
    weekend_match = re.search(
        r"(?:周六周日|周末|每周六日|周六日|礼拜六礼拜天|礼拜六日).*?(?:全天)?.*?(?:不接单|不出车|不跑|不干活|不接活|休息)",
        content
    )
    if weekend_match:
        # 3月有 8-9 个周末天，映射为 off_days min_days=8
        result.off_days.append(OffDaysConstraint(
            min_days=8,
            penalty_once=penalty_amount,
            penalty_cap=penalty_cap,
        ))
        logger.info("Rule-based: parsed off_days (weekend) min_days=8")
        return True

    # 模式1："至少要有4个整天不接单" / "至少要有2天完全歇着" / "至少要有6个完整天不出车"
    match = re.search(
        r"(?:至少|最少)?.*?(\d+)\s*(?:个)?(?:整|完整)?天.*?(?:不接单|完全歇|不出车|歇着|不接.*?不空|不出工|不干活)",
        content
    )
    if match:
        min_days = int(match.group(1))

    # 模式2："放空一整天不接单"
    if min_days is None:
        match = re.search(r"(?:放空|歇)\s*(一|\d+)\s*(?:整)?天.*?不接单", content)
        if match:
            days_str = match.group(1)
            min_days = 1 if days_str == "一" else int(days_str)

    # 模式3：倒装 — "歇个N天整" / "歇N天" + 不接活/不跑路/不出工
    if min_days is None:
        match = re.search(
            r"(?:歇个?|休息|放假|休整)\s*(\d+)\s*(?:天整|整天|天)",
            content
        )
        if match and any(kw in content for kw in [
            "不接活", "不跑路", "不出工", "不干活", "不接单", "不出车",
            "啥也不干", "什么也不干",
        ]):
            min_days = int(match.group(1))

    # 模式4："留出N个完整天不出工" / "N个完整的休息日"
    if min_days is None:
        match = re.search(
            r"(?:留出|安排|保证)\s*(\d+)\s*(?:个)?(?:完整)?(?:天|日).*?(?:不出工|不接单|不出车|休息)",
            content
        )
        if match:
            min_days = int(match.group(1))

    # 模式5："安排一个完整的休息日" — 中文数字
    if min_days is None:
        match = re.search(
            r"(?:安排|留出|保证|至少)\s*(一|两|二|三|四|五|六|七|八|九|十)\s*(?:个)?(?:完整的?)?(?:休息日|天)",
            content
        )
        if match:
            cn_val = _cn_to_number(match.group(1))
            if cn_val:
                min_days = cn_val

    if min_days is None:
        return False

    result.off_days.append(OffDaysConstraint(
        min_days=min_days,
        penalty_once=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed off_days min_days=%d", min_days)
    return True


def _extract_km_number(content: str) -> float | None:
    """从文本中提取公里数，支持阿拉伯数字和中文数字。"""
    # 先尝试阿拉伯数字
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|km|千米)", content)
    if m:
        return float(m.group(1))
    # 再尝试中文数字 + 公里
    m = re.search(r"([一二两三四五六七八九十百千]+)\s*(?:公里|km|千米)", content)
    if m:
        val = _cn_to_number(m.group(1))
        if val:
            return float(val)
    return None


def _try_parse_max_distance(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析距离限制（运距/空驶/月度空驶）。"""
    # ---- 月度空驶 ----
    # 模式1："一个月空驶赶路里程总和不得超过100公里"
    monthly_match = re.search(
        r"(?:一个月|月度|月内|自然月|本月).*?(?:空驶|赶路|空载行驶|空跑|空车).*?(?:不(?:得|能)?超过|(?:控制|限制)在)\s*(\d+)\s*(?:公里|km|千米)",
        content
    )
    if not monthly_match:
        # 模式1b："本月空载行驶累计里程严禁超过100公里"
        monthly_match = re.search(
            r"(?:一个月|月度|月内|自然月|本月).*?(?:空驶|空载行驶|空跑|空车).*?(?:严禁|禁止)?超过\s*(\d+)\s*(?:公里|km|千米)",
            content
        )
    if monthly_match:
        result.max_distances.append(MaxDistanceConstraint(
            constraint_type="monthly_deadhead",
            max_km=float(monthly_match.group(1)),
            penalty_per_violation=penalty_amount,
            penalty_cap=penalty_cap,
        ))
        logger.info("Rule-based: parsed monthly_deadhead max=%skm", monthly_match.group(1))
        return True

    # ---- 运距限制 ----
    # 模式1："装货点至卸货点距离不得超过N公里"
    haul_match = re.search(
        r"(?:装货|起点).*?(?:卸货|终点).*?(?:距离|里程)?.*?(?:不(?:得|能)?超过|(?:控制|限制)在)\s*(\d+)\s*(?:公里|km|千米)",
        content
    )
    if not haul_match:
        # 模式2："从装货地到卸货地...控制在N公里以内"
        haul_match = re.search(
            r"(?:从)?(?:装货|起点).*?(?:到|至).*?(?:卸货|终点).*?(\d+)\s*(?:公里|km|千米)\s*(?:以内|之内|以下|内)",
            content
        )
    if not haul_match:
        # 模式3："运输公里数必须在N公里之内" / "运输里程...N公里以内"
        haul_match = re.search(
            r"(?:运输|运距|运送).*?(?:公里数|里程|距离).*?(\d+)\s*(?:公里|km|千米)\s*(?:以内|之内|以下|内)",
            content
        )
    if not haul_match:
        # 模式3b：中文数字运距 — "运输公里数必须在一百五十公里之内"
        haul_cn_match = re.search(
            r"(?:运输|运距|运送).*?(?:公里数|里程|距离).*?([一二两三四五六七八九十百千]+)\s*(?:公里|km|千米)\s*(?:以内|之内|以下|内)",
            content
        )
        if haul_cn_match:
            cn_val = _cn_to_number(haul_cn_match.group(1))
            if cn_val:
                result.max_distances.append(MaxDistanceConstraint(
                    constraint_type="haul",
                    max_km=float(cn_val),
                    penalty_per_violation=penalty_amount,
                    penalty_cap=penalty_cap,
                ))
                logger.info("Rule-based: parsed max_haul max=%dkm (cn)", cn_val)
                return True
    if not haul_match:
        # 模式4："从装货点跑到卸货点不能比N公里更远"
        haul_match = re.search(
            r"(?:从)?(?:装货|起点).*?(?:到|至).*?(?:卸货|终点).*?不(?:得|能)?比\s*(\d+)\s*(?:公里|km|千米).*?(?:更远|更长)",
            content
        )
    if haul_match:
        result.max_distances.append(MaxDistanceConstraint(
            constraint_type="haul",
            max_km=float(haul_match.group(1)),
            penalty_per_violation=penalty_amount,
            penalty_cap=penalty_cap,
        ))
        logger.info("Rule-based: parsed max_haul max=%skm", haul_match.group(1))
        return True

    # ---- 空驶限制（单次） ----
    # 模式1："空驶距离不得超过N公里" / "赴装货点空驶距离"
    pickup_match = re.search(
        r"(?:空驶|赴装货点).*?(?:距离)?.*?不(?:得|能)?超过\s*(\d+)\s*(?:公里|km|千米)",
        content
    )
    if not pickup_match:
        # 模式2："空跑距离最多不能超过N公里" / "空跑不超过N公里"
        pickup_match = re.search(
            r"(?:空跑|空路程|空车).*?(?:距离)?.*?(?:不(?:得|能)?超过|最多.*?不(?:得|能)?超过)\s*(\d+)\s*(?:公里|km|千米)",
            content
        )
    if not pickup_match:
        # 模式3："去接货跑的空路程...不能超过N公里"
        pickup_match = re.search(
            r"(?:接货|去接|赶去).*?(?:空路程|空跑|空驶|空车).*?不(?:得|能)?超过\s*(\d+)\s*(?:公里|km|千米)",
            content
        )
    if not pickup_match:
        # 模式4："空驶/空跑/空路程...控制在N公里以内"
        pickup_match = re.search(
            r"(?:空驶|空跑|空路程|空车).*?(?:控制|限制)在\s*(\d+)\s*(?:公里|km|千米)\s*(?:以内|之内|内)",
            content
        )
    if pickup_match:
        result.max_distances.append(MaxDistanceConstraint(
            constraint_type="pickup",
            max_km=float(pickup_match.group(1)),
            penalty_per_violation=penalty_amount,
            penalty_cap=penalty_cap,
        ))
        logger.info("Rule-based: parsed max_pickup max=%skm", pickup_match.group(1))
        return True

    # ---- 噪声文本兜底：中英混杂格式 ----
    # 模式："max运距limit=120km" / "distance_type=haul" / "运距limit=Nkm"
    noisy_haul = re.search(
        r"(?:max)?\s*运距\s*(?:limit)?\s*[=:：]?\s*(\d+)\s*(?:km|公里|千米)",
        content, re.IGNORECASE
    )
    if noisy_haul:
        # 判断类型：默认 haul，如果文本含 pickup/空驶 则为 pickup
        ctype = "haul"
        if re.search(r"(?:pickup|空驶|空跑|赴装货)", content, re.IGNORECASE):
            ctype = "pickup"
        elif re.search(r"(?:monthly|月度|月内|累计)", content, re.IGNORECASE):
            ctype = "monthly_deadhead"
        # 也检查 distance_type= 显式声明
        dtype_match = re.search(r"distance_type\s*=\s*(\w+)", content, re.IGNORECASE)
        if dtype_match:
            dt = dtype_match.group(1).lower()
            if dt in ("haul", "pickup", "monthly_deadhead"):
                ctype = dt
        result.max_distances.append(MaxDistanceConstraint(
            constraint_type=ctype,
            max_km=float(noisy_haul.group(1)),
            penalty_per_violation=penalty_amount,
            penalty_cap=penalty_cap,
        ))
        logger.info("Rule-based: parsed max_distance (noisy) type=%s max=%skm", ctype, noisy_haul.group(1))
        return True

    return False

def _try_parse_max_orders(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析每日接单上限：含'接单不超过N单'。"""
    match = re.search(
        r"(?:同一天|每天|每日|一天)?.*?接单.*?不(?:得|能)?超过\s*(\d+)\s*单",
        content
    )
    if not match:
        # 模式2："日接单量上限N单" / "接单上限N单" / "最多接N单"
        match = re.search(
            r"(?:日|每天|每日)?\s*接单.*?(?:上限|最多|不超)\s*(\d+)\s*单",
            content
        )
    if not match:
        return False

    max_per_day = int(match.group(1))
    result.max_orders.append(MaxOrdersConstraint(
        max_per_day=max_per_day,
        penalty_per_extra=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed max_orders %d/day", max_per_day)
    return True


def _try_parse_first_order_deadline(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析首单时间限制：含'首单不晚于N点'。"""
    # 模式1："首单不晚于N点"
    match = re.search(
        r"首单.*?不(?:得|能)?晚于.*?(\d{1,2})\s*点",
        content
    )
    if not match:
        # 模式2："首单开工时间不得超过中午12点" / "首单不得超过N点"
        match = re.search(
            r"首单.*?不(?:得|能)?超过.*?(?:中午|上午|下午|晚上|凌晨)?\s*(\d{1,2})\s*点",
            content
        )
    if not match:
        # 模式3："首单必须在上午9点前开工" / "首单在N点前"
        match = re.search(
            r"首单.*?(?:必须)?在.*?(?:中午|上午|下午|晚上|凌晨)?\s*(\d{1,2})\s*点\s*(?:前|之前).*?(?:开工|出发|接单|完成)",
            content
        )
    if not match:
        # 模式4："首单...N点前" 更宽松
        match = re.search(
            r"首单.*?(\d{1,2})\s*点\s*(?:前|之前)",
            content
        )
    if not match:
        return False

    deadline_hour = int(match.group(1))
    result.first_order_deadline.append(FirstOrderDeadlineConstraint(
        deadline_hour=deadline_hour,
        penalty_per_day=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed first_order_deadline %d:00", deadline_hour)
    return True


def _try_parse_geo_fence(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析地理围栏（矩形）：含'北纬A至B，东经C至D'。"""
    # 模式1："北纬A至B，东经C至D"
    match = re.search(
        r"北纬\s*([\d.]+)\s*至\s*([\d.]+).*?东经\s*([\d.]+)\s*至\s*([\d.]+)",
        content
    )
    if not match:
        # 模式2："纬度A到B、经度C到D" / "纬度A至B，经度C至D"
        match = re.search(
            r"纬度\s*([\d.]+)\s*(?:到|至|~|-|—)\s*([\d.]+)[，,、\s]+经度\s*([\d.]+)\s*(?:到|至|~|-|—)\s*([\d.]+)",
            content
        )
    if not match:
        # 模式3（噪声文本）："geo围栏：lat∈[A,B], lng∈[C,D]" / "lat=[A,B]" 等中英混杂
        match = re.search(
            r"lat\s*[∈=:：]\s*\[?\s*([\d.]+)\s*[,，]\s*([\d.]+)\s*\]?\s*[,，;；\s]+\s*lng\s*[∈=:：]\s*\[?\s*([\d.]+)\s*[,，]\s*([\d.]+)\s*\]?",
            content, re.IGNORECASE
        )
    if not match:
        return False

    result.geo_fences.append(GeoFenceConstraint(
        lat_min=float(match.group(1)),
        lat_max=float(match.group(2)),
        lng_min=float(match.group(3)),
        lng_max=float(match.group(4)),
        penalty_once=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed geo_fence lat[%.2f,%.2f] lng[%.2f,%.2f]",
                float(match.group(1)), float(match.group(2)),
                float(match.group(3)), float(match.group(4)))
    return True


def _try_parse_forbidden_zone(
    content: str, penalty_amount: float, penalty_cap: float | None,
    result: ParsedPreferences,
) -> bool:
    """解析禁入区域（圆形）：含'以（lat，lng）为圆心、半径R公里'。"""
    # 模式1："以（lat，lng）为圆心、半径R公里"
    match = re.search(
        r"以[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)].*?(?:为)?圆心.*?半径\s*([\d.]+)\s*(?:公里|km|千米)",
        content
    )
    if not match:
        # 模式2："以坐标（lat，lng）为中心、半径R公里" — 多了"坐标"两字
        match = re.search(
            r"以\s*(?:坐标)?\s*[（(]\s*([\d.]+)[，,]\s*([\d.]+)\s*[）)]\s*(?:为)?(?:圆心|中心).*?半径\s*([\d.]+)\s*(?:公里|km|千米)",
            content
        )
    if not match:
        return False

    result.forbidden_zones.append(ForbiddenZoneConstraint(
        center_lat=float(match.group(1)),
        center_lng=float(match.group(2)),
        radius_km=float(match.group(3)),
        penalty_per_entry=penalty_amount,
        penalty_cap=penalty_cap,
    ))
    logger.info("Rule-based: parsed forbidden_zone center=(%.2f,%.2f) radius=%.1fkm",
                float(match.group(1)), float(match.group(2)), float(match.group(3)))
    return True


# ===========================================================================
# 主入口
# ===========================================================================

def rule_based_parse(driver_status: dict[str, Any]) -> ParsedPreferences:
    """规则化偏好解析器：不依赖 LLM，用正则和关键词从偏好原文提取结构化约束。

    当 LLM 不可用时作为保底方案调用。能覆盖全部 14 种已知偏好类型。
    无法识别的偏好仍放入 custom 列表，由后续逐步 LLM 评估。
    """
    driver_id = str(driver_status.get("driver_id", ""))
    preferences = driver_status.get("preferences", [])

    result = ParsedPreferences(
        driver_id=driver_id,
        cost_per_km=float(driver_status.get("cost_per_km", 1.5)),
        initial_lat=float(driver_status.get("current_lat", 0.0)),
        initial_lng=float(driver_status.get("current_lng", 0.0)),
    )

    if not preferences:
        return result

    for p in preferences:
        content = str(p.get("content", ""))
        penalty_amount = float(p.get("penalty_amount", 0))
        penalty_cap = _safe_cap_value(p.get("penalty_cap"))

        parsed = False

        # 按优先级尝试各解析器（高罚分的优先）
        if not parsed:
            parsed = _try_parse_family_event(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_special_cargo(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_go_home(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_visit_target(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_quiet_window(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_forbidden_category(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_rest_constraint(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_off_days(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_max_distance(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_max_orders(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_first_order_deadline(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_geo_fence(content, penalty_amount, penalty_cap, result)
        if not parsed:
            parsed = _try_parse_forbidden_zone(content, penalty_amount, penalty_cap, result)

        # 无法识别 → custom 兜底
        if not parsed:
            result.custom.append(CustomConstraint(
                original_text=content,
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            ))
            logger.warning("Rule-based: unrecognized → custom: %.60s...", content)

    recognized = (
        len(result.rest_constraints) + len(result.quiet_windows)
        + len(result.forbidden_categories) + len(result.max_distances)
        + len(result.max_orders) + len(result.first_order_deadline)
        + len(result.off_days) + len(result.geo_fences)
        + len(result.forbidden_zones) + len(result.go_home)
        + len(result.special_cargos) + len(result.family_events)
        + len(result.visit_targets)
    )
    logger.info(
        "Rule-based fallback for %s: %d/%d recognized, %d custom",
        driver_id, recognized, len(preferences), len(result.custom),
    )
    return result
