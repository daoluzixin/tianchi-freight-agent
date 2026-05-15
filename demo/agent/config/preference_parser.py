"""偏好解析器：用一次 LLM 调用将司机的自然语言偏好解析为结构化约束。

核心设计：
  1. 首次 decide() 时调用，解析结果缓存全月复用
  2. 定义了覆盖所有已知偏好类型的 JSON Schema
  3. 对未知类型的偏好，解析为通用的 "custom" 约束，交由 LLM 每步判断
  4. 单次 LLM 调用约消耗 2000-4000 token，占 500 万预算的 <0.1%
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agent.preference_parser")


# ===========================================================================
# 结构化约束 Schema（覆盖所有可能的偏好类型）
# ===========================================================================

@dataclass
class RestConstraint:
    """连续休息约束：每天/每周需连续休息 N 小时。"""
    min_hours: float = 0.0
    weekday_only: bool = False  # 仅工作日
    penalty_per_day: float = 0.0
    penalty_cap: float | None = None


@dataclass
class QuietWindowConstraint:
    """禁止活动时段：某个时间段内不接单不空驶。"""
    start_hour: int = 0
    start_minute: int = 0
    end_hour: int = 0
    end_minute: int = 0
    penalty_per_day: float = 0.0
    penalty_cap: float | None = None


@dataclass
class ForbiddenCategoryConstraint:
    """禁止品类。"""
    categories: list[str] = field(default_factory=list)
    is_soft: bool = False  # "尽量不" = soft
    penalty_per_order: float = 0.0
    penalty_cap: float | None = None


@dataclass
class MaxDistanceConstraint:
    """距离限制（运距/空驶/月度空驶）。"""
    constraint_type: str = ""  # "haul" | "pickup" | "monthly_deadhead"
    max_km: float = 999999.0
    penalty_per_violation: float = 0.0  # 对 monthly_deadhead 是按 km 计罚
    penalty_cap: float | None = None


@dataclass
class MaxOrdersConstraint:
    """每日最大接单数。"""
    max_per_day: int = 999
    penalty_per_extra: float = 0.0
    penalty_cap: float | None = None


@dataclass
class FirstOrderDeadlineConstraint:
    """接单日首单时间限制。"""
    deadline_hour: int = 24
    penalty_per_day: float = 0.0
    penalty_cap: float | None = None


@dataclass
class OffDaysConstraint:
    """月度 off-day 要求。"""
    min_days: int = 0
    penalty_once: float = 0.0
    penalty_cap: float | None = None


@dataclass
class GeoFenceConstraint:
    """地理围栏：始终在某区域内。"""
    lat_min: float = -90.0
    lat_max: float = 90.0
    lng_min: float = -180.0
    lng_max: float = 180.0
    penalty_once: float = 0.0
    penalty_cap: float | None = None


@dataclass
class ForbiddenZoneConstraint:
    """禁入区域：不得进入某圆形范围。"""
    center_lat: float = 0.0
    center_lng: float = 0.0
    radius_km: float = 0.0
    penalty_per_entry: float = 0.0
    penalty_cap: float | None = None


@dataclass
class GoHomeConstraint:
    """每天回家约束。"""
    home_lat: float = 0.0
    home_lng: float = 0.0
    deadline_hour: int = 23
    quiet_start_hour: int = 23
    quiet_end_hour: int = 8
    radius_km: float = 1.0
    penalty_per_day: float = 0.0
    penalty_cap: float | None = None


@dataclass
class SpecialCargoConstraint:
    """必接特殊货源。"""
    cargo_id: str = ""
    available_from: str = ""  # 时间字符串
    pickup_lat: float = 0.0
    pickup_lng: float = 0.0
    penalty_if_missed: float = 0.0


@dataclass
class FamilyEventConstraint:
    """家事/特殊事件约束。"""
    trigger_time: str = ""  # "2026-03-10 10:00:00"
    waypoints: list[dict[str, Any]] = field(default_factory=list)  # 途经点和等待
    home_lat: float = 0.0
    home_lng: float = 0.0
    home_deadline: str = ""
    stay_until: str = ""
    penalty_per_minute_late: float = 0.0
    penalty_once_if_failed: float = 0.0
    penalty_cap: float | None = None


@dataclass
class VisitTargetConstraint:
    """月度到访目标点。"""
    target_lat: float = 0.0
    target_lng: float = 0.0
    radius_km: float = 1.0
    min_days: int = 0
    penalty_once: float = 0.0
    penalty_cap: float | None = None


@dataclass
class CustomConstraint:
    """兜底：无法自动分类的自定义约束，每步需 LLM 判断。"""
    original_text: str = ""
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class ParsedPreferences:
    """一位司机的所有解析后约束。"""
    driver_id: str = ""
    cost_per_km: float = 1.5
    initial_lat: float = 0.0
    initial_lng: float = 0.0

    rest_constraints: list[RestConstraint] = field(default_factory=list)
    quiet_windows: list[QuietWindowConstraint] = field(default_factory=list)
    forbidden_categories: list[ForbiddenCategoryConstraint] = field(default_factory=list)
    max_distances: list[MaxDistanceConstraint] = field(default_factory=list)
    max_orders: list[MaxOrdersConstraint] = field(default_factory=list)
    first_order_deadline: list[FirstOrderDeadlineConstraint] = field(default_factory=list)
    off_days: list[OffDaysConstraint] = field(default_factory=list)
    geo_fences: list[GeoFenceConstraint] = field(default_factory=list)
    forbidden_zones: list[ForbiddenZoneConstraint] = field(default_factory=list)
    go_home: list[GoHomeConstraint] = field(default_factory=list)
    special_cargos: list[SpecialCargoConstraint] = field(default_factory=list)
    family_events: list[FamilyEventConstraint] = field(default_factory=list)
    visit_targets: list[VisitTargetConstraint] = field(default_factory=list)
    custom: list[CustomConstraint] = field(default_factory=list)

    def has_hard_quiet_window(self) -> bool:
        return len(self.quiet_windows) > 0

    def has_go_home(self) -> bool:
        return len(self.go_home) > 0

    def has_family_event(self) -> bool:
        return len(self.family_events) > 0


# ===========================================================================
# LLM 解析 Prompt
# ===========================================================================

PARSE_SYSTEM_PROMPT = """你是货运偏好解析引擎。将司机的自然语言偏好规则解析为结构化JSON。

输出格式要求：一个JSON对象，包含以下可选字段（没有的不填）：

{
  "rest_constraints": [{"min_hours": 8, "weekday_only": false, "penalty_per_day": 300, "penalty_cap": 3000}],
  "quiet_windows": [{"start_hour": 23, "start_minute": 0, "end_hour": 6, "end_minute": 0, "penalty_per_day": 200, "penalty_cap": 6000}],
  "forbidden_categories": [{"categories": ["化工塑料"], "is_soft": false, "penalty_per_order": 500, "penalty_cap": 5000}],
  "max_distances": [{"constraint_type": "haul|pickup|monthly_deadhead", "max_km": 100, "penalty_per_violation": 100, "penalty_cap": null}],
  "max_orders": [{"max_per_day": 3, "penalty_per_extra": 200, "penalty_cap": null}],
  "first_order_deadline": [{"deadline_hour": 12, "penalty_per_day": 200, "penalty_cap": 4000}],
  "off_days": [{"min_days": 4, "penalty_once": 6000, "penalty_cap": 6000}],
  "geo_fences": [{"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66, "penalty_once": 2000, "penalty_cap": 2000}],
  "forbidden_zones": [{"center_lat": 23.30, "center_lng": 113.52, "radius_km": 20, "penalty_per_entry": 1000, "penalty_cap": 10000}],
  "go_home": [{"home_lat": 23.12, "home_lng": 113.28, "deadline_hour": 23, "quiet_start_hour": 23, "quiet_end_hour": 8, "radius_km": 1.0, "penalty_per_day": 900, "penalty_cap": 27000}],
  "special_cargos": [{"cargo_id": "240646", "available_from": "2026-03-03 14:43:36", "pickup_lat": 24.81, "pickup_lng": 113.58, "penalty_if_missed": 10000}],
  "family_events": [{"trigger_time": "2026-03-10 10:00:00", "waypoints": [{"lat": 23.21, "lng": 113.37, "wait_minutes": 10}], "home_lat": 23.19, "home_lng": 113.36, "home_deadline": "2026-03-10 22:00:00", "stay_until": "2026-03-13 22:00:00", "penalty_per_minute_late": 5, "penalty_once_if_failed": 9000, "penalty_cap": null}],
  "visit_targets": [{"target_lat": 23.13, "target_lng": 113.26, "radius_km": 1.0, "min_days": 5, "penalty_once": 3000, "penalty_cap": 3000}],
  "custom": [{"original_text": "...", "penalty_amount": 100, "penalty_cap": 1000}]
}

解析规则：
1. "不接/禁止接" → forbidden_categories (is_soft=false)
2. "尽量不/少接" → forbidden_categories (is_soft=true)
3. "每天X点至Y点不接单不空驶" → quiet_windows
4. "连续休息/停车N小时" → rest_constraints
5. "距离不超过N公里" → max_distances (根据上下文判断 haul/pickup/monthly_deadhead)
6. "每天接单不超过N单" → max_orders
7. "首单不晚于" → first_order_deadline
8. "N天完全不出车/休息" → off_days
9. "不出某区域/位置范围" → geo_fences
10. "不进入某区域" → forbidden_zones
11. "每天X点前回家/到家" → go_home
12. "指定货源/熟货必接" → special_cargos
13. "某日某事需先去接人再回家" → family_events
14. "每月N天到某点" → visit_targets
15. 无法归类的 → custom

penalty_cap 为 null 表示无上限。只输出JSON，不输出解释。"""


def build_parse_prompt(driver_status: dict[str, Any]) -> str:
    """构造用户侧 prompt：包含司机信息和偏好文本。"""
    preferences = driver_status.get("preferences", [])
    if not preferences:
        return json.dumps({"driver_id": driver_status.get("driver_id"), "preferences": []}, ensure_ascii=False)

    pref_items = []
    for p in preferences:
        pref_items.append({
            "content": p.get("content", ""),
            "penalty_amount": p.get("penalty_amount", 0),
            "penalty_cap": p.get("penalty_cap"),
        })

    data = {
        "driver_id": driver_status.get("driver_id"),
        "cost_per_km": driver_status.get("cost_per_km", 1.5),
        "current_lat": driver_status.get("current_lat"),
        "current_lng": driver_status.get("current_lng"),
        "preferences": pref_items,
    }
    return json.dumps(data, ensure_ascii=False)


# ===========================================================================
# 解析结果转换
# ===========================================================================

def parse_llm_response(driver_status: dict[str, Any], llm_output: str) -> ParsedPreferences:
    """将 LLM 返回的 JSON 字符串转换为 ParsedPreferences 对象。"""
    result = ParsedPreferences(
        driver_id=str(driver_status.get("driver_id", "")),
        cost_per_km=float(driver_status.get("cost_per_km", 1.5)),
        initial_lat=float(driver_status.get("current_lat", 0.0)),
        initial_lng=float(driver_status.get("current_lng", 0.0)),
    )

    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM output as JSON: %s — using rule-based fallback", e)
        return _rule_based_fallback(driver_status)

    if not isinstance(data, dict):
        logger.error("LLM output is not a dict — using rule-based fallback")
        return _rule_based_fallback(driver_status)

    # 解析各类约束
    for item in data.get("rest_constraints", []):
        result.rest_constraints.append(RestConstraint(
            min_hours=float(item.get("min_hours", 0)),
            weekday_only=bool(item.get("weekday_only", False)),
            penalty_per_day=float(item.get("penalty_per_day", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("quiet_windows", []):
        result.quiet_windows.append(QuietWindowConstraint(
            start_hour=int(item.get("start_hour", 0)),
            start_minute=int(item.get("start_minute", 0)),
            end_hour=int(item.get("end_hour", 0)),
            end_minute=int(item.get("end_minute", 0)),
            penalty_per_day=float(item.get("penalty_per_day", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("forbidden_categories", []):
        result.forbidden_categories.append(ForbiddenCategoryConstraint(
            categories=list(item.get("categories", [])),
            is_soft=bool(item.get("is_soft", False)),
            penalty_per_order=float(item.get("penalty_per_order", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("max_distances", []):
        result.max_distances.append(MaxDistanceConstraint(
            constraint_type=str(item.get("constraint_type", "")),
            max_km=float(item.get("max_km", 999999)),
            penalty_per_violation=float(item.get("penalty_per_violation", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("max_orders", []):
        result.max_orders.append(MaxOrdersConstraint(
            max_per_day=int(item.get("max_per_day", 999)),
            penalty_per_extra=float(item.get("penalty_per_extra", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("first_order_deadline", []):
        result.first_order_deadline.append(FirstOrderDeadlineConstraint(
            deadline_hour=int(item.get("deadline_hour", 24)),
            penalty_per_day=float(item.get("penalty_per_day", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("off_days", []):
        result.off_days.append(OffDaysConstraint(
            min_days=int(item.get("min_days", 0)),
            penalty_once=float(item.get("penalty_once", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("geo_fences", []):
        result.geo_fences.append(GeoFenceConstraint(
            lat_min=float(item.get("lat_min", -90)),
            lat_max=float(item.get("lat_max", 90)),
            lng_min=float(item.get("lng_min", -180)),
            lng_max=float(item.get("lng_max", 180)),
            penalty_once=float(item.get("penalty_once", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("forbidden_zones", []):
        result.forbidden_zones.append(ForbiddenZoneConstraint(
            center_lat=float(item.get("center_lat", 0)),
            center_lng=float(item.get("center_lng", 0)),
            radius_km=float(item.get("radius_km", 0)),
            penalty_per_entry=float(item.get("penalty_per_entry", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("go_home", []):
        result.go_home.append(GoHomeConstraint(
            home_lat=float(item.get("home_lat", 0)),
            home_lng=float(item.get("home_lng", 0)),
            deadline_hour=int(item.get("deadline_hour", 23)),
            quiet_start_hour=int(item.get("quiet_start_hour", 23)),
            quiet_end_hour=int(item.get("quiet_end_hour", 8)),
            radius_km=float(item.get("radius_km", 1.0)),
            penalty_per_day=float(item.get("penalty_per_day", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("special_cargos", []):
        result.special_cargos.append(SpecialCargoConstraint(
            cargo_id=str(item.get("cargo_id", "")),
            available_from=str(item.get("available_from", "")),
            pickup_lat=float(item.get("pickup_lat", 0)),
            pickup_lng=float(item.get("pickup_lng", 0)),
            penalty_if_missed=float(item.get("penalty_if_missed", 0)),
        ))

    for item in data.get("family_events", []):
        result.family_events.append(FamilyEventConstraint(
            trigger_time=str(item.get("trigger_time", "")),
            waypoints=list(item.get("waypoints", [])),
            home_lat=float(item.get("home_lat", 0)),
            home_lng=float(item.get("home_lng", 0)),
            home_deadline=str(item.get("home_deadline", "")),
            stay_until=str(item.get("stay_until", "")),
            penalty_per_minute_late=float(item.get("penalty_per_minute_late", 0)),
            penalty_once_if_failed=float(item.get("penalty_once_if_failed", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("visit_targets", []):
        result.visit_targets.append(VisitTargetConstraint(
            target_lat=float(item.get("target_lat", 0)),
            target_lng=float(item.get("target_lng", 0)),
            radius_km=float(item.get("radius_km", 1.0)),
            min_days=int(item.get("min_days", 0)),
            penalty_once=float(item.get("penalty_once", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    for item in data.get("custom", []):
        result.custom.append(CustomConstraint(
            original_text=str(item.get("original_text", "")),
            penalty_amount=float(item.get("penalty_amount", 0)),
            penalty_cap=_safe_cap(item.get("penalty_cap")),
        ))

    return result


def _safe_cap(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _fallback_to_custom(result: ParsedPreferences, driver_status: dict[str, Any]) -> None:
    """解析失败时，将所有偏好放入 custom 兜底。"""
    preferences = driver_status.get("preferences", [])
    for p in preferences:
        result.custom.append(CustomConstraint(
            original_text=str(p.get("content", "")),
            penalty_amount=float(p.get("penalty_amount", 0)),
            penalty_cap=_safe_cap(p.get("penalty_cap")),
        ))


def _rule_based_fallback(driver_status: dict[str, Any]) -> ParsedPreferences:
    """延迟导入 rule_based_parser 以避免循环导入。"""
    from agent.config.rule_based_parser import rule_based_parse
    return rule_based_parse(driver_status)


# ===========================================================================
# 主入口：调用 LLM 解析偏好
# ===========================================================================

class PreferenceParser:
    """偏好解析器：首次调用时通过 LLM 解析，结果缓存复用。"""

    def __init__(self) -> None:
        self._cache: dict[str, ParsedPreferences] = {}

    def get_parsed(self, driver_id: str) -> ParsedPreferences | None:
        """获取已缓存的解析结果。"""
        return self._cache.get(driver_id)

    def parse(self, driver_status: dict[str, Any], api: Any) -> ParsedPreferences:
        """解析司机偏好。如已缓存则直接返回。

        Args:
            driver_status: get_driver_status 的返回值（含 preferences 字段）
            api: SimulationApiPort 实例（用于调 model_chat_completion）
        """
        driver_id = str(driver_status.get("driver_id", ""))
        if driver_id in self._cache:
            return self._cache[driver_id]

        preferences = driver_status.get("preferences", [])

        # 无偏好的司机：返回空约束
        if not preferences:
            result = ParsedPreferences(
                driver_id=driver_id,
                cost_per_km=float(driver_status.get("cost_per_km", 1.5)),
                initial_lat=float(driver_status.get("current_lat", 0.0)),
                initial_lng=float(driver_status.get("current_lng", 0.0)),
            )
            self._cache[driver_id] = result
            return result

        # 调用 LLM 解析
        user_prompt = build_parse_prompt(driver_status)
        logger.info("Parsing preferences for %s (%d rules)", driver_id, len(preferences))

        try:
            resp = api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            })
            choices = resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                result = parse_llm_response(driver_status, content)
            else:
                logger.warning("LLM returned empty choices for %s — using rule-based fallback", driver_id)
                result = _rule_based_fallback(driver_status)
        except Exception as e:
            logger.error("LLM parse failed for %s: %s — using rule-based fallback", driver_id, e)
            result = _rule_based_fallback(driver_status)

        # 完整性校验：对高罚分关键约束做 fallback 补全
        # 如果偏好文本中包含关键词但 LLM 解析结果缺失，用 rule_based_parser 补充
        result = self._patch_missing_critical(result, driver_status)

        self._cache[driver_id] = result
        logger.info("Parsed %s: rest=%d, quiet=%d, forbidden_cat=%d, max_dist=%d, "
                    "go_home=%d, special=%d, family=%d, visit=%d, custom=%d",
                    driver_id,
                    len(result.rest_constraints),
                    len(result.quiet_windows),
                    len(result.forbidden_categories),
                    len(result.max_distances),
                    len(result.go_home),
                    len(result.special_cargos),
                    len(result.family_events),
                    len(result.visit_targets),
                    len(result.custom))
        return result

    @staticmethod
    def _patch_missing_critical(result: ParsedPreferences,
                                driver_status: dict[str, Any]) -> ParsedPreferences:
        """LLM 解析后补全遗漏的高罚分约束。

        对 family_events、special_cargos、go_home 等高罚分约束做关键词检测：
        如果偏好文本中包含对应关键词但解析结果为空，则用 rule_based_parser 补充。
        这是泛化防护，不针对任何特定 driver_id。
        """
        preferences = driver_status.get("preferences", [])
        all_text = " ".join(str(p.get("content", "")) for p in preferences)

        needs_patch = False
        # 家事约束：高罚分（9000+），必须正确解析
        if not result.family_events and ("家事" in all_text or "临时约定·家事" in all_text):
            needs_patch = True
        # 特殊货源：高罚分（10000），必须正确解析
        # "临时约定"可能关联熟货/指定货源，需同时检查
        if not result.special_cargos and (
            "指定货源" in all_text or "必接" in all_text
            or "熟货" in all_text or "临时约定" in all_text
        ):
            needs_patch = True
        # 每日回家：高罚分，必须正确解析
        if not result.go_home and ("回家" in all_text or "到家" in all_text or "进家门" in all_text):
            needs_patch = True

        if not needs_patch:
            return result

        logger.warning("LLM parse missed critical constraints — patching with rule-based parser")
        from agent.config.rule_based_parser import rule_based_parse
        rb_result = rule_based_parse(driver_status)

        # 只补充缺失的高罚分约束，不覆盖 LLM 已正确解析的部分
        if not result.family_events and rb_result.family_events:
            result.family_events = rb_result.family_events
            logger.info("Patched: added %d family_events from rule-based parser",
                        len(rb_result.family_events))
        if not result.special_cargos and rb_result.special_cargos:
            result.special_cargos = rb_result.special_cargos
            logger.info("Patched: added %d special_cargos from rule-based parser",
                        len(rb_result.special_cargos))
        if not result.go_home and rb_result.go_home:
            result.go_home = rb_result.go_home
            logger.info("Patched: added %d go_home from rule-based parser",
                        len(rb_result.go_home))

        return result
