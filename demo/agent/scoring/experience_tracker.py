"""决策经验追踪器：纯规则实现的自我改进系统（零 Token 消耗）。

核心思想：
  在仿真过程中自动积累 (决策上下文, 实际结果) 对，按 (time_slot, region) 聚类存储。
  决策时检索同类历史经验，用于校准 position_bonus 和 wait_value。

理论来源：
  - Self-Generated In-Context Examples (NeurIPS 2025)
  - SLEA-RL: Step-Level Experience Augmented RL (2025)

设计要点：
  1. 经验从决策结果自动提取，不调用 LLM
  2. 延迟回填：决策时记录上下文，下一步拿到结果后回填
  3. 自适应衰减：高 confirm_count 的经验衰减慢（泛用规律），低 confirm_count 衰减快（偶发观察）
  4. 冷启动保护：前 3 天只积累不输出
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agent.experience_tracker")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 时段桶定义
_TIME_SLOTS = {
    "night": 0,       # 0:00 - 6:00
    "morning": 1,     # 6:00 - 11:00
    "midday": 2,      # 11:00 - 14:00
    "afternoon": 3,   # 14:00 - 20:00
    "evening": 4,     # 20:00 - 24:00
}

# 区域桶粒度（度）
_REGION_GRID_SIZE = 0.5

# 每个桶最多保留的经验数
_MAX_EXPERIENCES_PER_BUCKET = 20

# 冷启动天数：前 N 天只积累不输出
_COLD_START_DAYS = 3

# 置信度门槛：至少 N 条已落定经验才输出建议
_MIN_CONFIDENCE_COUNT = 3

# 自适应衰减率
_DECAY_RATES = {
    "stable": 0.98,    # confirm_count >= 3
    "emerging": 0.95,  # confirm_count == 2
    "tentative": 0.90, # confirm_count <= 1
}

# 相似结果判定阈值（收入差异比例）
_SIMILAR_INCOME_RATIO = 0.3


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DecisionExperience:
    """一条决策经验记录。"""

    # 上下文（决策时记录）
    time_slot: int                          # 时段桶
    region_key: tuple[int, int]             # 决策时位置的区域桶
    cargo_price: float = 0.0
    pickup_km: float = 0.0
    score_at_decision: float = 0.0

    # 结果（延迟回填）
    actual_income: float = 0.0
    next_order_wait_minutes: float = 0.0    # 卸货后等了多久才接到下一单
    delivery_region_key: tuple[int, int] = (0, 0)

    # 元数据
    day: int = 0
    weight: float = 1.0
    confirm_count: int = 1
    settled: bool = False                   # 结果是否已回填


@dataclass
class ExperienceSummary:
    """某个 (time_slot, region) 桶的经验摘要统计。"""
    count: int = 0
    avg_income: float = 0.0
    avg_next_wait: float = 0.0
    avg_score: float = 0.0
    confidence: float = 0.0   # 0~1，基于样本量和 confirm_count


@dataclass
class DeliveryRegionSummary:
    """某个卸货区域在特定时段的经验摘要。"""
    count: int = 0
    avg_income: float = 0.0
    avg_next_wait: float = 0.0
    confidence: float = 0.0


@dataclass
class BestExperienceDetail:
    """桶内收入最高的一条经验详情，用于 few-shot 注入。"""
    actual_income: float = 0.0
    cargo_price: float = 0.0
    pickup_km: float = 0.0
    score_at_decision: float = 0.0
    next_order_wait_minutes: float = 0.0
    delivery_region_key: tuple[int, int] = (0, 0)
    time_slot: int = 0
    day: int = 0
    confidence: float = 0.0  # R10-A-v2: 综合置信度
    net_income: float = 0.0  # R10-A-v2: 净收益 = actual_income - pickup_km * 2.0


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def hour_to_time_slot(hour: float) -> int:
    """将小时数映射到时段桶。"""
    if hour < 6:
        return 0   # night
    elif hour < 11:
        return 1   # morning
    elif hour < 14:
        return 2   # midday
    elif hour < 20:
        return 3   # afternoon
    else:
        return 4   # evening


def pos_to_region_key(lat: float, lng: float) -> tuple[int, int]:
    """将经纬度映射到区域桶。"""
    return (int(lat / _REGION_GRID_SIZE), int(lng / _REGION_GRID_SIZE))


def _get_decay_rate(confirm_count: int) -> float:
    """根据 confirm_count 返回衰减率。"""
    if confirm_count >= 3:
        return _DECAY_RATES["stable"]
    elif confirm_count == 2:
        return _DECAY_RATES["emerging"]
    else:
        return _DECAY_RATES["tentative"]


# ---------------------------------------------------------------------------
# ExperienceTracker
# ---------------------------------------------------------------------------

class ExperienceTracker:
    """决策经验追踪器：按 (time_slot, region) 聚类存储和检索经验。

    每个司机维护独立的经验库。支持：
      - record_decision(): 决策时记录待确认经验
      - settle_pending(): 拿到结果后回填
      - query_experience(): 检索同类历史经验的摘要
      - query_delivery_region(): 检索卸货区域的历史表现
      - daily_decay(): 每日衰减
    """

    def __init__(self) -> None:
        # driver_id → {(time_slot, region_key) → [DecisionExperience]}
        self._index: dict[str, dict[tuple[int, tuple[int, int]], list[DecisionExperience]]] = {}

        # driver_id → 卸货区域索引 {(time_slot, delivery_region_key) → [DecisionExperience]}
        self._delivery_index: dict[str, dict[tuple[int, tuple[int, int]], list[DecisionExperience]]] = {}

        # driver_id → 待回填经验（最近一次 take_order）
        self._pending: dict[str, DecisionExperience] = {}

        # driver_id → 上一次 take_order 完成时的 sim_minutes（用于计算 next_order_wait）
        self._last_order_complete_min: dict[str, int] = {}

        # 衰减追踪
        self._last_decay_day: dict[str, int] = {}

        # R8: 等待决策历史
        self._wait_history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # 记录决策
    # ------------------------------------------------------------------

    def record_decision(
        self,
        driver_id: str,
        sim_minutes: int,
        lat: float,
        lng: float,
        cargo_price: float,
        pickup_km: float,
        score: float,
        day: int,
    ) -> None:
        """在做出 take_order 决策时记录待确认经验。"""
        hour = (sim_minutes % 1440) / 60.0
        time_slot = hour_to_time_slot(hour)
        region_key = pos_to_region_key(lat, lng)

        exp = DecisionExperience(
            time_slot=time_slot,
            region_key=region_key,
            cargo_price=cargo_price,
            pickup_km=pickup_km,
            score_at_decision=score,
            day=day,
            weight=1.0,
            confirm_count=1,
            settled=False,
        )

        self._pending[driver_id] = exp
        logger.debug(
            "recorded pending experience: driver=%s slot=%d region=%s price=%.0f",
            driver_id, time_slot, region_key, cargo_price)

    # ------------------------------------------------------------------
    # 回填结果
    # ------------------------------------------------------------------

    def settle_pending(
        self,
        driver_id: str,
        actual_income: float,
        delivery_lat: float,
        delivery_lng: float,
        current_sim_minutes: int,
    ) -> None:
        """拿到 take_order 的执行结果后，回填待确认经验并入库。"""
        if driver_id not in self._pending:
            return

        exp = self._pending.pop(driver_id)

        # 计算"下一单等待时间"：从上一单完成到这一单开始的间隔
        last_complete = self._last_order_complete_min.get(driver_id)
        if last_complete is not None and last_complete > 0:
            # 本次决策的 sim_minutes 近似为"上一单完成后等待的结束时刻"
            # 实际 next_order_wait 是从上次卸货到本次接单决策的时间差
            # 但因为 settle 是在下一次 decide 时调用，current_sim_minutes 是当前步开始时刻
            # 我们用 current_sim_minutes 与 last_complete 的差值作为近似
            exp.next_order_wait_minutes = max(0, current_sim_minutes - last_complete)

        exp.actual_income = actual_income
        exp.delivery_region_key = pos_to_region_key(delivery_lat, delivery_lng)
        exp.settled = True

        # 更新上一单完成时刻
        self._last_order_complete_min[driver_id] = current_sim_minutes

        # R8: 回填最近的等待记录——标记「等待后最终接到的单的分数」
        wait_records = self._wait_history.get(driver_id, [])
        for wr in reversed(wait_records):
            if not wr["resolved"]:
                wr["resolved"] = True
                wr["next_score"] = exp.score_at_decision
                wr["next_income"] = actual_income
                break

        # 入库（含 confirm_count 合并逻辑）
        self._add_to_index(driver_id, exp)

        logger.debug(
            "settled experience: driver=%s income=%.0f next_wait=%.0f delivery_region=%s",
            driver_id, actual_income, exp.next_order_wait_minutes, exp.delivery_region_key)

    def discard_pending(self, driver_id: str) -> None:
        """上一步 take_order 被拒绝或不是 take_order 时，丢弃待确认经验。"""
        self._pending.pop(driver_id, None)

    # ------------------------------------------------------------------
    # R8: 记录等待决策
    # ------------------------------------------------------------------

    def record_wait_decision(
        self,
        driver_id: str,
        sim_minutes: int,
        lat: float,
        lng: float,
        best_score_rejected: float,
        day: int,
    ) -> None:
        """记录一次「选择等待」的决策，用于评估等待策略效果。

        当司机拒绝当前最优货源选择等待时，记录被拒绝的最优分数。
        后续如果接了单（settle_pending），可以对比 best_score_rejected
        与实际接到的单的分数，评估等待是否「等到了更好的货」。

        这些信息存储在 _wait_history 中，generate_daily_summary 会汇总输出。
        """
        hour = (sim_minutes % 1440) / 60.0
        time_slot = hour_to_time_slot(hour)
        region_key = pos_to_region_key(lat, lng)

        if driver_id not in self._wait_history:
            self._wait_history[driver_id] = []

        self._wait_history[driver_id].append({
            "time_slot": time_slot,
            "region_key": region_key,
            "best_score_rejected": best_score_rejected,
            "sim_minutes": sim_minutes,
            "day": day,
            "resolved": False,       # 尚未知道等待后的结果
            "next_score": None,
        })

        # 仅保留最近 50 条等待记录
        if len(self._wait_history[driver_id]) > 50:
            self._wait_history[driver_id] = self._wait_history[driver_id][-50:]

        logger.debug(
            "recorded wait decision: driver=%s slot=%d region=%s rejected_score=%.2f",
            driver_id, time_slot, region_key, best_score_rejected)

    # ------------------------------------------------------------------
    # 检索经验
    # ------------------------------------------------------------------

    def query_experience(
        self, driver_id: str, time_slot: int, region_key: tuple[int, int],
        current_day: int,
    ) -> ExperienceSummary | None:
        """检索 (time_slot, region) 的经验摘要。

        返回 None 表示经验不足（未过冷启动或样本不够）。
        """
        if current_day < _COLD_START_DAYS:
            return None

        bucket = self._get_bucket(driver_id, time_slot, region_key)
        settled = [e for e in bucket if e.settled and e.weight > 0.01]
        if len(settled) < _MIN_CONFIDENCE_COUNT:
            return None

        total_w = sum(e.weight for e in settled)
        if total_w <= 0:
            return None

        avg_income = sum(e.actual_income * e.weight for e in settled) / total_w
        avg_next_wait = sum(e.next_order_wait_minutes * e.weight for e in settled) / total_w
        avg_score = sum(e.score_at_decision * e.weight for e in settled) / total_w

        # 置信度：基于样本量和 confirm 质量
        avg_confirm = sum(e.confirm_count * e.weight for e in settled) / total_w
        count_factor = min(1.0, len(settled) / 10.0)
        confirm_factor = min(1.0, avg_confirm / 3.0)
        confidence = count_factor * 0.6 + confirm_factor * 0.4

        return ExperienceSummary(
            count=len(settled),
            avg_income=avg_income,
            avg_next_wait=avg_next_wait,
            avg_score=avg_score,
            confidence=confidence,
        )

    def query_best_experience(
        self, driver_id: str, time_slot: int, region_key: tuple[int, int],
        current_day: int,
    ) -> BestExperienceDetail | None:
        """检索 (time_slot, region) 桶内 actual_income 最高的经验详情。

        保留向后兼容，内部调用 query_top_experiences 取第一条。
        """
        top = self.query_top_experiences(driver_id, time_slot, region_key, current_day, top_k=1)
        return top[0] if top else None

    def query_top_experiences(
        self, driver_id: str, time_slot: int, region_key: tuple[int, int],
        current_day: int, top_k: int = 3,
    ) -> list[BestExperienceDetail]:
        """R10-A-v2: 检索桶内 top-K 经验，按置信度降序排列。

        置信度公式：
          净收益 = actual_income - pickup_km * 2.0
          置信度 = 0.40 * 净收益归一化 + 0.25 * 验证次数归一化
                 + 0.20 * 新鲜度归一化 + 0.15 * 等待代价归一化

        归一化方式：
          - 净收益: 桶内 min-max, 最高=1 最低=0
          - 验证次数: min(1.0, confirm_count / 3)
          - 新鲜度: weight（衰减后权重，桶内 max 归一化）
          - 等待代价: 1 / (1 + next_order_wait_minutes / 60)

        Returns:
            list[BestExperienceDetail]，长度 0~top_k。
        """
        if current_day < _COLD_START_DAYS:
            return []

        bucket = self._get_bucket(driver_id, time_slot, region_key)
        settled = [e for e in bucket if e.settled and e.weight > 0.01]
        if len(settled) < _MIN_CONFIDENCE_COUNT:
            return []

        # --- 计算各经验的净收益 ---
        _DEADHEAD_COST_PER_KM = 2.0
        net_incomes = [
            e.actual_income - e.pickup_km * _DEADHEAD_COST_PER_KM
            for e in settled
        ]

        # --- 归一化参数 ---
        net_min = min(net_incomes)
        net_max = max(net_incomes)
        net_range = net_max - net_min if net_max > net_min else 1.0

        weight_max = max(e.weight for e in settled)
        weight_max = weight_max if weight_max > 0 else 1.0

        # --- 逐条计算置信度 ---
        scored: list[tuple[float, float, DecisionExperience]] = []
        for exp, net_inc in zip(settled, net_incomes):
            income_norm = (net_inc - net_min) / net_range
            confirm_norm = min(1.0, exp.confirm_count / 3.0)
            freshness_norm = exp.weight / weight_max
            wait_norm = 1.0 / (1.0 + exp.next_order_wait_minutes / 60.0)

            confidence = (
                0.40 * income_norm
                + 0.25 * confirm_norm
                + 0.20 * freshness_norm
                + 0.15 * wait_norm
            )
            scored.append((confidence, net_inc, exp))

        # 按置信度降序，取 top_k
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[BestExperienceDetail] = []
        for conf, net_inc, exp in scored[:top_k]:
            results.append(BestExperienceDetail(
                actual_income=exp.actual_income,
                cargo_price=exp.cargo_price,
                pickup_km=exp.pickup_km,
                score_at_decision=exp.score_at_decision,
                next_order_wait_minutes=exp.next_order_wait_minutes,
                delivery_region_key=exp.delivery_region_key,
                time_slot=exp.time_slot,
                day=exp.day,
                confidence=round(conf, 3),
                net_income=round(net_inc, 1),
            ))

        return results

    def query_delivery_region(
        self, driver_id: str, time_slot: int,
        delivery_region_key: tuple[int, int], current_day: int,
    ) -> DeliveryRegionSummary | None:
        """检索卸货区域在特定时段的历史表现。

        用于 CargoScorer 的 position_bonus 校准。
        返回 None 表示经验不足。
        """
        if current_day < _COLD_START_DAYS:
            return None

        d_index = self._delivery_index.get(driver_id, {})
        key = (time_slot, delivery_region_key)
        bucket = d_index.get(key, [])
        settled = [e for e in bucket if e.settled and e.weight > 0.01]
        if len(settled) < _MIN_CONFIDENCE_COUNT:
            return None

        total_w = sum(e.weight for e in settled)
        if total_w <= 0:
            return None

        avg_income = sum(e.actual_income * e.weight for e in settled) / total_w
        avg_next_wait = sum(e.next_order_wait_minutes * e.weight for e in settled) / total_w

        count_factor = min(1.0, len(settled) / 10.0)
        confidence = count_factor

        return DeliveryRegionSummary(
            count=len(settled),
            avg_income=avg_income,
            avg_next_wait=avg_next_wait,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # 生成 daily_review 的经验摘要文本
    # ------------------------------------------------------------------

    def generate_daily_summary(self, driver_id: str, current_day: int) -> str:
        """生成供 daily_review 使用的经验摘要文本。"""
        if current_day < _COLD_START_DAYS:
            return "（冷启动期，暂无决策经验）"

        index = self._index.get(driver_id, {})
        if not index:
            return "（暂无决策经验）"

        # 收集所有桶的摘要
        slot_names = ["深夜(0-6)", "早高峰(6-11)", "午间(11-14)", "晚高峰(14-20)", "晚间(20-24)"]
        lines: list[str] = ["决策经验摘要："]

        # 按时段聚合
        slot_stats: dict[int, list[tuple[tuple[int, int], float, float, int]]] = defaultdict(list)
        for (ts, rk), bucket in index.items():
            settled = [e for e in bucket if e.settled and e.weight > 0.01]
            if len(settled) < 2:
                continue
            total_w = sum(e.weight for e in settled)
            if total_w <= 0:
                continue
            avg_inc = sum(e.actual_income * e.weight for e in settled) / total_w
            avg_wait = sum(e.next_order_wait_minutes * e.weight for e in settled) / total_w
            slot_stats[ts].append((rk, avg_inc, avg_wait, len(settled)))

        for ts in range(5):
            regions = slot_stats.get(ts, [])
            if not regions:
                continue
            # 按平均收入排序，展示 top-3 和 bottom-1
            regions.sort(key=lambda x: x[1], reverse=True)
            lines.append(f"  {slot_names[ts]}:")
            for rk, avg_inc, avg_wait, cnt in regions[:3]:
                lines.append(
                    f"    区域{rk}: 平均收入{avg_inc:.0f}元, 下一单等待{avg_wait:.0f}分钟 ({cnt}条经验)")
            if len(regions) > 3:
                rk, avg_inc, avg_wait, cnt = regions[-1]
                lines.append(
                    f"    最差区域{rk}: 平均收入{avg_inc:.0f}元, 等待{avg_wait:.0f}分钟 ({cnt}条)")

        # 稳定模式（confirm_count >= 3）
        stable_patterns: list[str] = []
        for (ts, rk), bucket in index.items():
            stable = [e for e in bucket if e.confirm_count >= 3 and e.settled]
            if stable:
                total_w = sum(e.weight for e in stable)
                if total_w > 0:
                    avg_inc = sum(e.actual_income * e.weight for e in stable) / total_w
                    stable_patterns.append(
                        f"{slot_names[ts]}+区域{rk}: 稳定收入{avg_inc:.0f}元 (验证{len(stable)}次)")

        if stable_patterns:
            lines.append("  已验证的稳定模式：")
            for p in stable_patterns[:5]:
                lines.append(f"    {p}")

        # R8: 等待策略效果统计
        wait_records = self._wait_history.get(driver_id, [])
        resolved = [wr for wr in wait_records if wr["resolved"]]
        if resolved:
            improved = sum(
                1 for wr in resolved
                if wr["next_score"] is not None
                and wr["next_score"] > wr["best_score_rejected"]
            )
            total = len(resolved)
            improve_rate = improved / total if total > 0 else 0
            lines.append(
                f"  等待策略效果：{total}次等待中{improved}次等到更好的货 "
                f"({improve_rate:.0%}成功率)")
            if improve_rate < 0.3 and total >= 5:
                lines.append("    建议：等待成功率偏低，考虑降低 wait_value 或提高 aggression")
            elif improve_rate > 0.7 and total >= 5:
                lines.append("    建议：等待策略效果好，可适当提高 wait_score_threshold 筛选更优货源")

        return "\n".join(lines) if len(lines) > 1 else "（经验数据不足）"

    def extract_semantic_memory(self, driver_id: str, current_day: int) -> list[dict[str, Any]]:
        """MUSE Semantic 层：从 Episodic 经验中提炼结构化语义记忆。

        提炼三类规律：
        1. 时段规律：哪个时段收益最高/最低
        2. 区域规律：哪些区域是"黄金区域"（高收入+低等待）
        3. 等待规律：等待策略在什么条件下有效

        Returns:
            语义记忆列表，每条包含 {type, pattern, confidence, detail}
        """
        if current_day < _COLD_START_DAYS + 2:
            return []

        index = self._index.get(driver_id, {})
        if not index:
            return []

        memories: list[dict[str, Any]] = []

        # --- 1. 时段收益规律 ---
        slot_names = ["night", "morning", "midday", "afternoon", "evening"]
        slot_income: dict[int, list[float]] = defaultdict(list)
        for (ts, _rk), bucket in index.items():
            settled = [e for e in bucket if e.settled and e.weight > 0.01]
            for e in settled:
                slot_income[ts].append(e.actual_income)

        slot_avg = {}
        for ts, incomes in slot_income.items():
            if len(incomes) >= 3:
                slot_avg[ts] = sum(incomes) / len(incomes)

        if len(slot_avg) >= 2:
            best_ts = max(slot_avg, key=slot_avg.get)  # type: ignore[arg-type]
            worst_ts = min(slot_avg, key=slot_avg.get)  # type: ignore[arg-type]
            if slot_avg[best_ts] > slot_avg[worst_ts] * 1.3:  # 30%+ 差异才记录
                memories.append({
                    "type": "time_slot_pattern",
                    "pattern": f"{slot_names[best_ts]}时段收入显著优于{slot_names[worst_ts]}时段",
                    "confidence": min(1.0, len(slot_income[best_ts]) / 10),
                    "detail": {
                        "best_slot": slot_names[best_ts],
                        "best_avg": round(slot_avg[best_ts], 0),
                        "worst_slot": slot_names[worst_ts],
                        "worst_avg": round(slot_avg[worst_ts], 0),
                    },
                })

        # --- 2. 黄金区域规律（高收入 + 低等待） ---
        region_scores: list[tuple[tuple[int, int], float, float, int]] = []
        for (ts, rk), bucket in index.items():
            settled = [e for e in bucket if e.settled and e.weight > 0.01
                       and e.next_order_wait_minutes > 0]
            if len(settled) >= 3:
                total_w = sum(e.weight for e in settled)
                if total_w > 0:
                    avg_inc = sum(e.actual_income * e.weight for e in settled) / total_w
                    avg_wait = sum(e.next_order_wait_minutes * e.weight for e in settled) / total_w
                    # 综合得分：收入高+等待短
                    composite = avg_inc / max(avg_wait, 10)
                    region_scores.append((rk, avg_inc, avg_wait, len(settled)))

        if region_scores:
            region_scores.sort(key=lambda x: x[1] / max(x[2], 10), reverse=True)
            # 取 top-2 黄金区域
            for rk, avg_inc, avg_wait, cnt in region_scores[:2]:
                if cnt >= 3:
                    memories.append({
                        "type": "golden_region",
                        "pattern": f"区域{rk}是黄金区域：平均收入{avg_inc:.0f}元且下一单等待仅{avg_wait:.0f}分钟",
                        "confidence": min(1.0, cnt / 8),
                        "detail": {
                            "region": rk,
                            "avg_income": round(avg_inc, 0),
                            "avg_wait": round(avg_wait, 0),
                            "sample_count": cnt,
                        },
                    })

        # --- 3. 等待策略规律 ---
        wait_records = self._wait_history.get(driver_id, [])
        resolved = [wr for wr in wait_records if wr["resolved"]]
        if len(resolved) >= 5:
            improved = sum(
                1 for wr in resolved
                if wr["next_score"] is not None
                and wr["next_score"] > wr["best_score_rejected"]
            )
            rate = improved / len(resolved)
            if rate > 0.6:
                memories.append({
                    "type": "wait_strategy",
                    "pattern": f"等待策略有效（{rate:.0%}成功率），适合在分数不够高时等待",
                    "confidence": min(1.0, len(resolved) / 10),
                    "detail": {"success_rate": round(rate, 2), "sample_count": len(resolved)},
                })
            elif rate < 0.3:
                memories.append({
                    "type": "wait_strategy",
                    "pattern": f"等待策略低效（{rate:.0%}成功率），应优先接单而非等待",
                    "confidence": min(1.0, len(resolved) / 10),
                    "detail": {"success_rate": round(rate, 2), "sample_count": len(resolved)},
                })

        return memories

    # ------------------------------------------------------------------
    # 步级等待反馈
    # ------------------------------------------------------------------

    def get_recent_wait_success_rate(
        self, driver_id: str, n: int = 3,
    ) -> tuple[float, int]:
        """返回最近 N 次已 resolved 等待的成功率。

        "成功" = 等待后接到的单分数高于等待时拒绝的最优分数。

        Returns:
            (success_rate, sample_count)。sample_count < n 时数据不足，
            调用方可选择忽略。success_rate 在 [0, 1] 之间。
        """
        wait_records = self._wait_history.get(driver_id, [])
        resolved = [wr for wr in wait_records if wr["resolved"]]
        recent = resolved[-n:] if len(resolved) >= n else resolved
        if not recent:
            return 0.5, 0  # 无数据时返回中性值

        improved = sum(
            1 for wr in recent
            if wr["next_score"] is not None
            and wr["next_score"] > wr["best_score_rejected"]
        )
        return improved / len(recent), len(recent)

    # ------------------------------------------------------------------
    # 每日衰减
    # ------------------------------------------------------------------

    def daily_decay(self, driver_id: str, current_day: int) -> None:
        """每日衰减：根据 confirm_count 施加不同衰减率。"""
        last_day = self._last_decay_day.get(driver_id, -1)
        if current_day <= last_day:
            return

        days_elapsed = current_day - last_day if last_day >= 0 else 1
        self._last_decay_day[driver_id] = current_day

        index = self._index.get(driver_id, {})
        for bucket in index.values():
            for exp in bucket:
                decay_rate = _get_decay_rate(exp.confirm_count)
                exp.weight *= decay_rate ** days_elapsed

        # 同步衰减 delivery_index
        d_index = self._delivery_index.get(driver_id, {})
        for bucket in d_index.values():
            for exp in bucket:
                decay_rate = _get_decay_rate(exp.confirm_count)
                exp.weight *= decay_rate ** days_elapsed

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_bucket(
        self, driver_id: str, time_slot: int, region_key: tuple[int, int],
    ) -> list[DecisionExperience]:
        """获取指定桶的经验列表。"""
        index = self._index.get(driver_id, {})
        return index.get((time_slot, region_key), [])

    def _add_to_index(self, driver_id: str, exp: DecisionExperience) -> None:
        """将已落定的经验入库，含 confirm_count 合并逻辑。"""
        if driver_id not in self._index:
            self._index[driver_id] = {}
        if driver_id not in self._delivery_index:
            self._delivery_index[driver_id] = {}

        index = self._index[driver_id]
        d_index = self._delivery_index[driver_id]
        key = (exp.time_slot, exp.region_key)

        if key not in index:
            index[key] = []
        bucket = index[key]

        # confirm_count 合并：检查是否有结果相似的旧经验
        merged = False
        for old_exp in bucket:
            if not old_exp.settled:
                continue
            # 判断收入方向是否一致（差异 < 30%）
            if old_exp.actual_income > 0 and exp.actual_income > 0:
                ratio = abs(exp.actual_income - old_exp.actual_income) / max(
                    old_exp.actual_income, 1.0)
                if ratio < _SIMILAR_INCOME_RATIO:
                    old_exp.confirm_count += 1
                    old_exp.weight = 1.0  # 被重新验证，刷新权重
                    merged = True
                    logger.debug(
                        "merged experience: slot=%d region=%s confirm_count=%d",
                        exp.time_slot, exp.region_key, old_exp.confirm_count)
                    break

        # 无论是否合并，新经验也入库（提供更多数据点）
        bucket.append(exp)

        # 桶内容量管理：超过上限时淘汰权重最低的
        if len(bucket) > _MAX_EXPERIENCES_PER_BUCKET:
            bucket.sort(key=lambda e: e.weight, reverse=True)
            index[key] = bucket[:_MAX_EXPERIENCES_PER_BUCKET]

        # 同步到 delivery_index
        if exp.delivery_region_key != (0, 0):
            d_key = (exp.time_slot, exp.delivery_region_key)
            if d_key not in d_index:
                d_index[d_key] = []
            d_index[d_key].append(exp)
            # 容量管理
            if len(d_index[d_key]) > _MAX_EXPERIENCES_PER_BUCKET:
                d_index[d_key].sort(key=lambda e: e.weight, reverse=True)
                d_index[d_key] = d_index[d_key][:_MAX_EXPERIENCES_PER_BUCKET]
