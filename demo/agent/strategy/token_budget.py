"""Token 预算管理器 v5：追踪和控制 LLM token 消耗 + 弹性分配 + API 状态感知。

每个司机有 5M token 上限（4 小时运行时间）。
本模块提供预算追踪、弹性分配和限流能力，确保 token 用在刀刃上。

预算分配策略：
  - 总预算 5M，可用 80% = 4M（留 20% 安全余量）
  - Phase 1 (偏好解析): 固定 10K 预留
  - Phase 2 (每日回顾): 固定 100K 预留 (31天 × ~3K)
  - Phase 3 (决策增强): 动态分配剩余预算

v5 改进：
  - 弹性再分配：当实际使用率低于预期时，主动降低 LLM 调用门槛，充分利用预算
  - API 状态感知：跟踪 API 健康状态，降级时不浪费预算做无效尝试
  - 使用率分析：提供实时建议，帮助优化预算利用
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agent.token_budget")

# 预算常量
TOTAL_BUDGET = 5_000_000
SAFETY_MARGIN = 0.80  # 只用 80%
USABLE_BUDGET = int(TOTAL_BUDGET * SAFETY_MARGIN)  # 4M

# 各模块预留
BUDGET_PARSE = 10_000         # 偏好解析
BUDGET_DAILY_REVIEW = 100_000  # 每日策略回顾 (31天 × ~3K)
BUDGET_CUSTOM_EVAL = 200_000   # Custom 偏好评估
BUDGET_EMERGENCY = 100_000     # 应急兜底

# 剩余给决策增强
BUDGET_DECISION = USABLE_BUDGET - BUDGET_PARSE - BUDGET_DAILY_REVIEW - BUDGET_CUSTOM_EVAL - BUDGET_EMERGENCY


@dataclass
class BudgetCategory:
    """单个预算分类。"""
    name: str
    allocated: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.allocated - self.used)

    @property
    def utilization(self) -> float:
        if self.allocated == 0:
            return 0.0
        return self.used / self.allocated

    def can_spend(self, amount: int) -> bool:
        return self.remaining >= amount

    def spend(self, amount: int) -> None:
        self.used += amount


class TokenBudgetManager:
    """管理单个司机的 token 预算（v5 弹性版）。"""

    def __init__(self, total_budget: int = USABLE_BUDGET) -> None:
        self._total_budget = total_budget
        self._total_used = 0
        self._call_count = 0

        # 各模块预算
        self._categories: dict[str, BudgetCategory] = {
            "parse": BudgetCategory("偏好解析", BUDGET_PARSE),
            "daily_review": BudgetCategory("每日策略回顾", BUDGET_DAILY_REVIEW),
            "decision": BudgetCategory("决策增强", BUDGET_DECISION),
            "custom_eval": BudgetCategory("Custom评估", BUDGET_CUSTOM_EVAL),
            "emergency": BudgetCategory("应急兜底", BUDGET_EMERGENCY),
        }

        # 统计
        self._step_token_history: list[int] = []

        # v5: API 状态感知
        self._api_healthy = True
        self._api_failure_count = 0
        self._api_last_success_step = 0

        # v5: 弹性分配参数
        self._expected_utilization_per_day = total_budget / 31  # 理想每天用量
        self._llm_call_threshold_boost = 0.0  # 降低门槛的促进因子

    @property
    def total_used(self) -> int:
        return self._total_used

    @property
    def total_remaining(self) -> int:
        return max(0, self._total_budget - self._total_used)

    @property
    def utilization(self) -> float:
        return self._total_used / self._total_budget if self._total_budget > 0 else 0.0

    @property
    def api_healthy(self) -> bool:
        return self._api_healthy

    def can_spend(self, category: str, estimated_tokens: int) -> bool:
        """判断某类别是否还有预算可用。

        即使该类别预算用完，只要总预算还有余量也允许（从 emergency 借）。
        """
        cat = self._categories.get(category)
        if cat and cat.can_spend(estimated_tokens):
            return True
        # 总预算检查
        return self.total_remaining >= estimated_tokens

    def record_usage(self, category: str, tokens_used: int) -> None:
        """记录 token 消耗。"""
        self._total_used += tokens_used
        self._call_count += 1

        cat = self._categories.get(category)
        if cat:
            cat.spend(tokens_used)
        else:
            # 未分类的用量计入 emergency
            self._categories["emergency"].spend(tokens_used)

        self._step_token_history.append(tokens_used)

        logger.debug(
            "token usage: category=%s, used=%d, total=%d/%d (%.1f%%)",
            category, tokens_used, self._total_used, self._total_budget,
            self.utilization * 100,
        )

    def get_decision_budget_per_step(self, remaining_steps: int) -> int:
        """计算每步可用的决策 token 预算。

        根据剩余步数动态分配，确保后面的步骤也有 token 可用。
        """
        decision_remaining = self._categories["decision"].remaining
        if remaining_steps <= 0:
            return decision_remaining

        # 基本均分 + 10% 缓冲
        per_step = int(decision_remaining / remaining_steps * 0.9)
        # 单步不超过 3000 token（避免浪费）
        return min(per_step, 3000)

    # ── v5: API 状态感知 ──────────────────────────────────────────────────

    def notify_api_status(self, success: bool, step: int = 0) -> None:
        """更新 API 健康状态（由 StrategyAdvisor 回调）。"""
        if success:
            self._api_healthy = True
            self._api_failure_count = 0
            self._api_last_success_step = step
        else:
            self._api_failure_count += 1
            if self._api_failure_count >= 2:
                self._api_healthy = False

    # ── v5: 弹性分配 ─────────────────────────────────────────────────────

    def recalibrate(self, current_day: int) -> None:
        """弹性再分配：根据实际使用情况动态调整 LLM 调用门槛。

        如果实际使用率远低于预期，降低 LLM 调用门槛使预算得到更充分利用。
        在每日 daily_review 时调用。
        """
        if current_day <= 1:
            return

        # 计算实际 vs 理想使用率
        expected_used = self._expected_utilization_per_day * current_day
        actual_ratio = self._total_used / max(1, expected_used)

        # 如果实际使用不足理想值的 30%，显著降低门槛
        if actual_ratio < 0.3:
            self._llm_call_threshold_boost = 0.6  # 大幅降低门槛
            logger.info(
                "budget recalibrate day=%d: actual/expected=%.1f%%, boosting LLM usage (boost=0.6)",
                current_day, actual_ratio * 100)
        elif actual_ratio < 0.6:
            self._llm_call_threshold_boost = 0.3  # 适度降低
            logger.info(
                "budget recalibrate day=%d: actual/expected=%.1f%%, moderate boost (boost=0.3)",
                current_day, actual_ratio * 100)
        else:
            self._llm_call_threshold_boost = 0.0  # 使用正常

    def rebalance_budget(self, current_day: int, api_degraded: bool = False) -> None:
        """根据运行进度弹性重分配预算。

        核心思想：
          - 如果当前利用率远低于预期（如2.98%），说明规则决策已足够好，
            可以将多余预留转移到 decision 类别，在高价值场景更激进使用 LLM。
          - 如果 API 已降级，冻结 LLM 类别预算，保留仅供恢复后使用。

        Args:
            current_day: 当前仿真天数 (0-30)
            api_degraded: 当前 API 是否处于降级状态
        """
        progress = current_day / 31.0  # 0.0 ~ 1.0
        expected_utilization = progress * 0.35  # 预期用 35% 预算

        if api_degraded:
            # API 降级时不重分配，保持现状等待恢复
            logger.info(
                "budget rebalance skipped: API degraded (day=%d, used=%.1f%%)",
                current_day, self.utilization * 100)
            return

        # 如果实际利用率远低于预期（不到一半），释放多余预留给 decision
        if progress > 0.1 and self.utilization < expected_utilization * 0.5:
            # 将 daily_review 和 custom_eval 的未使用部分转入 decision
            for cat_name in ("daily_review", "custom_eval"):
                cat = self._categories[cat_name]
                # 保留已用 + 20% 余量，剩余释放
                keep = int(cat.used * 1.2) + 5000
                if cat.allocated > keep:
                    freed = cat.allocated - keep
                    cat.allocated = keep
                    self._categories["decision"].allocated += freed
                    logger.info(
                        "budget rebalance: freed %d tokens from %s -> decision",
                        freed, cat_name)

    # ── 决策判断 ──────────────────────────────────────────────────────────

    def should_use_llm_for_decision(self, state_info: dict[str, Any]) -> bool:
        """根据当前状态判断是否值得在本步使用 LLM 增强决策。

        高价值场景（优先使用 LLM）：
          1. 货源数量多（>= 3个），选择困难
          2. 分差微小的 top-N 选项
          3. 月初（策略探索期）或月末（冲刺期）
          4. 长时间未接单后首次有货
          5. 有 custom 约束未覆盖

        低价值场景（省 token）：
          1. 只有 1 个货源，无需选择
          2. 处于休息/安静窗口（规则已决定）
          3. 明显的最优选择（分差大）

        v5 改进：
          - API 不健康时直接返回 False，避免无效尝试
          - 弹性加成：实际使用率低时自动降低触发门槛
        """
        # API 降级时直接跳过
        if not self._api_healthy:
            return False

        # 如果决策预算不足，不用 LLM
        if not self.can_spend("decision", 1500):
            return False

        # 提取判断信息
        cargo_count = state_info.get("cargo_count", 0)
        score_gap = state_info.get("score_gap", 999)
        best_score = state_info.get("best_score", 0)
        has_custom = state_info.get("has_custom_constraints", False)
        day = state_info.get("current_day", 15)
        steps_without_order = state_info.get("steps_without_order", 0)

        # 规则：哪些情况值得调 LLM
        reasons = []

        # v5 弹性加成：预算充足时降低门槛
        boosted_gap_threshold = 10 + self._llm_call_threshold_boost * 15  # 最多提升到 19

        # 多货源选择困难（门槛可弹性提升）
        if cargo_count >= 3 and score_gap < boosted_gap_threshold:
            reasons.append("多货源分差小")
        elif cargo_count >= 2 and self._llm_call_threshold_boost >= 0.3:
            # 预算充足时，2 个货源也值得用 LLM
            reasons.append("预算充足+多货源")

        # 有 custom 约束
        if has_custom:
            reasons.append("有 custom 约束需评估")

        # 月末冲刺期
        if day >= 25:
            reasons.append("月末冲刺")

        # 月初探索期（前3天多用 LLM 学习货源分布）
        if day <= 3:
            reasons.append("月初探索")

        # 长时间空窗后
        if steps_without_order >= 5:
            reasons.append("长时间未接单")

        # best_score 在边界区间（正负交界，难以判断）
        if -5 <= best_score <= 15:
            reasons.append("得分在边界区间")

        # v5: 预算极度充足时，任何有 2+ 货源的场景都值得尝试
        if not reasons and self._llm_call_threshold_boost >= 0.6 and cargo_count >= 2:
            reasons.append("预算极度充足，主动利用")

        if reasons:
            logger.debug("LLM decision triggered: %s (boost=%.1f)",
                        ", ".join(reasons), self._llm_call_threshold_boost)
            return True

        return False

    def should_do_daily_review(self, day: int, last_review_day: int) -> bool:
        """判断是否需要做每日策略回顾。"""
        if day == last_review_day:
            return False
        return self.can_spend("daily_review", 3000)

    def get_summary(self) -> dict[str, Any]:
        """获取预算使用摘要。"""
        return {
            "total_budget": self._total_budget,
            "total_used": self._total_used,
            "total_remaining": self.total_remaining,
            "utilization": f"{self.utilization * 100:.1f}%",
            "call_count": self._call_count,
            "avg_per_call": self._total_used // max(1, self._call_count),
            "api_healthy": self._api_healthy,
            "llm_boost": self._llm_call_threshold_boost,
            "categories": {
                name: {
                    "allocated": cat.allocated,
                    "used": cat.used,
                    "remaining": cat.remaining,
                    "utilization": f"{cat.utilization * 100:.1f}%",
                }
                for name, cat in self._categories.items()
            },
        }

    def estimate_remaining_steps(self, current_sim_minutes: int) -> int:
        """估算剩余仿真步数。

        假设 31 天仿真 = 44640 分钟，平均每步 ~60-90 分钟。
        """
        remaining_minutes = 44640 - current_sim_minutes
        # 经验值：每步约消耗 80 分钟仿真时间
        return max(1, remaining_minutes // 80)

    def get_adaptive_llm_threshold(self, current_day: int) -> float:
        """根据预算使用进度动态调整 LLM 调用阈值。

        返回值越低，表示越容易触发 LLM 调用。
        返回值范围: 0.0（总是调用）~ 1.0（几乎不调用）

        策略：
          - 预算充裕时降低阈值（多用 LLM）
          - 预算紧张时提高阈值（省着用）
          - 月末冲刺期适当降低阈值
        """
        budget_pressure = self.utilization  # 0.0 ~ 1.0
        progress = current_day / 31.0

        # 基础阈值：预算压力越大，阈值越高
        base_threshold = budget_pressure * 0.8

        # 月末冲刺期（最后5天）降低阈值
        if current_day >= 26:
            base_threshold *= 0.6

        # 预算极度充裕（用量<预期的30%）时大幅降低阈值
        expected_use = progress * 0.3 * self._total_budget
        if expected_use > 0 and self._total_used < expected_use * 0.3:
            base_threshold *= 0.4

        return min(1.0, max(0.0, base_threshold))
