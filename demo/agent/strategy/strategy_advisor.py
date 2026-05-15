"""策略顾问：周期性调用 LLM 进行全局策略回顾和参数调优。

核心功能：
  1. 每日策略回顾：分析前一天表现，调整评分权重和等待阈值
  2. 关键节点决策：月初/月中/月末策略切换
  3. Custom 偏好逐步评估：对无法规则化的偏好进行 LLM 判定

设计原则：
  - 每日回顾只在跨天时触发一次（~3K token）
  - 调优结果缓存为当天策略参数
  - 不改变核心规则逻辑，只调整软参数
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agent.config.driver_config import DriverConfig, get_config
from agent.core.state_tracker import DriverState

logger = logging.getLogger("agent.strategy_advisor")


# ===========================================================================
# 策略参数（可由 LLM 动态调优）
# ===========================================================================

@dataclass
class StrategyParams:
    """可调策略参数，由每日回顾更新。"""
    # 等待决策参数
    wait_score_threshold: float = 0.0        # 低于此分拒绝接单
    wait_value_multiplier: float = 0.6       # 等待价值乘数
    max_wait_minutes: int = 30               # 单次最大等待时长
    min_wait_minutes: int = 10               # 单次最小等待时长

    # 评分调整 —— 供 CargoScorer 使用
    deadhead_penalty_factor: float = 1.0     # 空驶惩罚系数
    efficiency_bonus_factor: float = 1.0     # 效率加成系数
    position_bonus_factor: float = 1.0       # 位置加成系数

    # 评分子项参数 —— 消除 CargoScorer 中的 magic number
    time_cost_per_minute: float = 0.05       # 每分钟时间机会成本（元）
    deadhead_ratio_threshold: float = 0.5    # 空驶比超过此值开始罚分
    deadhead_over_penalty: float = 20.0      # 超出阈值部分的每单位罚分
    soft_violation_penalty: float = 30.0     # 软约束违反固定罚分
    position_bonus_cap: float = 15.0         # 位置加分上限
    position_near_km: float = 5.0            # 满分距离阈值（km）
    position_fade_km: float = 20.0           # 加分衰减截止距离（km）
    home_bonus_radius_km: float = 10.0       # 回家加分半径（km）
    home_bonus_per_km: float = 1.5           # 回家加分每 km 系数
    deadhead_budget_penalty_mult: float = 2.0 # 配额保护罚分倍率
    deadhead_budget_warning_mult: float = 3.0 # 配额余量预警倍率
    profit_per_km_bonus_mult: float = 5.0    # 每公里利润加成倍率

    # 接单激进度
    aggression: float = 0.5                  # 0=保守(多等待) 1=激进(有就接)
    # 当 aggression 高时：降低等待阈值，更容易接单
    # 当 aggression 低时：提高等待阈值，更愿意等待高分货源

    # 月度节奏
    daily_income_target: float = 0.0         # 每日收入目标（0=不设）
    monthly_target_on_track: bool = True     # 是否跟上月度目标进度

    # 时段偏好
    peak_hours: list[tuple[int, int]] = field(default_factory=lambda: [(8, 11), (14, 18)])
    off_peak_aggression_boost: float = 0.2   # 非高峰时段提高激进度


DAILY_REVIEW_SYSTEM_PROMPT = """你是货运策略优化顾问。根据司机昨日的运营数据和趋势，给出今日的策略参数调整建议。

输入：司机的约束配置摘要 + 昨日运营数据 + 月度累计数据 + 近3日趋势。
输出：一个JSON对象，包含以下可调参数：

{
  "wait_score_threshold": <float>,    // 低于此分的货不接，等更好的（-10~30）
  "wait_value_multiplier": <float>,   // 等待价值乘数，越高越倾向等待（0.3~1.0）
  "max_wait_minutes": <int>,          // 单次等待上限（10~60）
  "min_wait_minutes": <int>,          // 单次最小等待时长（5~20）
  "deadhead_penalty_factor": <float>, // 空驶惩罚系数，越高越排斥远距取货（0.5~2.0）
  "efficiency_bonus_factor": <float>, // 每公里利润的加成系数（0.5~2.0）
  "position_bonus_factor": <float>,   // 位置加成系数，越高越看重卸货位置（0.5~2.0）
  "time_cost_per_minute": <float>,    // 每分钟时间机会成本（0.02~0.15）
  "soft_violation_penalty": <float>,  // 软约束违反罚分（10~60）
  "aggression": <float>,              // 接单激进度 0=保守(多等) 1=激进(有就接)（0~1）
  "reasoning": "<一句话解释调整理由>"
}

策略建议原则：
1. 月初(1-10天)：探索为主，aggression=0.4-0.6，多等待观察货源分布
2. 月中(11-20天)：稳定执行，根据收入进度调整 aggression
3. 月末(21-31天)：冲刺为主，aggression=0.6-0.9，降低等待阈值多接单
4. 如果昨日空驶比高(>40%)，提高 deadhead_penalty_factor
5. 如果昨日收入低于日均目标，提高 aggression
6. 如果昨日等待时间过长(>30%步数在等待)，降低 wait_score_threshold
7. 考虑偏好约束的罚分风险：如果月度罚分累计已高，降低 aggression 更谨慎
8. 关注近3日趋势：收入连续下降时提高 aggression，连续上升时可适当保守
9. 如果空驶配额消耗过快（剩余天数占比 > 剩余配额占比），提高 deadhead_penalty_factor
10. 如果 position_bonus 效果好（卸货后下一单获取快），提高 position_bonus_factor
11. 【重要】如果提供了 opro_experience（历史参数实验记录），请参考历史最优参数组合，
    在其基础上做小幅探索性调整，而非从头猜测。收益高的组合值得复用，收益低的要回避。

只输出JSON，不输出其他文字。"""


DECISION_ENHANCE_SYSTEM_PROMPT = """你是货运决策助手。基于当前状态和候选货源信息，做出最优决策。

你需要判断：
1. 应该接哪个货？还是应该等待？
2. 如果等待，建议等多久？
3. 如果接单，选哪个货最优？

决策要点：
- 考虑当前时段（高峰vs低谷）对后续货源供给的影响
- 考虑卸货位置对下一单获取的影响（位置价值）
- 考虑月度进度（是否需要冲收入）
- 考虑约束满足情况（今日休息是否完成、off-day进度等）
- 考虑空驶预算消耗情况

输出格式：
{
  "action": "take_order" | "wait",
  "cargo_id": "<如果 take_order，填写 cargo_id>",
  "wait_minutes": <如果 wait，建议等待分钟数 10-60>,
  "confidence": <0-1，决策置信度>,
  "reasoning": "<一句话决策理由>"
}

只输出JSON。"""


CUSTOM_EVAL_SYSTEM_PROMPT = """你是偏好约束评估员。判断当前决策是否可能违反以下自然语言描述的偏好约束。

偏好约束（自然语言描述）：
{custom_constraints}

当前情境：
{context}

问题：如果执行 "{proposed_action}" 动作（参数：{action_params}），是否违反上述任一约束？

输出JSON：
{{
  "violates": true | false,
  "violated_constraint": "<违反了哪条，null如果不违反>",
  "penalty_estimate": <预估罚分，0如果不违反>,
  "reasoning": "<一句话理由>"
}}

只输出JSON。"""


@dataclass
class ParameterExperiment:
    """一次参数实验的记录（OPRO 经验缓冲区条目）。"""
    day: int
    params_snapshot: dict[str, float]
    result: dict[str, float]  # {income, deadhead_ratio, orders, penalty_risk}


class ExperienceBuffer:
    """OPRO 风格经验缓冲区：存储历史参数-收益对，供 LLM 参考优化。

    核心思想：让 LLM 看到"哪些参数组合带来了好/坏的结果"，
    从而在已有经验基础上提出更优的新参数。
    """

    def __init__(self, max_size: int = 10):
        self._buffer: list[ParameterExperiment] = []
        self._max_size = max_size

    def add(self, experiment: ParameterExperiment) -> None:
        """添加一次实验记录，按收益排序保留 top-K。"""
        self._buffer.append(experiment)
        # 按净效益（income - penalty_risk）排序
        self._buffer.sort(
            key=lambda x: x.result.get("income", 0) - x.result.get("penalty_risk", 0),
            reverse=True)
        if len(self._buffer) > self._max_size:
            self._buffer = self._buffer[:self._max_size]

    def to_prompt_context(self, top_k: int = 5) -> str:
        """生成 OPRO 风格的 few-shot context 供 LLM 参考。"""
        if not self._buffer:
            return "（首次运行，暂无历史经验）"
        lines = ["历史参数实验（按效果排序，越靠前越好）："]
        for i, exp in enumerate(self._buffer[:top_k]):
            net = exp.result.get("income", 0) - exp.result.get("penalty_risk", 0)
            lines.append(
                f"  #{i+1} Day{exp.day}: aggression={exp.params_snapshot.get('aggression', 0):.2f}, "
                f"wait_threshold={exp.params_snapshot.get('wait_score_threshold', 0):.1f}, "
                f"deadhead_factor={exp.params_snapshot.get('deadhead_penalty_factor', 0):.2f} "
                f"→ income={exp.result.get('income', 0):.0f}, "
                f"deadhead={exp.result.get('deadhead_ratio', 0):.1%}, "
                f"orders={exp.result.get('orders', 0)}, "
                f"net≈{net:.0f}")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._buffer)


class StrategyAdvisor:
    """策略顾问：管理动态策略参数和 LLM 增强决策。

    v5 增强：
      - API 降级感知：连续 N 次 403/429 后自动切换为纯规则模式，避免崩溃
      - 决策缓存：相似场景复用历史 LLM 决策，减少无效调用

    v6 增强（OPRO 风格）：
      - 经验缓冲区：存储历史参数-收益对，daily_review 时作为 few-shot 注入 prompt
      - 让 LLM 基于历史经验做更有信息量的参数搜索
    """

    # API 降级阈值：连续失败 N 次后进入纯规则模式
    _API_DEGRADE_THRESHOLD = 2
    # 降级恢复：纯规则模式下每 N 步尝试恢复一次
    _API_RECOVER_INTERVAL = 20

    def __init__(self) -> None:
        self._params: dict[str, StrategyParams] = {}  # driver_id → 当前策略参数
        self._last_review_day: dict[str, int] = {}    # driver_id → 上次回顾的天
        self._daily_stats: dict[str, list[dict]] = {} # driver_id → 每日统计

        # OPRO 经验缓冲区（每个司机独立）
        self._experience_buffer: dict[str, ExperienceBuffer] = {}

        # API 降级状态
        self._consecutive_api_failures: int = 0
        self._degraded_mode: bool = False
        self._steps_since_degrade: int = 0
        self._total_api_failures: int = 0

        # 决策缓存：(场景特征hash) → LLM 决策结果
        self._decision_cache: dict[str, dict[str, Any]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def get_params(self, driver_id: str) -> StrategyParams:
        """获取当前策略参数。"""
        if driver_id not in self._params:
            self._params[driver_id] = StrategyParams()
        return self._params[driver_id]

    @property
    def is_degraded(self) -> bool:
        """当前是否处于 API 降级（纯规则）模式。"""
        return self._degraded_mode

    def _on_api_success(self) -> None:
        """API 调用成功：重置失败计数，退出降级模式。"""
        self._consecutive_api_failures = 0
        if self._degraded_mode:
            logger.info("API recovered, exiting degraded mode")
            self._degraded_mode = False
            self._steps_since_degrade = 0

    def _on_api_failure(self, error: Exception) -> None:
        """API 调用失败：递增计数，判断是否进入降级模式。"""
        self._consecutive_api_failures += 1
        self._total_api_failures += 1
        if (self._consecutive_api_failures >= self._API_DEGRADE_THRESHOLD
                and not self._degraded_mode):
            self._degraded_mode = True
            self._steps_since_degrade = 0
            logger.warning(
                "Entering degraded mode after %d consecutive API failures: %s",
                self._consecutive_api_failures, error)

    def should_attempt_api_call(self) -> bool:
        """判断当前是否应该尝试 API 调用。

        降级模式下每 _API_RECOVER_INTERVAL 步尝试恢复一次。
        """
        if not self._degraded_mode:
            return True
        self._steps_since_degrade += 1
        if self._steps_since_degrade >= self._API_RECOVER_INTERVAL:
            self._steps_since_degrade = 0
            logger.info("Attempting API recovery probe")
            return True
        return False

    def should_review(self, driver_id: str, current_day: int) -> bool:
        """判断是否需要做每日策略回顾。"""
        last_day = self._last_review_day.get(driver_id, -1)
        return current_day > last_day

    def record_daily_stats(self, driver_id: str, state: DriverState) -> None:
        """记录每日统计数据，供趋势分析使用。同时更新 OPRO 经验缓冲区。"""
        day = state.current_day()
        days_elapsed = max(1, day + 1)
        stat = {
            "day": day,
            "income": round(state.total_gross_income / days_elapsed, 1),
            "deadhead_ratio": round(
                state.total_deadhead_km / max(1, state.total_distance_km), 3),
            "orders": state.today_order_count,
        }
        if driver_id not in self._daily_stats:
            self._daily_stats[driver_id] = []
        # 避免同一天重复记录
        existing_days = {s["day"] for s in self._daily_stats[driver_id]}
        if day not in existing_days:
            self._daily_stats[driver_id].append(stat)
            # OPRO：将昨日参数+结果存入经验缓冲区
            self._record_experiment(driver_id, day, stat)

    def daily_review(self, driver_id: str, state: DriverState,
                     config: DriverConfig, api: Any) -> StrategyParams:
        """每日策略回顾：调用 LLM 分析昨日表现，输出今日参数。

        v6 增强：注入 OPRO 经验缓冲区作为 few-shot context。

        Args:
            driver_id: 司机ID
            state: 当前状态
            config: 司机配置
            api: SimulationApiPort

        Returns:
            更新后的 StrategyParams
        """
        current_day = state.current_day()
        params = self.get_params(driver_id)

        # 记录昨日统计
        self.record_daily_stats(driver_id, state)

        # 构造回顾上下文
        review_context = self._build_review_context(driver_id, state, config)

        # OPRO：注入经验缓冲区
        exp_buf = self._get_experience_buffer(driver_id)
        experience_context = exp_buf.to_prompt_context(top_k=5)
        review_context["opro_experience"] = experience_context

        try:
            resp = api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": DAILY_REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(review_context, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            })

            choices = resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                updates = json.loads(content)
                self._apply_updates(params, updates)
                logger.info(
                    "daily review day=%d: aggression=%.2f, wait_threshold=%.1f, reason=%s",
                    current_day, params.aggression, params.wait_score_threshold,
                    updates.get("reasoning", ""),
                )
        except Exception as e:
            logger.warning("daily review failed: %s, using default params", e)
            # 回退：根据天数做简单调整
            self._fallback_day_adjust(params, current_day)

        self._last_review_day[driver_id] = current_day
        return params

    def enhance_decision(self, driver_id: str, state: DriverState,
                         config: DriverConfig, candidates: list[dict[str, Any]],
                         api: Any) -> dict[str, Any] | None:
        """LLM 增强决策：在复杂场景中调用 LLM 辅助选择。

        v5 增强：降级模式下直接返回 None，不调 API；
        支持决策缓存避免重复调用。

        Returns:
            {"action": "take_order"/"wait", ...} 或 None（LLM 无法决策时）
        """
        # 降级模式检查
        if not self.should_attempt_api_call():
            return None

        params = self.get_params(driver_id)
        context = self._build_decision_context(state, config, candidates, params)

        # 尝试缓存命中
        cache_key = self._build_decision_cache_key(state, candidates)
        if cache_key in self._decision_cache:
            self._cache_hits += 1
            cached = self._decision_cache[cache_key]
            logger.debug("Decision cache hit (hits=%d)", self._cache_hits)
            return cached
        self._cache_misses += 1

        try:
            resp = api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": DECISION_ENHANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
                "response_format": {"type": "json_object"},
            })

            self._on_api_success()

            choices = resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                result = json.loads(content)
                action = result.get("action", "")
                confidence = float(result.get("confidence", 0))

                # 置信度太低则不采纳
                if confidence < 0.4:
                    logger.debug("LLM decision low confidence=%.2f, skip", confidence)
                    return None

                if action == "take_order":
                    cargo_id = str(result.get("cargo_id", "")).strip().strip('"').strip("'")
                    # 验证 cargo_id 有效（两侧都做 strip 以容忍 LLM 格式差异）
                    valid_ids = {str(c.get("cargo_id", "")).strip(): str(c.get("cargo_id", ""))
                                 for c in candidates}
                    if cargo_id in valid_ids:
                        cargo_id = valid_ids[cargo_id]  # 使用原始值确保一致
                        logger.info("LLM enhanced: take %s (conf=%.2f, reason=%s)",
                                    cargo_id, confidence, result.get("reasoning", ""))
                        decision = {"action": "take_order", "params": {"cargo_id": cargo_id}}
                        self._decision_cache[cache_key] = decision
                        return decision

                elif action == "wait":
                    wait_min = int(result.get("wait_minutes", params.max_wait_minutes))
                    wait_min = max(params.min_wait_minutes, min(wait_min, 60))
                    logger.info("LLM enhanced: wait %d min (conf=%.2f, reason=%s)",
                                wait_min, confidence, result.get("reasoning", ""))
                    decision = {"action": "wait", "params": {"duration_minutes": wait_min}}
                    # 不缓存 wait 决策（时效性强）
                    return decision

        except Exception as e:
            self._on_api_failure(e)
            logger.warning("LLM enhanced decision failed: %s", e)

        return None

    def evaluate_custom_constraints(self, driver_id: str, state: DriverState,
                                    config: DriverConfig, customs: list[Any],
                                    proposed_action: str, action_params: dict,
                                    api: Any) -> tuple[bool, float]:
        """评估 custom 偏好是否被违反。

        v5 增强：降级模式下保守返回 (False, 0)（不阻止接单），
        避免 API 不可用时系统崩溃。

        Args:
            customs: CustomConstraint 列表
            proposed_action: 拟执行的动作
            action_params: 动作参数

        Returns:
            (violates, estimated_penalty)
        """
        if not customs:
            return False, 0.0

        # 降级模式：安全 fallback，不阻止接单
        if not self.should_attempt_api_call():
            logger.debug("custom eval skipped (degraded mode)")
            return False, 0.0

        # 构造约束文本
        constraint_texts = []
        for c in customs:
            constraint_texts.append(f"- {c.original_text} (罚分: {c.penalty_amount})")

        context = {
            "driver_id": driver_id,
            "sim_minutes": state.sim_minutes,
            "current_day": state.current_day(),
            "hour": state.hour_in_day(),
            "current_pos": [state.current_lat, state.current_lng],
            "today_orders": state.today_order_count,
            "total_income": state.total_gross_income,
        }

        prompt = CUSTOM_EVAL_SYSTEM_PROMPT.format(
            custom_constraints="\n".join(constraint_texts),
            context=json.dumps(context, ensure_ascii=False),
            proposed_action=proposed_action,
            action_params=json.dumps(action_params, ensure_ascii=False),
        )

        try:
            resp = api.model_chat_completion({
                "messages": [
                    {"role": "system", "content": "你是偏好约束评估员。只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            })

            self._on_api_success()

            choices = resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                result = json.loads(content)
                violates = bool(result.get("violates", False))
                penalty = float(result.get("penalty_estimate", 0))
                if violates:
                    logger.info("custom constraint violated: %s, penalty=%.1f",
                                result.get("violated_constraint"), penalty)
                return violates, penalty

        except Exception as e:
            self._on_api_failure(e)
            logger.warning("custom eval failed: %s", e)

        # API 失败时保守返回：不阻止接单
        return False, 0.0

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _build_review_context(self, driver_id: str, state: DriverState,
                              config: DriverConfig) -> dict[str, Any]:
        """构造每日回顾的上下文信息。"""
        current_day = state.current_day()
        days_elapsed = current_day + 1
        days_remaining = 31 - current_day

        # 月度进度
        avg_daily_income = state.total_gross_income / max(1, days_elapsed)
        avg_daily_distance = state.total_distance_km / max(1, days_elapsed)
        deadhead_ratio = state.total_deadhead_km / max(1, state.total_distance_km)
        order_days_count = len(state.order_days)
        off_days_count = len(state.off_days)

        context = {
            "driver_id": driver_id,
            "current_day": current_day,
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,

            # 月度累计
            "total_income": round(state.total_gross_income, 1),
            "total_distance_km": round(state.total_distance_km, 1),
            "total_deadhead_km": round(state.total_deadhead_km, 1),
            "deadhead_ratio": round(deadhead_ratio, 3),
            "order_days": order_days_count,
            "off_days": off_days_count,

            # 每日均值
            "avg_daily_income": round(avg_daily_income, 1),
            "avg_daily_distance": round(avg_daily_distance, 1),

            # 约束状况
            "monthly_off_days_required": config.monthly_off_days_required,
            "off_days_deficit": max(0, config.monthly_off_days_required - off_days_count),
            "has_quiet_window": config.quiet_window is not None,
            "must_return_home": config.must_return_home,
            "min_rest_minutes": config.min_continuous_rest_minutes,

            # 距离约束
            "max_monthly_deadhead_km": config.max_monthly_deadhead_km,
            "deadhead_budget_remaining": (
                config.max_monthly_deadhead_km - state.total_deadhead_km
                if config.max_monthly_deadhead_km else None
            ),
        }

        # 添加近 3 日趋势数据
        recent_stats = self._daily_stats.get(driver_id, [])[-3:]
        if recent_stats:
            context["recent_daily_trend"] = recent_stats

        return context

    def _build_decision_context(self, state: DriverState, config: DriverConfig,
                                candidates: list[dict[str, Any]],
                                params: StrategyParams) -> dict[str, Any]:
        """构造决策增强的上下文。"""
        context = {
            "sim_minutes": state.sim_minutes,
            "current_day": state.current_day(),
            "hour": state.hour_in_day(),
            "current_pos": [round(state.current_lat, 4), round(state.current_lng, 4)],
            "today_orders": state.today_order_count,
            "today_rest": state.longest_rest_today,
            "total_income": round(state.total_gross_income, 1),
            "total_deadhead_km": round(state.total_deadhead_km, 1),
            "strategy_aggression": params.aggression,
            "strategy_wait_threshold": params.wait_score_threshold,
            "candidates": [],
        }

        for c in candidates[:5]:  # 最多 5 个候选
            context["candidates"].append({
                "cargo_id": c.get("cargo_id"),
                "price": c.get("price"),
                "pickup_km": round(c.get("pickup_km", 0), 1),
                "haul_km": round(c.get("haul_km", 0), 1),
                "score": round(c.get("score", 0), 2),
                "category": c.get("category", ""),
                "delivery_lat": round(float(c.get("delivery_lat", 0)), 3),
                "delivery_lng": round(float(c.get("delivery_lng", 0)), 3),
            })

        # 添加约束提示
        constraints_summary = []
        if config.must_return_home:
            constraints_summary.append(f"每天{config.home_deadline_hour}点前回家")
        if config.max_daily_orders:
            constraints_summary.append(f"每天最多{config.max_daily_orders}单")
        if config.max_monthly_deadhead_km:
            remaining = config.max_monthly_deadhead_km - state.total_deadhead_km
            constraints_summary.append(f"月度空驶余额{remaining:.0f}km")
        if constraints_summary:
            context["active_constraints"] = constraints_summary

        return context

    def _build_decision_cache_key(self, state: DriverState,
                                    candidates: list[dict[str, Any]]) -> str:
        """构造决策缓存键：基于时段+位置网格+候选货源id列表。

        相同时段、相同网格位置、相同候选货源集 → 视为等价场景。
        """
        hour = state.hour_in_day()
        # 位置量化到 0.1 度网格
        lat_grid = int(state.current_lat * 10)
        lng_grid = int(state.current_lng * 10)
        # 候选 cargo_id 排序后拼接
        cargo_ids = "|".join(sorted(str(c.get("cargo_id", "")) for c in candidates[:3]))
        return f"{hour}_{lat_grid}_{lng_grid}_{cargo_ids}"

    def _apply_updates(self, params: StrategyParams, updates: dict[str, Any]) -> None:
        """将 LLM 返回的参数更新应用到 StrategyParams。"""
        if "wait_score_threshold" in updates:
            params.wait_score_threshold = max(-10, min(30, float(updates["wait_score_threshold"])))
        if "wait_value_multiplier" in updates:
            params.wait_value_multiplier = max(0.3, min(1.0, float(updates["wait_value_multiplier"])))
        if "max_wait_minutes" in updates:
            params.max_wait_minutes = max(10, min(60, int(updates["max_wait_minutes"])))
        if "min_wait_minutes" in updates:
            params.min_wait_minutes = max(5, min(20, int(updates["min_wait_minutes"])))
        if "deadhead_penalty_factor" in updates:
            params.deadhead_penalty_factor = max(0.5, min(2.0, float(updates["deadhead_penalty_factor"])))
        if "efficiency_bonus_factor" in updates:
            params.efficiency_bonus_factor = max(0.5, min(2.0, float(updates["efficiency_bonus_factor"])))
        if "position_bonus_factor" in updates:
            params.position_bonus_factor = max(0.5, min(2.0, float(updates["position_bonus_factor"])))
        if "time_cost_per_minute" in updates:
            params.time_cost_per_minute = max(0.02, min(0.15, float(updates["time_cost_per_minute"])))
        if "soft_violation_penalty" in updates:
            params.soft_violation_penalty = max(10, min(60, float(updates["soft_violation_penalty"])))
        if "aggression" in updates:
            params.aggression = max(0.0, min(1.0, float(updates["aggression"])))

    def _get_experience_buffer(self, driver_id: str) -> ExperienceBuffer:
        """获取或创建指定司机的经验缓冲区。"""
        if driver_id not in self._experience_buffer:
            self._experience_buffer[driver_id] = ExperienceBuffer(max_size=10)
        return self._experience_buffer[driver_id]

    def _record_experiment(self, driver_id: str, day: int, stat: dict) -> None:
        """将当天的参数快照+结果记录到经验缓冲区（OPRO 核心）。"""
        params = self.get_params(driver_id)
        # 快照关键可调参数
        params_snapshot = {
            "aggression": params.aggression,
            "wait_score_threshold": params.wait_score_threshold,
            "wait_value_multiplier": params.wait_value_multiplier,
            "deadhead_penalty_factor": params.deadhead_penalty_factor,
            "efficiency_bonus_factor": params.efficiency_bonus_factor,
            "position_bonus_factor": params.position_bonus_factor,
            "time_cost_per_minute": params.time_cost_per_minute,
        }
        # 结果指标（含罚分风险估算）
        penalty_risk = self._estimate_penalty_risk(driver_id, day)
        result = {
            "income": stat.get("income", 0),
            "deadhead_ratio": stat.get("deadhead_ratio", 0),
            "orders": stat.get("orders", 0),
            "penalty_risk": penalty_risk,
        }
        experiment = ParameterExperiment(
            day=day, params_snapshot=params_snapshot, result=result)
        buf = self._get_experience_buffer(driver_id)
        buf.add(experiment)
        logger.debug(
            "OPRO buffer [%s] day=%d: size=%d, net=%.0f",
            driver_id, day,
            buf.size,
            result["income"] - result["penalty_risk"])

    def _estimate_penalty_risk(self, driver_id: str, day: int) -> float:
        """估算当前的罚分风险，基于约束满足情况。"""
        config = get_config(driver_id)
        if not config:
            return 0.0

        risk = 0.0
        days_elapsed = max(1, day + 1)

        # 1) Off-day 风险：如果进度落后
        if config.monthly_off_days_required > 0:
            expected_off_days = config.monthly_off_days_required * days_elapsed / 31
            stats = self._daily_stats.get(driver_id, [])
            active_days = sum(1 for s in stats if s.get("orders", 0) > 0)
            actual_off_days = days_elapsed - active_days
            if actual_off_days < expected_off_days:
                risk += config.penalty_weights.get("off_days", 3000) * 0.3

        # 2) 空驶配额风险
        if config.max_monthly_deadhead_km is not None:
            stats = self._daily_stats.get(driver_id, [])
            if stats:
                latest = stats[-1]
                ratio = latest.get("deadhead_ratio", 0)
                if ratio > 0.4:
                    risk += 500 * ratio

        # 3) 休息违规风险（简化估算）
        if config.min_continuous_rest_minutes > 0:
            stats = self._daily_stats.get(driver_id, [])
            recent = stats[-3:] if len(stats) >= 3 else stats
            high_order_days = sum(1 for s in recent if s.get("orders", 0) >= 3)
            if high_order_days >= 2:
                risk += config.penalty_weights.get("rest", 300) * high_order_days

        return risk

    def _fallback_day_adjust(self, params: StrategyParams, day: int) -> None:
        """LLM 回顾失败时的规则 fallback：按月度阶段调整。"""
        if day <= 5:
            # 月初：探索期
            params.aggression = 0.4
            params.wait_score_threshold = 10.0
            params.wait_value_multiplier = 0.7
        elif day <= 15:
            # 月中前半：稳定期
            params.aggression = 0.5
            params.wait_score_threshold = 5.0
            params.wait_value_multiplier = 0.6
        elif day <= 25:
            # 月中后半：加速期
            params.aggression = 0.6
            params.wait_score_threshold = 0.0
            params.wait_value_multiplier = 0.5
        else:
            # 月末：冲刺期
            params.aggression = 0.8
            params.wait_score_threshold = -5.0
            params.wait_value_multiplier = 0.4
            params.max_wait_minutes = 15
