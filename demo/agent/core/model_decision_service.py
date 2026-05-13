"""模型决策服务 v5：泛化版 + LLM 增强决策 + API 降级容错。

架构：
  首步: PreferenceParser(LLM) → build_config_from_parsed → register_config
  每日: StrategyAdvisor.daily_review(LLM) → 动态调参
  每步: StateTracker → SchedulePlanner → RuleEngine → CargoScorer
       → TokenBudget 判断是否值得调 LLM → StrategyAdvisor.enhance_decision
       → Custom 偏好 LLM 评估（如有）

设计原则：
  1. 首次 decide() 时用一次 LLM 将偏好文本解析为结构化配置（~3K token）
  2. 每日跨天时做一次策略回顾，动态调整决策参数（~3K token）
  3. 每步根据 TokenBudget 判断是否需要 LLM 增强：
     - 高价值场景（多货源分差小、有 custom 约束等）→ 用 LLM (~1.5K token)
     - 简单场景 → 纯规则决策（0 token）
  4. 整月预计使用 ~1.5M token，充分利用 5M 预算

v5 改进：
  - API 降级容错：LLM API 连续失败后自动切换纯规则模式，不会崩溃
  - 步骤合并优化：多次连续等待合并为大粒度步骤，减少总步数
  - 查询退避改进：空查后加大等待粒度，减少无效 query
"""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort

from agent.config.driver_config import (
    DriverConfig, get_config, register_config, build_config_from_parsed,
)
from agent.config.preference_parser import PreferenceParser
from agent.core.state_tracker import StateTracker, DriverState, haversine_km
from agent.core.schedule_planner import SchedulePlanner, ScheduleAction, ScheduleDecision
from agent.core.rule_engine import RuleEngine, FilteredCargo
from agent.scoring.cargo_scorer import CargoScorer, ScoredCargo
from agent.strategy.token_budget import TokenBudgetManager
from agent.strategy.strategy_advisor import StrategyAdvisor, StrategyParams

# 热点 reposition 参数
_HOTSPOT_REPOSITION_MIN_KM = 10.0   # 最近热点距离超过此值才值得空驶
_HOTSPOT_REPOSITION_MAX_KM = 80.0   # 超过此距离的热点不考虑
_HOTSPOT_IDLE_STEPS_THRESHOLD = 2   # 连续无货步数达到此值才触发
_LOAD_WAIT_REST_THRESHOLD_MIN = 20  # 装货等待超过此时间才值得插入休息

# query_cargo 动态冷却参数
_QUERY_COOLDOWN_PEAK = 8       # 高峰时段冷却（分钟）
_QUERY_COOLDOWN_NORMAL = 15    # 普通时段冷却
_QUERY_COOLDOWN_OFFPEAK = 25   # 低谷时段冷却
_QUERY_COOLDOWN_EMPTY_MULT = 1.5  # 上次查询返回空时的冷却倍率


class ModelDecisionService:
    """多阶段决策服务，实现 AgentDecisionPort.decide 接口。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")

        # 内部模块
        self._tracker = StateTracker()
        self._planner = SchedulePlanner()
        self._rule_engine = RuleEngine()
        self._scorer = CargoScorer()
        self._parser = PreferenceParser()
        self._budget = TokenBudgetManager()
        self._advisor = StrategyAdvisor()

        # 运行时状态
        self._initialized: set[str] = set()  # 已完成初始化的 driver_id
        self._recent_scores: dict[str, list[float]] = {}  # 最近 N 步最优得分
        self._steps_without_order: dict[str, int] = {}    # 连续无接单步数
        self._custom_constraints: dict[str, list[Any]] = {}  # 缓存 custom 约束
        self._last_query_sim_min: dict[str, int] = {}     # 上次 query_cargo 的仿真时刻
        self._last_query_empty: dict[str, bool] = {}      # 上次 query 是否返回空
        self._pending_action: dict[str, dict[str, Any]] = {}  # 缓存上一步 action，用于下次 decide 时更新 state

    def decide(self, driver_id: str) -> dict[str, Any]:
        """主决策入口。"""

        # 1. 获取当前状态
        status = self._api.get_driver_status(driver_id)
        state = self._tracker.init_from_status(driver_id, status)

        # 补偿更新：用上一步缓存的 action 更新 state（因为 orchestrator 不调用 update_after_action）
        if driver_id in self._pending_action:
            prev_action = self._pending_action.pop(driver_id)
            # 构造一个最小 result 用于 update_after_action
            pseudo_result = {"simulation_progress_minutes": state.sim_minutes}
            self._tracker.update_after_action(state, prev_action, pseudo_result)

        # 首次调用时：解析偏好 → 注册配置 → 重建历史
        if driver_id not in self._initialized:
            self._first_step_init(driver_id, status, state)
            self._initialized.add(driver_id)

        config = get_config(driver_id)

        # 2. 每日策略回顾
        self._maybe_daily_review(driver_id, state, config)

        self._logger.info(
            "step=%d driver=%s time=%d loc=(%.4f,%.4f) income=%.1f dist=%.1f budget_used=%d",
            state.step_count, driver_id, state.sim_minutes,
            state.current_lat, state.current_lng,
            state.total_gross_income, state.total_distance_km,
            self._budget.total_used,
        )

        # 3. 时间规划器：判断当前应该做什么
        schedule = self._planner.plan(state, config)
        self._logger.info("schedule: %s", schedule)

        # 4. 根据调度决策执行
        action: dict[str, Any] | None = None

        if schedule.action == ScheduleAction.REST:
            action = self._make_wait(schedule.wait_minutes)

        elif schedule.action == ScheduleAction.OFF_DAY:
            # Off-day 一次性等待更长，减少步骤
            action = self._make_batch_wait(schedule.wait_minutes)

        elif schedule.action == ScheduleAction.GO_HOME:
            action = self._make_reposition(config.home_pos[0], config.home_pos[1])

        elif schedule.action == ScheduleAction.REPOSITION:
            if schedule.target_pos:
                action = self._make_reposition(schedule.target_pos[0], schedule.target_pos[1])
            else:
                action = self._make_wait(30)

        elif schedule.action == ScheduleAction.FAMILY_EVENT:
            action = self._make_wait(schedule.wait_minutes)

        else:
            # 5. WORK 模式：查询货源、过滤、评分、决策
            action = self._work_mode(driver_id, state, config, status)

        # 缓存本步 action，下次 decide 时用于补偿更新 state
        self._pending_action[driver_id] = action
        return action

    # =========================================================================
    # 首步初始化：偏好解析 + 配置注册 + 历史重建
    # =========================================================================

    def _first_step_init(self, driver_id: str, status: dict[str, Any],
                         state: DriverState) -> None:
        """首次 decide() 调用时的完整初始化流程。"""
        # Step 1: 用 LLM 解析偏好为结构化约束
        parsed = self._parser.parse(status, self._api)
        self._budget.record_usage("parse", 4000)  # 估算

        self._logger.info(
            "preference parsed for %s: rest=%d quiet=%d forbidden=%d go_home=%d custom=%d",
            driver_id,
            len(parsed.rest_constraints),
            len(parsed.quiet_windows),
            len(parsed.forbidden_categories),
            len(parsed.go_home),
            len(parsed.custom),
        )

        # 缓存 custom 约束（后续逐步 LLM 评估用）
        if parsed.custom:
            self._custom_constraints[driver_id] = parsed.custom

        # Step 2: 将结构化约束转为 DriverConfig 并注册
        config = build_config_from_parsed(parsed)
        # 从 status 同步仿真引擎的真实速度到决策配置
        speed_kmph = float(status.get("reposition_speed_km_per_hour", 48.0))
        config.reposition_speed_kmpm = speed_kmph / 60.0
        register_config(driver_id, config)
        self._logger.info("dynamic config registered for %s (speed=%.2f km/min)", driver_id, config.reposition_speed_kmpm)

        # Step 3: 更新 CargoScorer 的 cost_per_km
        self._scorer.cost_per_km = config.cost_per_km

        # Step 4: 从历史记录重建累计状态
        self._init_from_history(driver_id, state)

        # Step 5: HotspotTracker 冷启动 — 若无历史种子，用司机初始位置
        if not self._scorer.hotspot_tracker.get_hotspots():
            if state.current_lat != 0.0 or state.current_lng != 0.0:
                self._scorer.hotspot_tracker.observe(
                    [{"pickup_lat": state.current_lat, "pickup_lng": state.current_lng}],
                    state.current_day(),
                )
                self._logger.info(
                    "HotspotTracker cold start: seeded with driver initial pos (%.4f, %.4f)",
                    state.current_lat, state.current_lng)

    def _init_from_history(self, driver_id: str, state: DriverState) -> None:
        """从历史记录重建状态，并提取货源模式种子 HotspotTracker。"""
        try:
            hist = self._api.query_decision_history(driver_id, 9999)
            records = hist.get("records", [])
            if records:
                self._tracker.rebuild_from_history(driver_id, records)
                self._logger.info("rebuilt state from %d history records", len(records))

                # 从历史接单记录中提取 pickup 位置，种子化 HotspotTracker
                seed_cargos: list[dict[str, Any]] = []
                for rec in records:
                    action_obj = rec.get("action", {})
                    result = rec.get("result", {})
                    if (str(action_obj.get("action", "")).lower() == "take_order"
                            and bool(result.get("accepted", False))):
                        # 从 result 中提取 pickup 坐标作为货源位置
                        pickup_lat = float(result.get("pickup_lat", 0.0))
                        pickup_lng = float(result.get("pickup_lng", 0.0))
                        if pickup_lat != 0.0 or pickup_lng != 0.0:
                            seed_cargos.append({
                                "pickup_lat": pickup_lat,
                                "pickup_lng": pickup_lng,
                            })
                if seed_cargos:
                    self._scorer.hotspot_tracker.observe(seed_cargos, state.current_day())
                    self._logger.info(
                        "seeded HotspotTracker with %d historical pickup locations",
                        len(seed_cargos))
        except Exception as e:
            self._logger.warning("history rebuild failed: %s", e)

    # =========================================================================
    # 每日策略回顾
    # =========================================================================

    def _maybe_daily_review(self, driver_id: str, state: DriverState,
                            config: DriverConfig) -> None:
        """在跨天时触发策略回顾。

        优化：前5天和收入为0时使用规则fallback，节省LLM调用时间。
        仅在有实际运营数据（day>=5且有收入）时才调用LLM做精细调参。
        v5 增强：API 降级时直接用规则 fallback。
        """
        current_day = state.current_day()
        if not self._advisor.should_review(driver_id, current_day):
            return

        # 优化：前5天或无收入时直接用规则fallback，不调LLM
        # 这些场景下LLM返回的建议价值不大（总是“提高激进度”）
        if current_day <= 5 or state.total_gross_income <= 0:
            params = self._advisor.get_params(driver_id)
            self._advisor._fallback_day_adjust(params, current_day)
            self._advisor._last_review_day[driver_id] = current_day
            self._logger.info(
                "daily review day=%d: using rule fallback (early/no-income), aggression=%.2f",
                current_day, params.aggression)
            return

        # API 降级时直接用 fallback
        if self._advisor.is_degraded:
            params = self._advisor.get_params(driver_id)
            self._advisor._fallback_day_adjust(params, current_day)
            self._advisor._last_review_day[driver_id] = current_day
            self._logger.info(
                "daily review day=%d: using rule fallback (API degraded), aggression=%.2f",
                current_day, params.aggression)
            return

        if not self._budget.should_do_daily_review(current_day,
                                                    self._advisor._last_review_day.get(driver_id, -1)):
            # 预算不够做回顾，用 fallback
            params = self._advisor.get_params(driver_id)
            self._advisor._fallback_day_adjust(params, current_day)
            self._advisor._last_review_day[driver_id] = current_day
            return

        self._advisor.daily_review(driver_id, state, config, self._api)
        self._budget.record_usage("daily_review", 3000)  # 估算

        # v5: 弹性预算再分配 — 每日回顾后评估预算使用率
        self._budget.recalibrate(current_day)
        self._budget.rebalance_budget(current_day, api_degraded=self._advisor.is_degraded)

    # =========================================================================
    # 工作模式（拆分为子流程）
    # =========================================================================

    def _work_mode(self, driver_id: str, state: DriverState,
                   config: DriverConfig, status: dict[str, Any]) -> dict[str, Any]:
        """正常工作模式：查货 → 过滤 → 评分 → 智能决策。"""
        params = self._advisor.get_params(driver_id)

        # 阶段 1: 查询 & 过滤
        scored = self._query_filter_and_score(driver_id, state, config, params)

        if not scored:
            self._steps_without_order[driver_id] = (
                self._steps_without_order.get(driver_id, 0) + 1)
            # 无货源时尝试主动 reposition 到热点区域
            reposition_action = self._try_hotspot_reposition(
                driver_id, state, config, params)
            if reposition_action is not None:
                return reposition_action
            return self._make_wait(self._smart_wait(state, params))

        best = scored[0]
        avg_recent = self._get_avg_recent_score(driver_id)
        self._record_score(driver_id, best.score)

        # 阶段 2: LLM 增强决策（如果预算允许）
        llm_action = self._try_llm_enhanced(
            driver_id, state, config, params, scored)
        if llm_action is not None:
            # 检查是否需要插入休息（装货等待期利用）
            if llm_action.get("action") == "take_order":
                pre_rest = self._maybe_pre_rest_for_load_wait(
                    llm_action, scored, state, config)
                if pre_rest is not None:
                    return pre_rest
            return llm_action

        # 阶段 3: 纯规则决策
        action = self._rule_based_decision(
            driver_id, state, config, params, scored, avg_recent)

        # 检查是否需要插入休息（装货等待期利用）
        if action.get("action") == "take_order":
            pre_rest = self._maybe_pre_rest_for_load_wait(
                action, scored, state, config)
            if pre_rest is not None:
                return pre_rest

        return action

    # ----- 子流程 1: 查询、过滤、评分 -----

    def _query_filter_and_score(
        self, driver_id: str, state: DriverState,
        config: DriverConfig, params: StrategyParams,
    ) -> list[ScoredCargo]:
        """查询附近货源 → 特殊单必接 → 规则过滤 → 评分排序。

        返回空列表表示本轮无可用货源。
        副作用：将查到的货源喂给 HotspotTracker。

        查询冷却：如果距上次 query_cargo 不足 _QUERY_COOLDOWN_MINUTES 分钟，
        跳过查询直接返回空列表，避免浪费 scan_minutes 仿真时间。
        """
        # 动态查询冷却：根据时段和供给密度调整冷却时间
        cooldown = self._compute_query_cooldown(driver_id, state)
        last_query = self._last_query_sim_min.get(driver_id, -9999)
        if state.sim_minutes - last_query < cooldown:
            self._logger.debug("query cooldown: skip (last=%d, now=%d, cd=%d)", last_query, state.sim_minutes, cooldown)
            return []

        lat, lng = state.current_lat, state.current_lng
        cargo_resp = self._api.query_cargo(
            driver_id=driver_id, latitude=lat, longitude=lng)
        self._last_query_sim_min[driver_id] = state.sim_minutes
        items = cargo_resp.get("items", [])
        self._logger.info("query_cargo returned %d items", len(items))

        if not items:
            self._last_query_empty[driver_id] = True
            return []

        self._last_query_empty[driver_id] = False
        cargos = self._extract_cargos(items)

        # 喂给热点追踪器
        self._scorer.hotspot_tracker.observe(cargos, state.current_day())

        # 喂给供给预测器
        self._scorer.supply_predictor.observe(
            cargos, state.sim_minutes, state.current_lat, state.current_lng)

        # 特殊货源检查：必接
        for cargo in cargos:
            cargo_id = str(cargo.get("cargo_id", ""))
            if self._rule_engine.should_take_special_cargo(cargo_id, state, config):
                self._logger.info("special cargo detected: %s, must take", cargo_id)
                self._steps_without_order[driver_id] = 0
                # 返回一个哨兵列表，由调用方直接处理
                # 用 score=math.inf 保证它在后续流程中被选中
                from agent.core.rule_engine import FilteredCargo as _FC
                sentinel = ScoredCargo(
                    filtered=_FC(cargo=cargo, pickup_km=0, haul_km=0,
                                 is_soft_violated=False),
                    score=float("inf"),
                    breakdown={"special": float("inf")},
                )
                return [sentinel]

        # 规则引擎过滤
        filtered = self._rule_engine.filter_cargos(cargos, state, config)
        self._logger.info("rule engine: %d/%d passed", len(filtered), len(cargos))

        if not filtered:
            return []

        # 评分排序（传入策略参数）
        return self._scorer.score_and_rank(
            filtered, state, config, top_k=5, params=params)

    # ----- 子流程 2: LLM 增强决策 -----

    def _try_llm_enhanced(
        self, driver_id: str, state: DriverState,
        config: DriverConfig, params: StrategyParams,
        scored: list[ScoredCargo],
    ) -> dict[str, Any] | None:
        """尝试 LLM 增强决策。返回动作字典或 None（不使用/无结论）。"""
        best = scored[0]

        # 特殊单直接接，不走 LLM
        if best.score == float("inf"):
            cargo_id = str(best.cargo.get("cargo_id", ""))
            return self._make_take_order(cargo_id)

        score_gap = (scored[0].score - scored[1].score) if len(scored) >= 2 else 999
        use_llm = self._budget.should_use_llm_for_decision({
            "cargo_count": len(scored),
            "score_gap": score_gap,
            "best_score": best.score,
            "has_custom_constraints": bool(self._custom_constraints.get(driver_id)),
            "current_day": state.current_day(),
            "steps_without_order": self._steps_without_order.get(driver_id, 0),
        })

        if not use_llm:
            return None

        enhanced_candidates = [
            {
                "cargo_id": sc.cargo.get("cargo_id"),
                "price": sc.cargo.get("price"),
                "pickup_km": sc.pickup_km,
                "haul_km": sc.haul_km,
                "score": sc.score,
                "category": sc.cargo.get("category", ""),
                "delivery_lat": sc.cargo.get("delivery_lat", 0),
                "delivery_lng": sc.cargo.get("delivery_lng", 0),
            }
            for sc in scored[:5]
        ]

        llm_result = self._advisor.enhance_decision(
            driver_id, state, config, enhanced_candidates, self._api)
        self._budget.record_usage("decision", 1500)
        # v5: 同步 API 健康状态到预算管理器
        self._budget.notify_api_status(
            success=not self._advisor.is_degraded,
            step=state.step_count)

        if not llm_result:
            return None

        if llm_result.get("action") == "take_order":
            self._steps_without_order[driver_id] = 0
            blocked = self._check_custom_constraints(
                driver_id, state, config, "take_order",
                llm_result.get("params", {}), best.score)
            if blocked:
                return self._make_wait(self._smart_wait(state, params))
            return llm_result

        # LLM 建议等待
        self._steps_without_order[driver_id] = (
            self._steps_without_order.get(driver_id, 0) + 1)
        return llm_result

    # ----- 子流程 3: 纯规则决策 -----

    def _rule_based_decision(
        self, driver_id: str, state: DriverState,
        config: DriverConfig, params: StrategyParams,
        scored: list[ScoredCargo], avg_recent: float,
    ) -> dict[str, Any]:
        """无 LLM 或 LLM 无结论时的规则决策。"""
        best = scored[0]

        # 特殊单直接接
        if best.score == float("inf"):
            cargo_id = str(best.cargo.get("cargo_id", ""))
            return self._make_take_order(cargo_id)

        adjusted_threshold = params.wait_score_threshold * (1 - params.aggression)

        if best.score > adjusted_threshold:
            # 正分货源：值得接
            blocked = self._check_custom_constraints(
                driver_id, state, config, "take_order",
                {"cargo_id": str(best.cargo.get("cargo_id", ""))},
                best.score)
            if blocked:
                # 尝试次优
                if len(scored) >= 2:
                    best = scored[1]
                else:
                    return self._make_wait(self._smart_wait(state, params))

            cargo_id = str(best.cargo.get("cargo_id", ""))
            self._logger.info("take best cargo: %s (score=%.2f)", cargo_id, best.score)
            self._steps_without_order[driver_id] = 0
            return self._make_take_order(cargo_id)

        # 低分：判断等待 vs 勉强接
        wait_value = self._compute_smart_wait_value(state, config, avg_recent, params)
        if wait_value > best.score:
            self._logger.info("wait (score=%.2f < wait_value=%.2f)", best.score, wait_value)
            self._steps_without_order[driver_id] = (
                self._steps_without_order.get(driver_id, 0) + 1)
            return self._make_wait(self._smart_wait(state, params))

        # 等待价值也低，接最优的负分单
        cargo_id = str(best.cargo.get("cargo_id", ""))
        self._logger.info(
            "reluctant take: %s (score=%.2f, wait=%.2f)",
            cargo_id, best.score, wait_value)
        self._steps_without_order[driver_id] = 0
        return self._make_take_order(cargo_id)

    # ----- 子流程 4: Custom 约束检查 -----

    def _check_custom_constraints(
        self, driver_id: str, state: DriverState,
        config: DriverConfig, action: str,
        action_params: dict[str, Any], best_score: float,
    ) -> bool:
        """检查 custom 约束是否阻止执行动作。返回 True 表示被阻止。"""
        customs = self._custom_constraints.get(driver_id, [])
        if not customs or not self._budget.can_spend("custom_eval", 1000):
            return False

        violates, penalty = self._advisor.evaluate_custom_constraints(
            driver_id, state, config, customs,
            action, action_params, self._api)
        self._budget.record_usage("custom_eval", 1000)
        if violates and penalty > best_score:
            self._logger.info(
                "custom constraint blocks %s, penalty=%.1f > score=%.1f",
                action, penalty, best_score)
            return True
        return False

    # ----- 子流程 5: 热点主动 reposition -----

    def _try_hotspot_reposition(
        self, driver_id: str, state: DriverState,
        config: DriverConfig, params: StrategyParams,
    ) -> dict[str, Any] | None:
        """附近无货时，主动空驶到最近的货源密集区。

        触发条件：
          1. 连续无货步数 >= 阈值
          2. HotspotTracker 有数据
          3. 最近热点距离在 [min_km, max_km] 范围内
          4. 空驶不会超出月度空驶配额
        """
        idle_steps = self._steps_without_order.get(driver_id, 0)
        if idle_steps < _HOTSPOT_IDLE_STEPS_THRESHOLD:
            return None

        hotspots = self._scorer.hotspot_tracker.get_hotspots()
        if not hotspots:
            return None

        lat, lng = state.current_lat, state.current_lng
        best_hs = None
        best_dist = float("inf")
        for hs in hotspots:
            d = haversine_km(lat, lng, hs[0], hs[1])
            if d < best_dist:
                best_dist = d
                best_hs = hs

        if best_hs is None:
            return None

        # 太近了没必要动，太远了不划算
        if best_dist < _HOTSPOT_REPOSITION_MIN_KM or best_dist > _HOTSPOT_REPOSITION_MAX_KM:
            return None

        # 空驶配额检查
        if config.max_monthly_deadhead_km is not None:
            remaining = config.max_monthly_deadhead_km - state.total_deadhead_km
            if remaining < best_dist * 1.5:  # 留余量
                return None

        self._logger.info(
            "hotspot reposition: idle %d steps, moving %.1fkm to (%.4f,%.4f)",
            idle_steps, best_dist, best_hs[0], best_hs[1])
        return self._make_reposition(best_hs[0], best_hs[1])

    # ----- 子流程 6: 装货等待期休息插入 -----

    def _maybe_pre_rest_for_load_wait(
        self, take_action: dict[str, Any],
        scored: list[ScoredCargo], state: DriverState,
        config: DriverConfig,
    ) -> dict[str, Any] | None:
        """接单前检查装货等待窗口，若等待较长且今日休息未满足，先插入 wait 累积休息。

        触发条件：
          1. 今日休息未满足
          2. 装货等待 >= 阈值
          3. 等待期间不会错过装货窗口
        返回 wait 动作或 None（不需要插入）。
        """
        # 今日休息已满足，无需插入
        if config.min_continuous_rest_minutes <= 0:
            return None
        rest_deficit = config.min_continuous_rest_minutes - state.longest_rest_today
        if rest_deficit <= 0:
            return None

        # 找到对应的 ScoredCargo
        cargo_id = take_action.get("params", {}).get("cargo_id", "")
        target_sc = None
        for sc in scored:
            if str(sc.cargo.get("cargo_id", "")) == str(cargo_id):
                target_sc = sc
                break
        if target_sc is None:
            return None

        # 估算装货等待时间
        load_wait = self._scorer.estimate_load_wait_for_cargo(
            target_sc.cargo, state, target_sc.pickup_km, config)
        if load_wait < _LOAD_WAIT_REST_THRESHOLD_MIN:
            return None

        # 插入的 wait 时长 = min(休息缺口, 装货等待时间 - 安全余量)
        # 留 5 分钟余量确保赶得上装货窗口
        safe_wait = max(0, int(load_wait) - 5)
        rest_wait = min(rest_deficit, safe_wait)
        if rest_wait < 10:  # 太短不值得
            return None

        self._logger.info(
            "pre-rest before load: cargo=%s load_wait=%.0fmin "
            "rest_deficit=%dmin -> inserting %dmin wait",
            cargo_id, load_wait, rest_deficit, rest_wait)
        return self._make_wait(rest_wait)

    # =========================================================================
    # 智能等待
    # =========================================================================

    def _smart_wait(self, state: DriverState, params: StrategyParams) -> int:
        """根据策略参数和时段计算智能等待时长。

        v5 优化：
          - 低谷时段等待上限提升到 240min（合并更多步骤）
          - 连续空查递增更激进，减少无效步骤
          - 深夜时段直接等到早高峰，避免空转
        """
        hour = state.hour_in_day()
        base_wait = params.max_wait_minutes

        # 高峰时段少等（货多）
        is_peak = any(start <= hour <= end for start, end in params.peak_hours)
        if is_peak:
            base_wait = max(params.min_wait_minutes, base_wait // 2)

        # 低谷时段多等：深夜时直接等到早高峰
        if hour >= 22 or hour <= 5:
            # 计算距离下一个高峰开始的分钟数
            if hour >= 22:
                minutes_to_peak = (24 - hour + 8) * 60  # 等到明早 8 点
            else:
                minutes_to_peak = (8 - hour) * 60
            # 在 [base_wait*3, 等到早高峰] 之间取较小值，上限 240
            base_wait = min(240, max(base_wait * 3, min(minutes_to_peak, 240)))

        # 激进模式少等
        base_wait = int(base_wait * (1 - params.aggression * 0.5))

        # 连续无货时递增等待（更激进的递增）
        idle_steps = self._steps_without_order.get(state.driver_id, 0)
        if idle_steps >= 2:
            base_wait = int(base_wait * min(3.0, 1.0 + idle_steps * 0.5))

        return max(params.min_wait_minutes, min(240, base_wait))

    def _compute_smart_wait_value(self, state: DriverState, config: DriverConfig,
                                  avg_score_recent: float, params: StrategyParams) -> float:
        """增强版等待价值估算：结合供给预测器和历史均分。"""
        hour = state.hour_in_day()
        day = state.current_day()

        # 尝试用供给预测器的数据驱动估算
        predicted_wait = self._scorer.supply_predictor.predict_wait_value(
            state.current_lat, state.current_lng, state.sim_minutes,
            wait_minutes=params.max_wait_minutes)

        # 融合预测值和历史均分
        if predicted_wait > 0 and avg_score_recent > 0:
            # 有两个信号源时做加权融合
            base_wait_value = predicted_wait * 0.6 + avg_score_recent * params.wait_value_multiplier * 0.4
        elif predicted_wait > 0:
            base_wait_value = predicted_wait * params.wait_value_multiplier
        else:
            base_wait_value = avg_score_recent * params.wait_value_multiplier

        # 时段调整（供给预测器已部分包含时段信息，这里做轻微调整）
        is_peak = any(start <= hour <= end for start, end in params.peak_hours)
        if is_peak:
            base_wait_value *= 1.15
        elif hour >= 22 or hour <= 5:
            base_wait_value *= 0.5

        # 月末压力
        if day >= 25:
            base_wait_value *= (1 - params.aggression)

        # 连续等待衰减
        steps_waiting = self._steps_without_order.get(state.driver_id, 0)
        if steps_waiting >= 3:
            base_wait_value *= max(0.3, 1 - steps_waiting * 0.15)

        return base_wait_value

    # =========================================================================
    # 辅助工具
    # =========================================================================

    def _extract_cargos(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """从 query_cargo 返回的 items 中提取标准化的 cargo 字典。

        原始数据使用嵌套结构 start.lat/start.lng 和 end.lat/end.lng，
        以及 cargo_name 作为品类字段。此方法将其展平为 rule_engine 和
        cargo_scorer 期望的 pickup_lat/pickup_lng/delivery_lat/delivery_lng/category。
        """
        cargos = []
        for item in items:
            cargo = item.get("cargo", {})
            if not cargo:
                continue
            cargo_copy = dict(cargo)
            cargo_copy["query_distance_km"] = item.get("distance_km", 0.0)

            # 展平嵌套的地理坐标
            start = cargo.get("start", {})
            end = cargo.get("end", {})
            if isinstance(start, dict):
                cargo_copy.setdefault("pickup_lat", float(start.get("lat", 0.0)))
                cargo_copy.setdefault("pickup_lng", float(start.get("lng", 0.0)))
            if isinstance(end, dict):
                cargo_copy.setdefault("delivery_lat", float(end.get("lat", 0.0)))
                cargo_copy.setdefault("delivery_lng", float(end.get("lng", 0.0)))

            # 映射品类字段
            if "category" not in cargo_copy and "cargo_name" in cargo_copy:
                cargo_copy["category"] = cargo_copy["cargo_name"]

            cargos.append(cargo_copy)
        return cargos

    def _make_take_order(self, cargo_id: str) -> dict[str, Any]:
        return {"action": "take_order", "params": {"cargo_id": cargo_id}}

    def _make_reposition(self, lat: float, lng: float) -> dict[str, Any]:
        return {"action": "reposition", "params": {"latitude": lat, "longitude": lng}}

    def _make_wait(self, minutes: int) -> dict[str, Any]:
        minutes = max(1, min(minutes, 240))
        return {"action": "wait", "params": {"duration_minutes": minutes}}

    def _make_batch_wait(self, minutes: int) -> dict[str, Any]:
        """大粒度等待，用于休息等确定性场景，最大480分钟。"""
        minutes = max(1, min(minutes, 480))
        return {"action": "wait", "params": {"duration_minutes": minutes}}

    def _compute_query_cooldown(self, driver_id: str, state: DriverState) -> int:
        """根据时段和供给密度动态计算 query 冷却时间。

        优化：增大冷却上限到80分钟，连续空查时指数退避，减少无效步骤。
        """
        hour = state.hour_in_day()

        # 基础冷却：按时段分级
        if 8 <= hour <= 11 or 14 <= hour <= 18:
            base_cd = _QUERY_COOLDOWN_PEAK
        elif hour >= 22 or hour <= 5:
            base_cd = _QUERY_COOLDOWN_OFFPEAK
        else:
            base_cd = _QUERY_COOLDOWN_NORMAL

        # 供给密度调整：供给丰富时缩短冷却
        richness = self._scorer.supply_predictor.get_supply_richness(
            state.current_lat, state.current_lng, state.sim_minutes)
        # richness 0~1，高供给时缩短冷却
        density_factor = 1.0 - richness * 0.3  # 最多缩短 30%

        # 上次查询返回空时延长冷却（增强退避）
        if self._last_query_empty.get(driver_id, False):
            idle_steps = self._steps_without_order.get(driver_id, 0)
            # 连续无货时指数退避：1.5x, 2.25x, 3x...
            backoff = min(3.0, _QUERY_COOLDOWN_EMPTY_MULT * (1.0 + idle_steps * 0.3))
            density_factor *= backoff

        cooldown = int(base_cd * density_factor)
        return max(5, min(80, cooldown))  # 限制在 5~80 分钟

    def _record_score(self, driver_id: str, score: float) -> None:
        if driver_id not in self._recent_scores:
            self._recent_scores[driver_id] = []
        scores = self._recent_scores[driver_id]
        scores.append(score)
        if len(scores) > 20:
            self._recent_scores[driver_id] = scores[-20:]

    def _get_avg_recent_score(self, driver_id: str) -> float:
        scores = self._recent_scores.get(driver_id, [])
        if not scores:
            return 50.0
        return sum(scores) / len(scores)
