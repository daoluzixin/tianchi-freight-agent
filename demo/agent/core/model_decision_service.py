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
from agent.scoring.experience_tracker import ExperienceTracker, hour_to_time_slot, pos_to_region_key
from agent.core.timeline_projector import SCAN_COST_MINUTES, SAFETY_BUFFER_MINUTES
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
        self._experience = ExperienceTracker()

        # 将经验追踪器注入 StrategyAdvisor，供 daily_review 使用
        self._advisor._experience_tracker = self._experience

        # 运行时状态
        self._initialized: set[str] = set()  # 已完成初始化的 driver_id
        self._recent_scores: dict[str, list[float]] = {}  # 最近 N 步最优得分
        self._steps_without_order: dict[str, int] = {}    # 连续无接单步数
        self._custom_constraints: dict[str, list[Any]] = {}  # 缓存 custom 约束
        self._last_query_sim_min: dict[str, int] = {}     # 上次 query_cargo 的仿真时刻
        self._last_query_empty: dict[str, bool] = {}      # 上次 query 是否返回空
        self._pending_action: dict[str, dict[str, Any]] = {}  # 缓存上一步 action，用于下次 decide 时更新 state
        self._pending_issued_at: dict[str, int] = {}  # action 发出时的 sim_minutes，用于跨天 order_days 补偿
        self._known_pref_count: dict[str, int] = {}  # 已知偏好数量（用于检测新偏好解锁）

    def decide(self, driver_id: str) -> dict[str, Any]:
        """主决策入口。"""

        # 1. 获取当前状态
        #    关键时序：当有 pending_action 时，必须先补偿状态再做跨天检测。
        #    否则前一天的 take_order 还未记入 order_days，_check_day_rollover
        #    会误将有接单的日子标记为 off-day，导致 off_days 计数虚高，
        #    真正需要的 planned off-day 被跳过。
        status = self._api.get_driver_status(driver_id)
        has_pending = driver_id in self._pending_action
        state = self._tracker.init_from_status(
            driver_id, status, skip_rollover=has_pending)

        # 补偿更新：用上一步缓存的 action 更新 state（因为 orchestrator 不调用 update_after_action）
        if has_pending:
            prev_action = self._pending_action.pop(driver_id)
            issued_at = self._pending_issued_at.pop(driver_id, None)

            # 跨天 take_order 补丁：如果 action 是 take_order 且发出时在前一天，
            # 确保发出日也记入 order_days，避免被误判为 off-day。
            action_name = str(prev_action.get("action", "")).lower()
            if action_name == "take_order" and issued_at is not None:
                issued_day = issued_at // 1440
                current_day = state.sim_minutes // 1440
                if issued_day < current_day:
                    state.order_days.add(issued_day)
                    self._logger.info(
                        "cross-day order fix: take_order issued day %d, completed day %d, "
                        "added day %d to order_days",
                        issued_day, current_day, issued_day)

            # 构造一个最小 result 用于 update_after_action
            # 注意：对于 take_order，必须标记 accepted=True，否则 order_days 不会被更新
            pseudo_result: dict[str, Any] = {"simulation_progress_minutes": state.sim_minutes}
            if action_name == "take_order":
                pseudo_result["accepted"] = True
                # P1 fix: 传入 pickup_deadhead_km 以正确累计空驶里程
                pickup_km = float(prev_action.get("_pickup_km", 0.0))
                if pickup_km > 0:
                    pseudo_result["pickup_deadhead_km"] = pickup_km
            self._tracker.update_after_action(state, prev_action, pseudo_result)

            # 经验回填：上一步如果是 take_order 且 accepted，回填实际结果
            if action_name == "take_order" and pseudo_result.get("accepted"):
                delivery_lat = float(status.get("current_lat", 0.0))
                delivery_lng = float(status.get("current_lng", 0.0))
                income = float(pseudo_result.get("income", 0.0))
                if income <= 0:
                    income = float(prev_action.get("_cargo_price", 0.0))
                self._experience.settle_pending(
                    driver_id, income, delivery_lat, delivery_lng, state.sim_minutes)
            else:
                self._experience.discard_pending(driver_id)

            # 补偿完成后再做跨天检测：此时 order_days 已经包含前一步的接单信息
            self._tracker.check_day_rollover(state)

            # 修复跨天 streak 污染：如果 pending action 是 wait 且跨天了，
            # 评测系统按自然日边界切割 wait，跨天 streak 不算当天 rest，必须重置。
            if action_name == "wait":
                duration = int(prev_action.get("params", {}).get("duration_minutes", 0))
                wait_start = state.sim_minutes - duration
                wait_start_day = wait_start // 1440
                wait_end_day = state.sim_minutes // 1440
                if wait_start_day < wait_end_day:
                    # wait 跨天了：只有当天的部分可计入 longest_rest_today
                    today_start = wait_end_day * 1440
                    today_portion = state.sim_minutes - today_start
                    state.longest_rest_today = today_portion
                    state.rest_satisfied_today = False
                    self._logger.info(
                        "cross-day wait correction: streak=%d but today_portion=%d, "
                        "longest_rest_today reset to %d",
                        state.current_rest_streak, today_portion, state.longest_rest_today)

        # 首次调用时：解析偏好 → 注册配置 → 重建历史
        if driver_id not in self._initialized:
            self._first_step_init(driver_id, status, state)
            self._initialized.add(driver_id)

        # 动态偏好解锁检测：某些偏好有时间窗口（start_time/end_time），
        # 仿真引擎只在当前仿真时间落入窗口时才返回该偏好。
        # 如果当前返回的 preferences 数量比上次多，说明有新偏好解锁，需重新解析。
        current_prefs = status.get("preferences", [])
        prev_count = self._known_pref_count.get(driver_id, 0)
        if len(current_prefs) > prev_count:
            self._logger.warning(
                "NEW PREFERENCES UNLOCKED for %s: %d -> %d, re-parsing config",
                driver_id, prev_count, len(current_prefs))
            self._reparse_preferences(driver_id, status, state)
        self._known_pref_count[driver_id] = len(current_prefs)

        config = get_config(driver_id)

        # 2. 每日经验衰减 + 策略回顾
        self._experience.daily_decay(driver_id, state.current_day())
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
            # 使用 _make_batch_wait 允许最多 480min，避免截断大粒度休息
            action = self._make_batch_wait(schedule.wait_minutes)

        elif schedule.action == ScheduleAction.OFF_DAY:
            # Off-day 一次性等待更长，减少步骤
            action = self._make_batch_wait(schedule.wait_minutes)

        elif schedule.action == ScheduleAction.GO_HOME:
            action = self._make_reposition(
                config.home_pos[0], config.home_pos[1],
                state.current_lat, state.current_lng)

        elif schedule.action == ScheduleAction.REPOSITION:
            if schedule.target_pos:
                action = self._make_reposition(
                    schedule.target_pos[0], schedule.target_pos[1],
                    state.current_lat, state.current_lng)
            else:
                action = self._make_wait(30)

        elif schedule.action == ScheduleAction.FAMILY_EVENT:
            # 家事等待可能很长（如在家休息到事件结束），使用 batch wait
            action = self._make_batch_wait(schedule.wait_minutes)

        else:
            # 5. WORK 模式：查询货源、过滤、评分、决策
            action = self._work_mode(driver_id, state, config, status)

        # 缓存本步 action，下次 decide 时用于补偿更新 state
        # 同时记录发出时间戳，用于跨天 take_order 的 order_days 补偿
        self._pending_action[driver_id] = action
        self._pending_issued_at[driver_id] = state.sim_minutes
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

    def _reparse_preferences(self, driver_id: str, status: dict[str, Any],
                              state: DriverState) -> None:
        """动态重新解析偏好（当检测到新偏好解锁时调用）。

        只更新偏好相关的配置字段（family_event、special_cargo 等），
        不重置运行时状态（rest_calendar、off_day 等）。
        """
        # 清除偏好解析器缓存，强制重新解析
        if driver_id in self._parser._cache:
            del self._parser._cache[driver_id]

        parsed = self._parser.parse(status, self._api)
        self._budget.record_usage("reparse", 2000)

        self._logger.info(
            "re-parsed preferences for %s: rest=%d quiet=%d forbidden=%d "
            "go_home=%d special=%d family=%d visit=%d custom=%d",
            driver_id,
            len(parsed.rest_constraints),
            len(parsed.quiet_windows),
            len(parsed.forbidden_categories),
            len(parsed.go_home),
            len(parsed.special_cargos),
            len(parsed.family_events),
            len(parsed.visit_targets),
            len(parsed.custom),
        )

        # 重建完整配置（包含新解锁的偏好）
        new_config = build_config_from_parsed(parsed)
        speed_kmph = float(status.get("reposition_speed_km_per_hour", 48.0))
        new_config.reposition_speed_kmpm = speed_kmph / 60.0

        # 从旧配置中保留运行时状态（不丢失已初始化的休息日历等）
        old_config = get_config(driver_id)
        if old_config:
            # 保留已规划的休息日历和 off-day 规划
            # （这些在 schedule_planner._init_rest_calendar 中初始化）
            pass  # rest_calendar 存在 DriverState 中，不在 config 中

        register_config(driver_id, new_config)
        self._logger.info("config updated for %s after preference unlock", driver_id)

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

        # R9-A: 首单 deadline 硬阻断 — 已过 deadline 且今天没接过单时，
        # 强制降低等待阈值、压低 wait_value，避免"一直等更好的"导致全天0单
        if (config.first_order_deadline_hour is not None
                and state.today_first_order_minute is None):
            hour = state.hour_in_day()
            hours_past = hour - config.first_order_deadline_hour
            if hours_past > 0:
                # 临时压低等待阈值和激进度，让 _rule_based_decision 更容易接单
                params.wait_score_threshold = min(params.wait_score_threshold, -10.0)
                params.aggression = min(1.0, params.aggression + 0.3)
                self._logger.info(
                    "R9-A first-order hard block: %dh past deadline, "
                    "threshold=%.1f aggression=%.2f",
                    hours_past, params.wait_score_threshold, params.aggression)

        # R9-C: 安静窗口二次验证 — 即使 SchedulePlanner 漏过（如偏好解析不全），
        # _work_mode 入口处再次检查：当前已在安静窗口内则直接 wait 到窗口结束。
        # 这是对 SchedulePlanner 的兜底保护，不依赖偏好解析的完整性。
        if config.quiet_window and config.quiet_window.is_active(state.sim_minutes):
            remaining = config.quiet_window.minutes_until_end(state.sim_minutes)
            self._logger.info(
                "R9-C quiet window safety net: currently in quiet window, "
                "wait %dmin until end", remaining)
            return self._make_batch_wait(min(remaining, 480))

        # 安静窗口前瞻保护：如果当前时间 + 预估 scan_cost 会进入安静窗口，
        # 直接返回 REST 而不是进入查询流程（query_cargo 会推进仿真时间）
        if config.quiet_window:
            scan_cost_estimate = SCAN_COST_MINUTES + 5  # 保守估计（基准 + 小幅富余）
            projected_time = state.sim_minutes + scan_cost_estimate
            if config.quiet_window.is_active(projected_time):
                remaining = config.quiet_window.minutes_until_end(projected_time)
                self._logger.info(
                    "quiet window lookahead: projected time %d in quiet window, rest %dmin",
                    projected_time, remaining)
                return self._make_batch_wait(min(remaining, 480))

        # 家事 deadline 保护：确保有足够时间回家，考虑接单执行时间
        if config.family_event and state.family_phase == "idle":
            fe = config.family_event
            if state.sim_minutes < fe.trigger_min:
                # 计算到第一个途经点或家的全链路行程时间
                waypoints = fe.waypoints or []
                first_wp = waypoints[0] if waypoints else None
                target_pos = (
                    (float(first_wp["lat"]), float(first_wp["lng"]))
                    if first_wp else fe.home_pos
                )
                dist_to_target = haversine_km(
                    state.current_lat, state.current_lng,
                    target_pos[0], target_pos[1])
                travel_min = dist_to_target / max(config.reposition_speed_kmpm, 0.01)
                time_budget = fe.trigger_min - state.sim_minutes
                # 保守策略：预留回家链路时间 + 一个最坏情况订单时间（约 1500min）
                # D010 的单最长 1362 分钟。我们预留 travel_time + 接单最大执行时间上限
                # 如果时间预算不足以覆盖"接一单 + 再回家"，则阻止接新单
                max_order_exec_time = 1500  # 最坏情况的单程执行时间（含 pickup + haul）
                chain_time = travel_min + 120  # 回家链路 + 缓冲
                if time_budget <= chain_time + max_order_exec_time:
                    # 时间不够接一单再回家，但还够直接回家 → 等 schedule_planner 发出回家指令
                    self._logger.info(
                        "family deadline guard: budget=%dmin <= chain=%.0fmin + max_order=%d, blocking work",
                        time_budget, chain_time, max_order_exec_time)
                    return self._make_wait(min(30, max(1, int(time_budget) // 4)))

        # Go-home 时间窗口保护：每天 deadline 前必须在家
        # 评测逻辑：deadline 时位置在家 1km 内 + 安静时段无 reposition
        # R8.4 修正：
        #   - R8.2 消除 wait(5) 死循环 → 改为直接 reposition
        #   - R8.3 发现 guard 在 deadline 后完全失效（time_to_deadline <= 0）
        #     导致 D009 17 次违规。修正：deadline 后如果不在家，仍然触发回家。
        if config.must_return_home and config.home_pos:
            current_minutes_in_day = state.sim_minutes % 1440
            deadline_min_in_day = config.home_deadline_hour * 60
            time_to_deadline = deadline_min_in_day - current_minutes_in_day

            dist_home = haversine_km(
                state.current_lat, state.current_lng,
                config.home_pos[0], config.home_pos[1])

            # 已过 deadline 但不在家 → 立即回家（今天已违规，但必须回家否则明天也违规）
            if time_to_deadline <= 0 and dist_home > 1.0:
                self._logger.info(
                    "go_home guard: PAST DEADLINE by %dmin, dist_home=%.1fkm, "
                    "reposition home now to minimize damage",
                    -time_to_deadline, dist_home)
                return self._make_reposition(
                    config.home_pos[0], config.home_pos[1],
                    state.current_lat, state.current_lng)

            if time_to_deadline > 0:
                travel_home_min = dist_home / max(config.reposition_speed_kmpm, 0.01)
                # R8.5: 动态最短一单时间（D009 平均单耗 300-600min，固定 60 严重低估）
                # 使用历史平均值的一半作为保守估计，下限 120 分钟
                avg_exec = getattr(state, 'avg_order_exec_time', None)
                min_order_time = max(120, int((avg_exec or 240) * 0.5))
                # 情况 1：连直接回家都来不及了 → 直接 reposition 回家
                if travel_home_min + SAFETY_BUFFER_MINUTES >= time_to_deadline:
                    self._logger.info(
                        "go_home guard: direct home travel=%.0fmin + buffer=%d >= "
                        "time_to_deadline=%dmin, reposition home now",
                        travel_home_min, SAFETY_BUFFER_MINUTES, time_to_deadline)
                    return self._make_reposition(
                        config.home_pos[0], config.home_pos[1],
                        state.current_lat, state.current_lng)
                # 情况 2：剩余时间不够"做一单 + 回家" → 也直接回家
                available_for_order = time_to_deadline - travel_home_min - SAFETY_BUFFER_MINUTES
                if available_for_order < min_order_time:
                    self._logger.info(
                        "go_home guard: available_for_order=%.0fmin < %dmin, "
                        "reposition home to ensure deadline",
                        available_for_order, min_order_time)
                    return self._make_reposition(
                        config.home_pos[0], config.home_pos[1],
                        state.current_lat, state.current_lng)

        # Off-day 前一天保护：如果次日是 planned off-day，
        # 且当前时间距次日 00:00 不足以安全完成一单（保守估计 1500min），
        # 则提前停止接单，等到 schedule_planner 发出 off-day lock。
        # 注意：评测系统看的是运送时间覆盖，跨天运送会导致 off-day 失效。
        if config.monthly_off_days_required > 0:
            next_day = state.current_day() + 1
            if next_day in state.planned_off_days:
                minutes_to_midnight = 1440 - (state.sim_minutes % 1440)
                # 保守策略：如果距次日 00:00 不足 720min（12h），停止接单
                # 长途单运输时间可达 24-36h，但 rule_engine 的 rest 前瞻已过滤跨天单
                # 720min 在过滤基础上提供额外保护，同时不过度牺牲接单时间
                if minutes_to_midnight <= 720:
                    self._logger.info(
                        "off-day guard: tomorrow(day %d) is planned off-day, "
                        "%dmin to midnight, blocking work",
                        next_day, minutes_to_midnight)
                    return self._make_wait(min(minutes_to_midnight, 240))

        # 步骤合并：连续无货且非高峰时段时跳过查询，直接大粒度等待
        idle_steps = self._steps_without_order.get(driver_id, 0)
        if idle_steps >= 4:
            hour = state.hour_in_day()
            is_peak = any(start <= hour <= end for start, end in params.peak_hours)
            if not is_peak:
                # 连续无货达到阈值且非高峰 → 直接大粒度等待，跳过整个查询链路
                merged_wait = self._compute_merged_wait(state, params, idle_steps)
                self._logger.info(
                    "no-cargo step merge: idle=%d, skip query, wait=%dmin",
                    idle_steps, merged_wait)
                self._steps_without_order[driver_id] = idle_steps + 1
                return self._make_batch_wait(merged_wait)

        # 检索当前场景的历史经验，传递给 scorer 用于校准
        current_hour = state.hour_in_day()
        current_ts = hour_to_time_slot(current_hour)
        current_rk = pos_to_region_key(state.current_lat, state.current_lng)
        exp_summary = self._experience.query_experience(
            driver_id, current_ts, current_rk, state.current_day())
        if exp_summary:
            self._scorer._current_experience = exp_summary
        else:
            self._scorer._current_experience = None

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
            if llm_action.get("action") == "take_order":
                # 检查是否需要插入休息（装货等待期利用）
                pre_rest = self._maybe_pre_rest_for_load_wait(
                    llm_action, scored, state, config)
                if pre_rest is not None:
                    return pre_rest
            return self._attach_pickup_km(llm_action, scored)

        # 阶段 3: 纯规则决策
        action = self._rule_based_decision(
            driver_id, state, config, params, scored, avg_recent)

        if action.get("action") == "take_order":
            # 检查是否需要插入休息（装货等待期利用）
            pre_rest = self._maybe_pre_rest_for_load_wait(
                action, scored, state, config)
            if pre_rest is not None:
                return pre_rest

        return self._attach_pickup_km(action, scored)

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
            # R8 fix: LLM 增强路径也要记录决策经验
            chosen_id = str(llm_result.get("params", {}).get("cargo_id", ""))
            chosen_sc = next(
                (sc for sc in scored if str(sc.cargo.get("cargo_id", "")) == chosen_id),
                best)
            self._experience.record_decision(
                driver_id, state.sim_minutes,
                state.current_lat, state.current_lng,
                float(chosen_sc.cargo.get("price", 0.0)),
                chosen_sc.pickup_km, chosen_sc.score, state.current_day())
            llm_result["_cargo_price"] = float(chosen_sc.cargo.get("price", 0.0))
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
            # 记录决策经验
            self._experience.record_decision(
                driver_id, state.sim_minutes,
                state.current_lat, state.current_lng,
                float(best.cargo.get("price", 0.0)),
                best.pickup_km, best.score, state.current_day())
            action = self._make_take_order(cargo_id)
            action["_cargo_price"] = float(best.cargo.get("price", 0.0))
            return action

        # 低分：判断等待 vs 勉强接
        wait_value = self._compute_smart_wait_value(state, config, avg_recent, params)

        # P2 fix: 首单 deadline 紧迫时强制降低等待价值，避免无止境等待
        if (config.first_order_deadline_hour is not None
                and state.today_first_order_minute is None):
            hour = state.hour_in_day()
            hours_past_deadline = hour - config.first_order_deadline_hour
            if hours_past_deadline > 0:
                # 已过 deadline：线性压低 wait_value，最多压到 -50
                decay = min(1.0, hours_past_deadline / 4.0)
                wait_value = wait_value * (1 - decay) + (-50) * decay
                self._logger.info(
                    "first-order deadline override: %dh past, wait_value=%.2f",
                    hours_past_deadline, wait_value)

        if wait_value > best.score:
            self._logger.info("wait (score=%.2f < wait_value=%.2f)", best.score, wait_value)
            self._steps_without_order[driver_id] = (
                self._steps_without_order.get(driver_id, 0) + 1)
            # R8: 记录等待决策，追踪「等了之后是否等到了更好的货」
            self._experience.record_wait_decision(
                driver_id, state.sim_minutes,
                state.current_lat, state.current_lng,
                best.score, state.current_day())
            return self._make_wait(self._smart_wait(state, params))

        # 等待价值也低，接最优的负分单
        cargo_id = str(best.cargo.get("cargo_id", ""))
        self._logger.info(
            "reluctant take: %s (score=%.2f, wait=%.2f)",
            cargo_id, best.score, wait_value)
        self._steps_without_order[driver_id] = 0
        # 记录决策经验
        self._experience.record_decision(
            driver_id, state.sim_minutes,
            state.current_lat, state.current_lng,
            float(best.cargo.get("price", 0.0)),
            best.pickup_km, best.score, state.current_day())
        action = self._make_take_order(cargo_id)
        action["_cargo_price"] = float(best.cargo.get("price", 0.0))
        return action

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
          5. 当天不是 off-day 候选（reposition 会破坏 off-day 判定）
        """
        # off-day 保护：如果当天是预规划的 off-day 或 off-day 候选，
        # 不做 reposition 以免破坏 off-day 判定
        if (config.monthly_off_days_required > 0
                and not state.today_has_repositioned
                and state.today_order_count == 0):
            current_day = state.current_day()
            off_days_done = len(state.off_days)
            if off_days_done < config.monthly_off_days_required:
                # 预规划的 off-day 当天绝对不 reposition
                if current_day in state.planned_off_days:
                    return None
                # 紧急情况：剩余天数不够时也不 reposition
                remaining_days = 31 - current_day
                off_days_needed = config.monthly_off_days_required - off_days_done
                if remaining_days <= off_days_needed + 1:
                    return None

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

        # 时间可行性校验：确保以当前速度能在合理步骤时间内完成行驶
        # 仿真引擎会校验 action_exec_cost 与实际行驶距离的一致性
        travel_minutes = best_dist / max(config.reposition_speed_kmpm, 0.01)
        if travel_minutes > 480:
            # 超长距离空驶不划算且可能触发时间不一致校验
            return None

        # 禁区校验：reposition 目标不能在禁区内
        if config.forbidden_zone:
            center = config.forbidden_zone["center"]
            radius = config.forbidden_zone["radius_km"]
            if haversine_km(best_hs[0], best_hs[1], center[0], center[1]) <= radius:
                self._logger.info(
                    "hotspot reposition blocked: target (%.4f,%.4f) inside forbidden zone",
                    best_hs[0], best_hs[1])
                return None

        # 安静窗口推演：检查 reposition 是否会跨入安静窗口
        # 注意：state.sim_minutes 是 step_start，实际 action_start = step_start + scan_cost
        # 这里加 10 分钟保守估计 scan_cost（评测用真实 scan_cost）
        if config.quiet_window:
            scan_cost_estimate = SCAN_COST_MINUTES
            action_start = state.sim_minutes + scan_cost_estimate
            action_end = action_start + int(travel_minutes)
            if self._rule_engine._overlaps_quiet_window(
                    action_start, action_end, config.quiet_window):
                self._logger.info(
                    "hotspot reposition blocked: would overlap quiet window")
                return None

        self._logger.info(
            "hotspot reposition: idle %d steps, moving %.1fkm (~%.0fmin) to (%.4f,%.4f)",
            idle_steps, best_dist, travel_minutes, best_hs[0], best_hs[1])
        return self._make_reposition(best_hs[0], best_hs[1], lat, lng)

    # ----- 子流程 7: 装货等待期休息插入 -----

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

    def _compute_merged_wait(self, state: DriverState, params: StrategyParams,
                             idle_steps: int) -> int:
        """连续无货时的大粒度合并等待，跳过查询直接休眠。

        策略：
          - 深夜 (22~5点): 直接等到早高峰，最多 480min
          - 白天非高峰: 按 idle_steps 递增，60~240min
          - 刚脱离高峰: 保守 45~90min（高峰刚结束可能恢复）
        """
        hour = state.hour_in_day()

        if hour >= 22 or hour <= 5:
            # 深夜: 等到明早高峰起始
            if hour >= 22:
                minutes_to_peak = (24 - hour + 8) * 60
            else:
                minutes_to_peak = (8 - hour) * 60
            return min(480, max(120, minutes_to_peak))

        # 白天非高峰: 递增式合并
        # idle_steps=4 → 60min, 5→90, 6→120, 7→150... 上限240
        merged = min(240, 30 + idle_steps * 30)

        # 激进模式稍微缩短
        merged = int(merged * (1 - params.aggression * 0.3))

        return max(45, merged)

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

        # R10-B: 步级等待反馈 — 最近等待成功率低时进一步衰减 wait_value
        # 如果最近 N 次等待都没等到更好的货（成功率=0），说明当前时段/区域
        # 等待无意义，应直接接单。这是经验驱动的自适应反馈，非硬编码规则。
        #
        # 长途司机识别：avg_order_exec_time >= 200min 视为长途，
        # 长途场景货源天然稀疏，连续几次"没等到更好的"是常态，
        # 需要更宽松的窗口和更温和的衰减，避免误判。
        avg_exec = state.avg_order_exec_time
        is_long_haul = avg_exec is not None and avg_exec >= 200
        wait_lookback = 5 if is_long_haul else 3
        decay_strong = 0.5 if is_long_haul else 0.3
        decay_mild = 0.75 if is_long_haul else 0.6

        wait_sr, wait_n = self._experience.get_recent_wait_success_rate(
            state.driver_id, n=wait_lookback)
        if wait_n >= wait_lookback:
            if wait_sr <= 0.0:
                base_wait_value *= decay_strong
                self._logger.debug(
                    "R10-B step-level wait feedback: 0/%d success, "
                    "long_haul=%s, wait_value *= %.2f",
                    wait_n, is_long_haul, decay_strong)
            elif wait_sr < (1.0 / wait_lookback + 0.01):
                # 只有 1 次成功：中等衰减
                base_wait_value *= decay_mild
                self._logger.debug(
                    "R10-B step-level wait feedback: %.0f%% success, "
                    "long_haul=%s, wait_value *= %.2f",
                    wait_sr * 100, is_long_haul, decay_mild)

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

    def _make_take_order(self, cargo_id: str, pickup_km: float = 0.0) -> dict[str, Any]:
        action: dict[str, Any] = {"action": "take_order", "params": {"cargo_id": cargo_id}}
        if pickup_km > 0:
            action["_pickup_km"] = pickup_km
        return action

    @staticmethod
    def _attach_pickup_km(action: dict[str, Any],
                          scored: list["ScoredCargo"]) -> dict[str, Any]:
        """为 take_order 动作附加 _pickup_km 元数据，供 pending 补偿时追踪空驶。"""
        if action.get("action") != "take_order":
            return action
        if "_pickup_km" in action:
            return action  # 已设置
        cargo_id = str(action.get("params", {}).get("cargo_id", ""))
        for sc in scored:
            if str(sc.cargo.get("cargo_id", "")) == cargo_id:
                action["_pickup_km"] = sc.pickup_km
                return action
        return action

    def _make_reposition(self, lat: float, lng: float,
                         current_lat: float = 0.0, current_lng: float = 0.0) -> dict[str, Any]:
        """生成 reposition 动作，含安全防护。

        防护逻辑：
          1. 坐标全零 → 退化为等待
          2. 已在目标附近(< 2km) → 退化为等待（避免短距离精度问题）
          3. 坐标 round(2) 对齐：确保 Agent 传出的坐标与结果文件序列化精度一致，
             避免评测脚本用 round 后坐标重算距离导致 ceil 差 1 分钟的 validation error

        注意：安静窗口推演由调用方负责（rule_engine 过滤 take_order，
              _try_hotspot_reposition 过滤主动空驶）。
              GO_HOME / FAMILY_EVENT 等必要空驶不在此处阻止。
        """
        # 对齐精度：结果文件序列化时会 round 到 2 位小数
        lat = round(lat, 2)
        lng = round(lng, 2)
        if lat == 0.0 and lng == 0.0:
            return self._make_wait(30)
        # 如果提供了当前坐标且已接近目标，不做无意义的 reposition
        # R8.4: 阈值从 2km 降到 0.5km，避免 go_home 场景中 1-2km 距离反复 wait(10)
        # 评测判定到家需要 1km 内，所以 0.5km 阈值确保已经"到家"
        if current_lat != 0.0 or current_lng != 0.0:
            dist = haversine_km(current_lat, current_lng, lat, lng)
            if dist < 0.5:
                return self._make_wait(10)
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
