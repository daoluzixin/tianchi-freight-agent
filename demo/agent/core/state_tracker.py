"""状态追踪器：在 Agent 进程内维护每位司机的累计运营状态。

不依赖 LLM 记忆，每步 O(1) 更新。首次调用时可通过 decision_history 重建历史状态。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """两点间 Haversine 大圆距离（km）。"""
    r = 6371.0
    p1, l1, p2, l2 = math.radians(lat1), math.radians(lng1), math.radians(lat2), math.radians(lng2)
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _calendar_weekday(day_idx: int) -> int:
    """2026-03 的 day_idx（0-based）对应周几。0=Mon … 6=Sun。"""
    base = date(2026, 3, 1)
    return (base + timedelta(days=day_idx)).weekday()


@dataclass
class DriverState:
    """单个司机的运营状态快照。"""
    driver_id: str
    cost_per_km: float = 1.5  # 从 DriverConfig 初始化
    sim_minutes: int = 0
    current_lat: float = 0.0
    current_lng: float = 0.0

    # 收益统计
    total_gross_income: float = 0.0
    total_distance_km: float = 0.0
    total_deadhead_km: float = 0.0  # 空驶总里程（reposition + pickup）

    # 每日统计
    today_order_count: int = 0
    today_first_order_minute: int | None = None  # 今日首笔成交的 action_start 时刻
    consecutive_no_cargo_steps: int = 0  # 连续无有效货源步数（用于判断是否空驶）

    # 休息追踪
    current_rest_streak: int = 0          # 当前连续休息分钟数
    longest_rest_today: int = 0           # 今日最长连续休息
    rest_satisfied_today: bool = False    # 今日休息是否已满足
    last_rest_end_min: int = 0            # 上次休息结束时刻（用于计算连续工作时长）

    # 月度统计
    order_days: set[int] = field(default_factory=set)     # 有成交接单的天
    off_days: set[int] = field(default_factory=set)       # 完全无活动的天（至今）
    visit_target_days: set[int] = field(default_factory=set)  # 到访目标点的天

    # 家事状态机
    family_phase: str = "idle"  # idle -> go_spouse -> wait_spouse -> go_home -> at_home -> done
    spouse_wait_accumulated: int = 0
    spouse_wait_start_min: int = 0  # 进入 wait_spouse 阶段时的仿真时刻

    # 熟货追踪
    special_cargo_taken: bool = False

    # off-day 活动追踪（D006/D008 评测要求当天 take_order + reposition 均为 0）
    today_has_repositioned: bool = False

    # 月度休息日历（首次 decide 时一次性规划）
    planned_off_days: set[int] = field(default_factory=set)  # 预规划的 off-day 日期（day index）
    rest_calendar_initialized: bool = False  # 是否已初始化休息日历

    # 订单执行时间追踪（用于 go_home guard 动态 min_order_time）
    _order_exec_total_min: float = 0.0  # 累计订单执行时间
    _order_exec_count: int = 0          # 累计完成订单数
    avg_order_exec_time: float | None = None  # 滑动平均（None = 无历史数据）

    # 通用
    step_count: int = 0
    total_tokens_used: int = 0
    _last_day: int = -1  # 用于检测跨天重置

    def current_day(self) -> int:
        return self.sim_minutes // 1440

    def hour_in_day(self) -> float:
        return (self.sim_minutes % 1440) / 60.0

    def is_weekday(self) -> bool:
        return _calendar_weekday(self.current_day()) < 5

    def net_income(self) -> float:
        return self.total_gross_income - self.total_distance_km * self.cost_per_km


class StateTracker:
    """管理所有司机的状态。"""

    def __init__(self) -> None:
        self._states: dict[str, DriverState] = {}

    def get_state(self, driver_id: str) -> DriverState:
        if driver_id not in self._states:
            self._states[driver_id] = DriverState(driver_id=driver_id)
        return self._states[driver_id]

    def init_from_status(self, driver_id: str, status: dict[str, Any],
                         skip_rollover: bool = False) -> DriverState:
        """从 get_driver_status 返回值初始化/更新状态。

        Args:
            skip_rollover: 如果为 True，跳过跨天检测。调用方需要在完成
                pending_action 补偿后手动调用 check_day_rollover()。
        """
        state = self.get_state(driver_id)
        state.sim_minutes = int(status.get("simulation_progress_minutes", 0))
        state.current_lat = float(status.get("current_lat", 0.0))
        state.current_lng = float(status.get("current_lng", 0.0))
        state.cost_per_km = float(status.get("cost_per_km", state.cost_per_km))
        if not skip_rollover:
            self._check_day_rollover(state)
        return state

    def check_day_rollover(self, state: DriverState) -> None:
        """公开的跨天检测接口，供 decide() 在 pending_action 补偿后调用。"""
        self._check_day_rollover(state)

    def rebuild_from_history(self, driver_id: str, records: list[dict[str, Any]]) -> DriverState:
        """从 query_decision_history 的全量记录重建状态。"""
        state = self.get_state(driver_id)
        for record in records:
            self._apply_record(state, record)
        return state

    def update_after_action(self, state: DriverState, action: dict[str, Any], result: dict[str, Any]) -> None:
        """每步决策执行后更新状态。"""
        state.step_count += 1

        # 更新仿真时间和位置
        if "simulation_progress_minutes" in result:
            state.sim_minutes = int(result["simulation_progress_minutes"])

        action_name = str(action.get("action", "")).lower()
        params = action.get("params", {}) or {}

        if action_name == "wait":
            duration = int(params.get("duration_minutes", 0))
            state.current_rest_streak += duration
            state.longest_rest_today = max(state.longest_rest_today, state.current_rest_streak)
            # 更新上次休息结束时刻（用于连续工作安全保护）
            state.last_rest_end_min = state.sim_minutes
        elif action_name == "take_order" and result.get("accepted") is False:
            # 明确标记为失败的 take_order（货物已失效等）不打断休息连续性
            # 注意：用 `is False` 而非 `not result.get(...)` 以区分缺失字段 vs 明确失败
            pass
        else:
            # 非 wait 动作打断休息连续性
            state.current_rest_streak = 0

        if action_name == "reposition":
            state.current_lat = float(params.get("latitude", state.current_lat))
            state.current_lng = float(params.get("longitude", state.current_lng))
            distance = float(result.get("distance_km", 0.0))
            state.total_distance_km += distance
            state.total_deadhead_km += distance
            state.today_has_repositioned = True

        elif action_name == "take_order":
            accepted = bool(result.get("accepted", False))
            if accepted:
                cargo_id = str(params.get("cargo_id", ""))
                # 更新位置到卸货点
                state.current_lat = float(result.get("current_lat", state.current_lat))
                state.current_lng = float(result.get("current_lng", state.current_lng))

                pickup_km = float(result.get("pickup_deadhead_km", 0.0))
                haul_km = float(result.get("haul_distance_km", 0.0))
                state.total_distance_km += pickup_km + haul_km
                state.total_deadhead_km += pickup_km

                # 更新毛收入：从 result 中获取实际收入
                income = float(result.get("income", 0.0))
                if income > 0:
                    state.total_gross_income += income
                else:
                    # fallback: 从 result 的 price 字段获取
                    price = float(result.get("price", 0.0))
                    if price > 0:
                        state.total_gross_income += price

                # 接单统计
                state.today_order_count += 1
                day = state.current_day()
                state.order_days.add(day)
                if state.today_first_order_minute is None:
                    state.today_first_order_minute = state.sim_minutes

                # 订单执行时间追踪（用于 go_home guard 动态 min_order_time）
                exec_time = float(result.get("action_exec_cost_minutes",
                                             result.get("round_cost_min", 0)))
                if exec_time > 0:
                    state._order_exec_total_min += exec_time
                    state._order_exec_count += 1
                    state.avg_order_exec_time = (
                        state._order_exec_total_min / state._order_exec_count)

                # 特殊货源追踪（从 DriverConfig 获取目标 cargo_id）
                from agent.config.driver_config import get_config
                cfg = get_config(state.driver_id)
                if cfg.special_cargo and cargo_id == cfg.special_cargo.cargo_id:
                    state.special_cargo_taken = True

        # 检查到访目标点
        self._check_visit_target(state)
        # 检查跨天重置
        self._check_day_rollover(state)

    def _check_day_rollover(self, state: DriverState) -> None:
        """检测日期变化，重置每日统计。"""
        import logging
        _log = logging.getLogger("agent.state_tracker")
        current_day = state.current_day()
        if current_day != state._last_day:
            if state._last_day == -1:
                # 首次初始化：仅设置 _last_day，不重置计数器
                state._last_day = current_day
                return
            # 真正的跨天：检查前一天是否为 off day
            # 区分两种口径：
            #   - 口径 A（D002/D007）：当天无成交接单即算 off-day
            #   - 口径 B（D006/D008）：当天 take_order + reposition 活动分钟均为 0
            # 统一使用更严格的口径 B：同时要求无接单 + 无空驶
            # 这不会影响口径 A 的司机（因为 off-day 当天本身就不会做 reposition）
            is_off = (state._last_day not in state.order_days
                      and not state.today_has_repositioned)
            _log.info(
                "DAY ROLLOVER: %s day %d->%d, order_days=%s, repos=%s, is_off=%s, off_days=%s",
                state.driver_id, state._last_day, current_day,
                state.order_days, state.today_has_repositioned, is_off, state.off_days)
            if is_off:
                state.off_days.add(state._last_day)
            state._last_day = current_day
            state.today_order_count = 0
            state.today_first_order_minute = None
            state.today_has_repositioned = False
            # 跨天时 longest_rest_today 重置为 0，而非延续 current_rest_streak。
            # 原因：评测系统按自然日边界切割 wait interval，跨天延续的 streak
            # 不会被完整计入新一天。如果设为 streak 值，会导致 schedule_planner
            # 误认为新一天已满足休息需求而跳过 rest。
            state.longest_rest_today = 0
            state.rest_satisfied_today = False

    def _check_visit_target(self, state: DriverState) -> None:
        """检查是否到访了目标点（从动态 config 读取坐标）。"""
        from agent.config.driver_config import get_config
        config = get_config(state.driver_id)
        if config.visit_target:
            if haversine_km(state.current_lat, state.current_lng,
                            config.visit_target[0], config.visit_target[1]) <= 1.0:
                state.visit_target_days.add(state.current_day())

    def _apply_record(self, state: DriverState, record: dict[str, Any]) -> None:
        """从历史记录重建单步状态。"""
        action_obj = record.get("action", {})
        result = record.get("result", {})
        pos_after = record.get("position_after", {})

        state.sim_minutes = int(result.get("simulation_progress_minutes", state.sim_minutes))
        if pos_after:
            state.current_lat = float(pos_after.get("lat", state.current_lat))
            state.current_lng = float(pos_after.get("lng", state.current_lng))

        action_name = str(action_obj.get("action", "")).lower()
        params = action_obj.get("params", {}) or {}

        if action_name == "wait":
            duration = int(params.get("duration_minutes", 0))
            state.current_rest_streak += duration
            state.longest_rest_today = max(state.longest_rest_today, state.current_rest_streak)
        else:
            state.current_rest_streak = 0

        if action_name == "reposition":
            dist = float(result.get("distance_km", 0.0))
            state.total_distance_km += dist
            state.total_deadhead_km += dist
            state.today_has_repositioned = True

        elif action_name == "take_order" and bool(result.get("accepted", False)):
            pickup_km = float(result.get("pickup_deadhead_km", 0.0))
            haul_km = float(result.get("haul_distance_km", 0.0))
            state.total_distance_km += pickup_km + haul_km
            state.total_deadhead_km += pickup_km
            # 重建收入
            income = float(result.get("income", 0.0))
            if income > 0:
                state.total_gross_income += income
            else:
                price = float(result.get("price", 0.0))
                if price > 0:
                    state.total_gross_income += price
            state.today_order_count += 1
            state.order_days.add(state.current_day())
            if state.today_first_order_minute is None:
                state.today_first_order_minute = state.sim_minutes
            cargo_id = str(params.get("cargo_id", ""))
            from agent.config.driver_config import get_config
            cfg = get_config(state.driver_id)
            if cfg.special_cargo and cargo_id == cfg.special_cargo.cargo_id:
                state.special_cargo_taken = True

        state.step_count += 1
        self._check_visit_target(state)
        self._check_day_rollover(state)
