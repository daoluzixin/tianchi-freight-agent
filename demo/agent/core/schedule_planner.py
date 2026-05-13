"""时间规划器：决定当前时段应该做什么（work / rest / go_home / family_event）。

基于 DriverConfig 的休息约束、安静窗口和特殊事件配置，输出高层调度指令，
由 RuleEngine 和 CargoScorer 具体执行。

v5 改进：
  - 增加连续驾驶安全保护：无论是否配置了每日休息，连续工作超阈值后强制休息
  - 适用于长途司机场景（无 min_continuous_rest_minutes 配置但仍需安全保护）
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from agent.config.driver_config import DriverConfig, QuietWindow, FamilyEvent, get_config
from agent.core.state_tracker import DriverState, haversine_km


class ScheduleAction(str, Enum):
    """调度层建议动作类型。"""
    WORK = "work"                   # 正常寻货/接单
    REST = "rest"                   # 休息
    GO_HOME = "go_home"            # 回家
    FAMILY_EVENT = "family_event"  # 执行家事状态机
    OFF_DAY = "off_day"            # 全天休息
    REPOSITION = "reposition"      # 主动空驶到目标位置


class ScheduleDecision:
    """规划器输出的调度决策。"""

    def __init__(self, action: ScheduleAction, reason: str = "",
                 target_pos: tuple[float, float] | None = None,
                 wait_minutes: int = 30,
                 priority: int = 0):
        self.action = action
        self.reason = reason
        self.target_pos = target_pos  # 用于 GO_HOME / FAMILY_EVENT / REPOSITION
        self.wait_minutes = wait_minutes  # REST 时建议等待时长
        self.priority = priority  # 越高越强制

    def __repr__(self) -> str:
        return f"ScheduleDecision({self.action.value}, reason={self.reason!r})"


class SchedulePlanner:
    """基于规则的时间段规划器。"""

    # 连续工作安全阈值（分钟）：超过此时长无休息则强制插入休息
    _MAX_CONTINUOUS_WORK_MINUTES = 720  # 12小时
    # 强制安全休息时长（分钟）
    _SAFETY_REST_MINUTES = 60

    def plan(self, state: DriverState, config: DriverConfig) -> ScheduleDecision:
        """根据当前状态和配置，返回本步应执行的调度决策。"""

        # 0) 安全保护：连续工作时间过长强制休息（适用于所有司机）
        safety_rest = self._check_continuous_work_safety(state, config)
        if safety_rest:
            return safety_rest

        # 1) 家事状态机优先级最高
        if config.family_event:
            family_decision = self._handle_family_event(state, config)
            if family_decision:
                return family_decision

        # 2) 回家约束：必须在安静窗口检查之前，
        #    否则远离家时会直接 REST 而不是先回家
        if config.must_return_home and config.home_pos:
            home_decision = self._handle_go_home(state, config)
            if home_decision:
                return home_decision

        # 3) 安静窗口 → 强制 REST
        #    到这里说明：已在家（回家约束已满足）或无回家约束。
        #    额外保护：如果有回家约束但不在家附近，不应 REST 而应继续赶路。
        if config.quiet_window and config.quiet_window.is_active(state.sim_minutes):
            # 有回家约束且不在家附近 → 跳过安静窗口，让后续逻辑处理
            if config.must_return_home and config.home_pos:
                dist_home = haversine_km(state.current_lat, state.current_lng,
                                         config.home_pos[0], config.home_pos[1])
                if dist_home > 2.0:
                    # 不在家，不应休息，应继续赶路（由 _handle_go_home 已处理，
                    # 但作为安全网：直接返回 GO_HOME）
                    return ScheduleDecision(
                        action=ScheduleAction.GO_HOME,
                        reason="安静窗口中但未到家，继续赶路",
                        target_pos=config.home_pos,
                        priority=90,
                    )
            remaining = config.quiet_window.minutes_until_end(state.sim_minutes)
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason=f"安静窗口中，剩余{remaining}分钟",
                wait_minutes=min(remaining, 60),
                priority=90,
            )

        # 4) 每日休息未满足 → 安排休息（在建议休息时段内）
        if config.min_continuous_rest_minutes > 0:
            rest_decision = self._handle_daily_rest(state, config)
            if rest_decision:
                return rest_decision

        # 5) Off-day 规划：月底需保证 off_days 数量
        if config.monthly_off_days_required > 0:
            off_decision = self._handle_off_day(state, config)
            if off_decision:
                return off_decision

        # 6) 到访目标点
        if config.visit_target and len(state.visit_target_days) < config.visit_days_required:
            visit_decision = self._handle_visit_target(state, config)
            if visit_decision:
                return visit_decision

        # 默认：正常工作
        return ScheduleDecision(action=ScheduleAction.WORK, reason="正常工作时段")

    def _check_continuous_work_safety(self, state: DriverState,
                                      config: DriverConfig) -> ScheduleDecision | None:
        """安全保护：连续工作超过阈值强制休息。

        适用于所有司机（含长途无休息配置的司机如 D003）。
        只有当配置中没有其他休息机制（min_continuous_rest_minutes == 0
        且无安静窗口）时才会生效，避免与其他休息逻辑冲突。
        """
        # 如果已有专门的休息机制覆盖，则不需要安全保护
        if config.min_continuous_rest_minutes > 0:
            return None
        if config.quiet_window and config.quiet_window.is_active(state.sim_minutes):
            return None

        # 计算自上次休息以来的连续工作时间
        continuous_work = state.sim_minutes - state.last_rest_end_min
        if continuous_work >= self._MAX_CONTINUOUS_WORK_MINUTES:
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason=f"安全保护：连续工作{continuous_work}min超过{self._MAX_CONTINUOUS_WORK_MINUTES}min阈值",
                wait_minutes=self._SAFETY_REST_MINUTES,
                priority=80,
            )

        return None

    def _handle_family_event(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """家事状态机处理。

        使用 FamilyEvent.waypoints 列表支持多途经点：
          - waypoints[0] 为第一个途经点（如接配偶）
          - waypoints[-1] 或 home_pos 为最终目的地
        如果 waypoints 为空，则直接用 home_pos 作为目标。
        """
        fe = config.family_event
        if not fe:
            return None

        # 家事已结束
        if state.family_phase == "done" or state.sim_minutes >= fe.end_min:
            if state.family_phase != "done":
                state.family_phase = "done"
            return None

        # 获取途经点信息
        waypoints = fe.waypoints or []
        first_waypoint = waypoints[0] if waypoints else None
        first_wp_pos = (
            (float(first_waypoint["lat"]), float(first_waypoint["lng"]))
            if first_waypoint else fe.home_pos
        )
        first_wp_wait = int(first_waypoint.get("wait_minutes", 10)) if first_waypoint else 0

        # 家事尚未触发
        if state.sim_minutes < fe.trigger_min:
            # 提前 60 分钟准备去第一个途经点
            if state.sim_minutes >= fe.trigger_min - 60 and state.family_phase == "idle":
                dist_to_wp = haversine_km(
                    state.current_lat, state.current_lng,
                    first_wp_pos[0], first_wp_pos[1])
                travel_min = dist_to_wp / config.reposition_speed_kmpm
                if travel_min >= 30:
                    state.family_phase = "go_spouse"
                    return ScheduleDecision(
                        action=ScheduleAction.REPOSITION,
                        reason="提前移动到途经点",
                        target_pos=first_wp_pos,
                        priority=85,
                    )
            return None

        # 家事期间状态机
        if state.family_phase == "idle":
            state.family_phase = "go_spouse"
            return ScheduleDecision(
                action=ScheduleAction.REPOSITION,
                reason="家事触发，前往第一个途经点",
                target_pos=first_wp_pos,
                priority=95,
            )

        if state.family_phase == "go_spouse":
            dist = haversine_km(state.current_lat, state.current_lng,
                                first_wp_pos[0], first_wp_pos[1])
            if dist <= 1.0:
                state.family_phase = "wait_spouse"
                state.spouse_wait_start_min = state.sim_minutes
                state.spouse_wait_accumulated = 0
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason="到达途经点，等待",
                    wait_minutes=first_wp_wait,
                    priority=95,
                )
            return ScheduleDecision(
                action=ScheduleAction.REPOSITION,
                reason="继续前往途经点",
                target_pos=first_wp_pos,
                priority=95,
            )

        if state.family_phase == "wait_spouse":
            # 用仿真时间差计算实际等待时长，而非规划器自行累加
            actual_waited = state.sim_minutes - state.spouse_wait_start_min
            if actual_waited >= first_wp_wait:
                state.family_phase = "go_home"
                return ScheduleDecision(
                    action=ScheduleAction.REPOSITION,
                    reason="途经等待完成，前往家",
                    target_pos=fe.home_pos,
                    priority=95,
                )
            remaining_wait = first_wp_wait - actual_waited
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason=f"途经等待中，还需{remaining_wait}分钟",
                wait_minutes=min(remaining_wait, 15),
                priority=95,
            )

        if state.family_phase == "go_home":
            dist = haversine_km(state.current_lat, state.current_lng,
                                fe.home_pos[0], fe.home_pos[1])
            if dist <= 1.0:
                state.family_phase = "at_home"
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason="已到家，家事期间在家休息",
                    wait_minutes=60,
                    priority=90,
                )
            return ScheduleDecision(
                action=ScheduleAction.REPOSITION,
                reason="前往家中",
                target_pos=fe.home_pos,
                priority=95,
            )

        if state.family_phase == "at_home":
            # 家事期间如果结束时间到了则标记完成
            if state.sim_minutes >= fe.end_min:
                state.family_phase = "done"
                return None
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason="家事期间在家休息",
                wait_minutes=min(60, fe.end_min - state.sim_minutes),
                priority=90,
            )

        return None

    def _handle_go_home(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """回家逻辑：每天 home_deadline_hour 前到家。"""
        hour = state.hour_in_day()
        if not config.home_pos:
            return None

        dist_home = haversine_km(state.current_lat, state.current_lng,
                                 config.home_pos[0], config.home_pos[1])
        travel_min = dist_home / config.reposition_speed_kmpm

        # 如果已经在安静时段（home_quiet_start ~ home_quiet_end）
        if hour >= config.home_quiet_start or hour < config.home_quiet_end:
            if dist_home <= 2.0:
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason="已在家，夜间休息",
                    wait_minutes=60,
                    priority=85,
                )
            else:
                return ScheduleDecision(
                    action=ScheduleAction.GO_HOME,
                    reason="夜间必须在家",
                    target_pos=config.home_pos,
                    priority=90,
                )

        # 判断是否需要提前出发
        deadline_minutes_in_day = config.home_deadline_hour * 60
        current_minutes_in_day = state.sim_minutes % 1440
        time_to_deadline = deadline_minutes_in_day - current_minutes_in_day

        if time_to_deadline > 0 and travel_min >= time_to_deadline - 30:
            # 需要立即出发才能赶上
            return ScheduleDecision(
                action=ScheduleAction.GO_HOME,
                reason=f"需提前出发回家，路程约{travel_min:.0f}分钟",
                target_pos=config.home_pos,
                priority=80,
            )

        return None

    def _handle_daily_rest(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """每日连续休息约束。

        优化：一次性输出剩余全部休息时长（最大120分钟），减少碎片化步骤。
        """
        # 仅平日要求
        if config.rest_weekday_only and not state.is_weekday():
            return None

        # 已满足今日休息要求
        if state.longest_rest_today >= config.min_continuous_rest_minutes:
            state.rest_satisfied_today = True
            return None

        # 在建议休息时段内且休息尚未满足 → 强制休息
        hour = state.hour_in_day()
        in_rest_window = False
        if config.suggested_rest_start_hour > config.suggested_rest_end_hour:
            # 跨天窗口
            in_rest_window = hour >= config.suggested_rest_start_hour or hour < config.suggested_rest_end_hour
        else:
            in_rest_window = config.suggested_rest_start_hour <= hour < config.suggested_rest_end_hour

        if in_rest_window:
            remaining = config.min_continuous_rest_minutes - state.current_rest_streak
            # 一次性输出尽可能多的休息时长，减少步骤数
            wait = min(remaining, 120)
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason=f"每日休息未满足（已休{state.current_rest_streak}min/{config.min_continuous_rest_minutes}min）",
                wait_minutes=wait,
                priority=70,
            )

        # 不在休息窗口但今日仍未满足：如果快到休息窗口了，提前准备
        # 这里不强制，让 WORK 继续
        return None

    def _handle_off_day(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """月度 off-day 规划。

        优化策略：优先在货源低谷日（周末）安排 off-day，而非月底集中休息。
        月底是冲刺期，休息会损失最多收入。
        """
        current_day = state.current_day()
        total_days = 31
        remaining_days = total_days - current_day
        off_days_done = len(state.off_days)
        off_days_needed = config.monthly_off_days_required - off_days_done

        if off_days_needed <= 0:
            return None

        # 今天已经接过单了，不能作为 off-day
        if state.today_order_count > 0:
            return None

        # 策略1：周末优先（周六=5, 周日=6 是货源低谷日）
        from agent.core.state_tracker import _calendar_weekday
        weekday = _calendar_weekday(current_day)
        is_weekend = weekday >= 5

        if is_weekend:
            # 周末且还需要 off-day → 安排休息（一次性等待一整天减少步骤）
            return ScheduleDecision(
                action=ScheduleAction.OFF_DAY,
                reason=f"周末低谷日安排 off-day（还需{off_days_needed}天）",
                wait_minutes=480,
                priority=60,
            )

        # 策略2：紧急兜底 - 剩余天数不够时强制安排
        if remaining_days <= off_days_needed + 1:
            return ScheduleDecision(
                action=ScheduleAction.OFF_DAY,
                reason=f"紧急：仅剩{remaining_days}天，还需{off_days_needed}天 off-day",
                wait_minutes=480,
                priority=75,
            )

        # 策略3：均匀分布 - 计算理想间隔，到了该休息的时候就休息
        if off_days_needed > 0 and remaining_days > off_days_needed:
            ideal_interval = remaining_days // (off_days_needed + 1)
            # 找到上一个 off-day 的天数
            last_off = max(state.off_days) if state.off_days else -1
            days_since_last_off = current_day - last_off
            if days_since_last_off >= ideal_interval:
                return ScheduleDecision(
                    action=ScheduleAction.OFF_DAY,
                    reason=f"均匀分布 off-day（间隔{days_since_last_off}天，理想{ideal_interval}天）",
                    wait_minutes=480,
                    priority=55,
                )

        return None

    def _handle_visit_target(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """到访目标点规划。"""
        if not config.visit_target:
            return None

        days_visited = len(state.visit_target_days)
        days_remaining = 31 - state.current_day()
        visits_needed = config.visit_days_required - days_visited

        if visits_needed <= 0:
            return None

        # 如果今天已经访问过了，不需要再去
        if state.current_day() in state.visit_target_days:
            return None

        # 如果剩余天数紧张，需要优先安排
        dist = haversine_km(state.current_lat, state.current_lng,
                            config.visit_target[0], config.visit_target[1])

        # 距离近且需要打卡 → 顺路去一下
        if dist <= 5.0:
            return ScheduleDecision(
                action=ScheduleAction.REPOSITION,
                reason=f"到访目标点打卡（距{dist:.1f}km），还需{visits_needed}天",
                target_pos=config.visit_target,
                priority=50,
            )

        # 紧急：剩余天数 ≤ 需要次数 + 2
        if days_remaining <= visits_needed + 2:
            return ScheduleDecision(
                action=ScheduleAction.REPOSITION,
                reason=f"紧急到访目标点（还需{visits_needed}天，仅剩{days_remaining}天）",
                target_pos=config.visit_target,
                priority=75,
            )

        return None
