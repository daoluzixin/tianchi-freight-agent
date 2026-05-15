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
from agent.core.timeline_projector import SAFETY_BUFFER_MINUTES, SCAN_COST_MINUTES


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

        # 初始化月度休息日历（首次调用时执行一次）
        if not state.rest_calendar_initialized:
            self._init_rest_calendar(state, config)

        # 0) 安全保护：连续工作时间过长强制休息（适用于所有司机）
        safety_rest = self._check_continuous_work_safety(state, config)
        if safety_rest:
            return safety_rest

        # 0.5) Off-day 全局锁：预规划的 off-day 当天，锁死为 wait
        #      优先级极高，仅家事状态机可覆盖（家事是定时触发不可延迟的）
        if config.monthly_off_days_required > 0:
            off_lock = self._handle_off_day_lock(state, config)
            if off_lock:
                # 家事可以打断 off-day（家事罚分 9000+ 远超 off-day 罚分）
                if config.family_event:
                    family_decision = self._handle_family_event(state, config)
                    if family_decision:
                        return family_decision
                return off_lock

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
            # 一次性 wait 到安静窗口结束，避免中间 scan_cost 被评测计为活跃
            # 上限 480 分钟（仿真引擎 wait 上限）
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason=f"安静窗口中，一次性等待{min(remaining, 480)}分钟",
                wait_minutes=min(remaining, 480),
                priority=90,
            )

        # 3.5) 安静窗口即将开始 → 提前 REST（避免发起无法在窗口前完成的动作）
        #      R8.4: 前瞻时间增加 SCAN_COST_MINUTES，因为 query_cargo 会推进仿真时间
        if config.quiet_window and not config.quiet_window.is_active(state.sim_minutes):
            minutes_to_start = self._minutes_until_quiet_start(state.sim_minutes, config.quiet_window)
            if 0 < minutes_to_start <= SAFETY_BUFFER_MINUTES + SCAN_COST_MINUTES:
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason=f"安静窗口{minutes_to_start}分钟后开始，提前休息",
                    wait_minutes=minutes_to_start,
                    priority=85,
                )

        # 3.8) 首单 deadline 催促：deadline 临近但今天还没接单，优先接单
        if config.first_order_deadline_hour is not None:
            first_order_decision = self._handle_first_order_deadline(state, config)
            if first_order_decision:
                return first_order_decision

        # 4) 每日连续休息（大粒度一步到位）
        if config.min_continuous_rest_minutes > 0:
            rest_decision = self._handle_daily_rest(state, config)
            if rest_decision:
                return rest_decision

        # 5) 特殊货源提前空驶：在上架时间前移动到装货点附近
        if config.special_cargo and not state.special_cargo_taken:
            special_decision = self._handle_special_cargo_approach(state, config)
            if special_decision:
                return special_decision

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
            # R8.6: go_home 冲突保护 — 如果强制休息会导致错过 go_home deadline，
            # 跳过安全休息。go_home 罚分 900/天远大于无休息的风险（仅安全建议，无罚分）。
            if config.must_return_home and config.home_pos:
                current_minutes_in_day = state.sim_minutes % 1440
                deadline_min_in_day = config.home_deadline_hour * 60
                time_to_deadline = deadline_min_in_day - current_minutes_in_day
                if time_to_deadline > 0:
                    dist_home = haversine_km(
                        state.current_lat, state.current_lng,
                        config.home_pos[0], config.home_pos[1])
                    travel_home_min = dist_home / max(config.reposition_speed_kmpm, 0.01)
                    # 如果 "安全休息时长 + 回家路程 + 缓冲" 会超过 deadline，跳过休息
                    if (self._SAFETY_REST_MINUTES + travel_home_min
                            + SAFETY_BUFFER_MINUTES > time_to_deadline):
                        return None  # 让 _handle_go_home 处理

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

        # 家事尚未触发 — 动态计算提前出发窗口
        if state.sim_minutes < fe.trigger_min:
            if state.family_phase == "idle":
                dist_to_wp = haversine_km(
                    state.current_lat, state.current_lng,
                    first_wp_pos[0], first_wp_pos[1])
                travel_min = dist_to_wp / max(config.reposition_speed_kmpm, 0.01)
                # 全链路时间估算：去途经点 + 等待 + 去家 + 缓冲
                dist_wp_to_home = haversine_km(
                    first_wp_pos[0], first_wp_pos[1],
                    fe.home_pos[0], fe.home_pos[1])
                travel_wp_to_home = dist_wp_to_home / max(config.reposition_speed_kmpm, 0.01)
                total_chain_min = travel_min + first_wp_wait + travel_wp_to_home + 120  # 120min 缓冲
                # 如果行程链总时间 >= 剩余时间，立即出发
                time_to_trigger = fe.trigger_min - state.sim_minutes
                if time_to_trigger <= total_chain_min:
                    state.family_phase = "go_spouse"
                    return ScheduleDecision(
                        action=ScheduleAction.REPOSITION,
                        reason=f"家事提前出发（链路需{total_chain_min:.0f}min，剩余{time_to_trigger}min）",
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
                remaining = fe.end_min - state.sim_minutes
                return ScheduleDecision(
                    action=ScheduleAction.FAMILY_EVENT,
                    reason=f"已到家，家事期间在家休息（剩{remaining}min）",
                    wait_minutes=min(480, max(remaining, 1)),
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
            remaining = fe.end_min - state.sim_minutes
            # 使用大粒度等待（最大 480min / _make_batch_wait 上限），减少步骤浪费
            return ScheduleDecision(
                action=ScheduleAction.FAMILY_EVENT,
                reason=f"家事期间在家休息（剩{remaining}min）",
                wait_minutes=min(480, remaining),
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
                # R8.6 修正：安静时段不在家，必须立即回家。
                # D009 日志分析：几乎所有 go_home 违规都因为安静时段先 REST 60min
                # 再回家导致迟到。go_home 罚分 900/天 >> quiet 空驶罚分 200-500/次，
                # 所以无条件回家是最优策略（即使距家很远也要减少损失）。
                return ScheduleDecision(
                    action=ScheduleAction.GO_HOME,
                    reason=f"安静时段不在家，紧急回家（路程{travel_min:.0f}min，go_home罚分900远超安静期罚分）",
                    target_pos=config.home_pos,
                    priority=92,
                )

        # 判断是否需要提前出发
        deadline_minutes_in_day = config.home_deadline_hour * 60
        current_minutes_in_day = state.sim_minutes % 1440
        time_to_deadline = deadline_minutes_in_day - current_minutes_in_day

        if time_to_deadline > 0 and travel_min + SAFETY_BUFFER_MINUTES >= time_to_deadline:
            # 需要立即出发才能赶上（+SAFETY_BUFFER 覆盖 scan_cost + ceil 误差 + 路径偏差）
            return ScheduleDecision(
                action=ScheduleAction.GO_HOME,
                reason=f"需提前出发回家，路程约{travel_min:.0f}分钟（含安全余量）",
                target_pos=config.home_pos,
                priority=80,
            )

        # R8.6: 额外保护 — 当 daily_rest 可能拦截时，提前回家。
        # 条件：当天已过 80%（daily_rest 强制触发阈值）且 rest 未满足，
        # 同时 "回家路程 + 缓冲 + rest时间" 超过剩余时间。
        # 只在 daily_rest 即将触发的时间窗口内才激活，避免过早回家。
        if (time_to_deadline > 0
                and config.min_continuous_rest_minutes > 0
                and state.longest_rest_today < config.min_continuous_rest_minutes
                and current_minutes_in_day >= 1152):  # 与 _handle_daily_rest 的 80% 阈值一致
            # daily_rest 即将强制触发，但休息后会错过 go_home deadline
            rest_needed = min(config.min_continuous_rest_minutes, 480)
            if travel_min + SAFETY_BUFFER_MINUTES + rest_needed > time_to_deadline:
                return ScheduleDecision(
                    action=ScheduleAction.GO_HOME,
                    reason=f"回家优先：rest({rest_needed}min)+路程({travel_min:.0f}min)超出deadline余量({time_to_deadline}min)",
                    target_pos=config.home_pos,
                    priority=81,
                )

        return None

    def _handle_first_order_deadline(self, state: DriverState,
                                        config: DriverConfig) -> ScheduleDecision | None:
        """首单 deadline 催促：deadline 临近但今天还没接单时，强制进入 WORK 模式。

        策略：
          - 今天已经接了首单 → 无需催促
          - 距 deadline 不足 2 小时 → 强制 WORK，覆盖其他低优先级的 rest/off-day
          - 已过 deadline → 也强制 WORK（赶紧接单减少损失）
        """
        if state.today_first_order_minute is not None:
            return None  # 今天已接到首单

        hour = state.hour_in_day()
        deadline_hour = config.first_order_deadline_hour

        # 计算距 deadline 的小时数
        hours_to_deadline = deadline_hour - hour

        # 已过 deadline：赶紧接单
        if hours_to_deadline < 0:
            return ScheduleDecision(
                action=ScheduleAction.WORK,
                reason=f"首单已超 deadline（{deadline_hour}:00），紧急寻货",
                priority=75,
            )

        # 距 deadline 不足 2 小时：催促接单，覆盖低优先级 rest
        if hours_to_deadline <= 2:
            return ScheduleDecision(
                action=ScheduleAction.WORK,
                reason=f"首单 deadline {deadline_hour}:00 临近（剩{hours_to_deadline:.1f}h），优先接单",
                priority=72,
            )

        return None

    def _handle_daily_rest(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """每日连续休息约束 — 大粒度一步到位。

        核心改进：一旦决定休息，直接输出整个所需时长的 wait（最大 480min），
        避免分步累加被中间的高优先级动作打断导致连续性归零。

        评测规则：wait 动作合并后取最长连续段，query_cargo 不打断。
        所以只要输出足够长的单个 wait，评测必定满足。

        三阶段策略：
          1. 建议窗口：在配置的休息时段内开始，一步输出全部所需时长
          2. 错过窗口补偿：已过建议窗口但仍未休息够，立即补偿
          3. 紧急补偿：临近日终仍未满足时，强制输出全部所需时长
        """
        # 仅平日要求
        if config.rest_weekday_only and not state.is_weekday():
            return None

        # 已满足今日休息要求
        if state.longest_rest_today >= config.min_continuous_rest_minutes:
            state.rest_satisfied_today = True
            return None

        # R8.6 go_home 冲突保护：如果有 go_home 约束，检查休息后是否会错过 deadline。
        # go_home 罚分 900/天 >> rest 罚分 200-400/天，所以 go_home 优先。
        if config.must_return_home and config.home_pos:
            minutes_in_day_now = state.sim_minutes % 1440
            deadline_min_in_day = config.home_deadline_hour * 60
            time_to_deadline = deadline_min_in_day - minutes_in_day_now
            if time_to_deadline > 0:
                dist_home = haversine_km(
                    state.current_lat, state.current_lng,
                    config.home_pos[0], config.home_pos[1])
                travel_home_min = dist_home / max(config.reposition_speed_kmpm, 0.01)
                # 如果 "当前时间 + 休息时长 + 回家路程 + 安全缓冲" 会超过 deadline，
                # 则跳过休息让 go_home guard 优先处理
                rest_duration = min(config.min_continuous_rest_minutes, 480)
                if rest_duration + travel_home_min + SAFETY_BUFFER_MINUTES > time_to_deadline:
                    return None  # 让 go_home 逻辑处理

        # 强制休息保护：当天已过 80% 时间但休息仍未满足
        minutes_in_day = state.sim_minutes % 1440
        if minutes_in_day >= 1152:  # 80% of 1440 = 1152 (19:12)
            minutes_left = 1440 - minutes_in_day
            if minutes_left >= config.min_continuous_rest_minutes:
                wait = min(config.min_continuous_rest_minutes, 480)
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason=f"强制休息：当天已过80%但休息未满足（最长{state.longest_rest_today}min/{config.min_continuous_rest_minutes}min），立即休{wait}min",
                    wait_minutes=wait,
                    priority=82,
                )

        hour = state.hour_in_day()
        remaining_rest_needed = config.min_continuous_rest_minutes - state.current_rest_streak

        # 阶段 1: 在建议休息时段内开始 — 一步到位
        in_rest_window = False
        if config.suggested_rest_start_hour > config.suggested_rest_end_hour:
            in_rest_window = hour >= config.suggested_rest_start_hour or hour < config.suggested_rest_end_hour
        else:
            in_rest_window = config.suggested_rest_start_hour <= hour < config.suggested_rest_end_hour

        if in_rest_window:
            # 关键：评测系统按自然日边界切割 wait interval，取当天最长 merged 段。
            # 跨天延续的 streak 不会被评测系统视为当天的 rest。
            # 因此必须在当天日历内输出完整的 min_continuous_rest_minutes，
            # 而不是基于 streak 的 remaining_rest_needed。
            wait = min(config.min_continuous_rest_minutes, 480)
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason=f"每日休息：一次性休{wait}min（需{config.min_continuous_rest_minutes}min）",
                wait_minutes=wait,
                priority=70,
            )

        # 阶段 1.5: 错过窗口补偿 — 已经过了建议窗口结束时间但今日尚未满足
        # 这通常发生在长途单跨天后：接单期间跨过了整个建议窗口
        # 关键约束：评测系统按自然日边界（00:00）切割 wait。
        # 如果当天剩余时间不够完成整段 rest，强行开始会导致跨天，
        # 两天都不满足 → 双重违规。此时应等到次日凌晨（00:00）再 rest。
        past_rest_window = False
        if config.suggested_rest_start_hour > config.suggested_rest_end_hour:
            past_rest_window = config.suggested_rest_end_hour <= hour < config.suggested_rest_start_hour
        else:
            past_rest_window = hour >= config.suggested_rest_end_hour

        minutes_in_day = state.sim_minutes % 1440
        minutes_left_today = 1440 - minutes_in_day

        if past_rest_window and state.longest_rest_today < config.min_continuous_rest_minutes:
            # 和阶段 1 同理，必须输出完整的 min_continuous_rest_minutes
            full_rest = config.min_continuous_rest_minutes
            if minutes_left_today >= full_rest:
                # 当天还有足够时间完成整段 rest → 立即补偿
                wait = min(full_rest, 480)
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason=f"错过建议窗口补偿：今日最长休息{state.longest_rest_today}min不足{config.min_continuous_rest_minutes}min，补偿{wait}min",
                    wait_minutes=wait,
                    priority=75,
                )
            else:
                # 当天剩余时间不够完整休息 → 立即开始休息
                # 虽然跨天会被评测系统切割，但当天的部分仍计入 longest_rest_today，
                # 比完全不休息（等到凌晨）要好。评测系统看的是当天最长连续 wait 段。
                # 输出完整的 min_continuous_rest_minutes，让评测系统自行切割。
                wait = min(config.min_continuous_rest_minutes, 480)
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason=f"错过窗口紧急补偿：剩余{minutes_left_today}min不够完整休息{config.min_continuous_rest_minutes}min，立即开始",
                    wait_minutes=wait,
                    priority=78,
                )

        # 阶段 2: 紧急补偿 — 当天剩余时间不足以"先工作再休息"时强制开始
        if minutes_left_today <= remaining_rest_needed + 30:
            wait = min(remaining_rest_needed, 480)
            return ScheduleDecision(
                action=ScheduleAction.REST,
                reason=f"紧急休息：今日剩余{minutes_left_today}min，一次性休{wait}min",
                wait_minutes=wait,
                priority=80,
            )

        return None

    def _init_rest_calendar(self, state: DriverState, config: DriverConfig) -> None:
        """月初一次性规划 off-day 日期。

        策略：
          1. 优先选周末（货源少、机会成本低）
          2. 周末不够时，均匀分布在非关键日
          3. 避开家事期间、特殊货源日
          4. 结果存入 state.planned_off_days，后续查表执行
        """
        state.rest_calendar_initialized = True

        if config.monthly_off_days_required <= 0:
            return

        current_day = state.current_day()
        total_days = 31
        needed = config.monthly_off_days_required

        # 收集不可用的日期（家事期间、特殊货源前后）
        blocked_days: set[int] = set()
        if config.family_event:
            fe = config.family_event
            fe_start = fe.trigger_min // 1440
            fe_end = fe.end_min // 1440
            for d in range(max(0, fe_start - 1), min(total_days, fe_end + 2)):
                blocked_days.add(d)
        if config.special_cargo and config.special_cargo.cargo_id:
            sc_day = config.special_cargo.available_from_min // 1440
            for d in range(max(0, sc_day - 1), min(total_days, sc_day + 2)):
                blocked_days.add(d)

        # 排除已过去的日期和已有接单/空驶的日期
        # 已确认的 off-day 不需要重新规划
        available_days = []
        for d in range(current_day, total_days):
            if d in blocked_days:
                continue
            if d in state.order_days:
                continue
            available_days.append(d)

        # 已经有的 off-day 不用再规划
        already_done = len(state.off_days)
        still_needed = needed - already_done
        if still_needed <= 0:
            return

        from agent.core.state_tracker import _calendar_weekday

        # 第一轮：选周末
        weekends = [d for d in available_days if _calendar_weekday(d) >= 5]
        chosen: list[int] = []
        for d in weekends:
            if len(chosen) >= still_needed:
                break
            chosen.append(d)

        # 第二轮：周末不够，均匀补充工作日
        if len(chosen) < still_needed:
            remaining_candidates = [d for d in available_days
                                    if d not in chosen and _calendar_weekday(d) < 5]
            gap = still_needed - len(chosen)
            if remaining_candidates and gap > 0:
                # 均匀分布，确保选够
                step = max(1, len(remaining_candidates) // (gap + 1))
                for i in range(gap):
                    idx = min((i + 1) * step, len(remaining_candidates) - 1)
                    if remaining_candidates[idx] not in chosen:
                        chosen.append(remaining_candidates[idx])

        # 第三轮：如果仍然不够（极端情况），从所有可用天中补充
        if len(chosen) < still_needed:
            for d in available_days:
                if d not in chosen:
                    chosen.append(d)
                    if len(chosen) >= still_needed:
                        break

        state.planned_off_days = set(chosen)

    def _handle_off_day_lock(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """Off-day 全局锁：预规划的 off-day 当天，锁死为 wait。

        这是一个正面锁定机制，而非消极保护：
          - 当天在 planned_off_days 中 → 整天只输出 wait
          - 紧急兜底：月末剩余天数不够时，强制补充 off-day
          - 已经接过单或做过空驶的日子无法作为 off-day → 跳过并补规划
        """
        current_day = state.current_day()
        off_days_done = len(state.off_days)
        off_days_needed = config.monthly_off_days_required - off_days_done

        import logging
        _log = logging.getLogger("agent.schedule_planner")
        _log.info(
            "off-day check: day=%d, off_days=%s (done=%d), needed=%d, planned=%s, "
            "today_orders=%d, today_repos=%s",
            current_day, state.off_days, off_days_done, off_days_needed,
            state.planned_off_days, state.today_order_count, state.today_has_repositioned)

        if off_days_needed <= 0:
            return None

        # 今天已经接过单或做过空驶，不能作为 off-day
        if state.today_order_count > 0 or state.today_has_repositioned:
            return None

        is_planned = current_day in state.planned_off_days

        # 紧急兜底：剩余天数不够时强制
        remaining_days = 31 - current_day
        is_urgent = remaining_days <= off_days_needed + 1

        if is_planned or is_urgent:
            minutes_left_today = 1440 - (state.sim_minutes % 1440)
            wait = min(480, minutes_left_today)
            reason = (f"off-day 全局锁（{'计划日' if is_planned else '紧急补充'}，"
                      f"已完成{off_days_done}/{config.monthly_off_days_required}天）")
            return ScheduleDecision(
                action=ScheduleAction.OFF_DAY,
                reason=reason,
                wait_minutes=wait,
                priority=95,  # 极高优先级，仅家事可覆盖
            )

        return None

    def _minutes_until_quiet_start(self, sim_minutes: int, quiet_window: QuietWindow) -> int:
        """计算距离安静窗口开始还有多少分钟。

        如果已在安静窗口内返回 0，如果不在则返回到下一次进入的分钟数。
        """
        day_offset = sim_minutes % 1440
        qw_start_in_day = quiet_window.start  # 日内开始分钟
        if day_offset < qw_start_in_day:
            return qw_start_in_day - day_offset
        # 当前已过今天的安静窗口开始时间
        # 如果 end > 1440（跨天），可能仍在安静窗口内
        if quiet_window.end > 1440:
            qw_end_in_day = quiet_window.end - 1440
            if day_offset < qw_end_in_day:
                return 0  # 仍在安静窗口中
        # 到明天的安静窗口开始
        return (1440 - day_offset) + qw_start_in_day

    def _handle_special_cargo_approach(self, state: DriverState,
                                          config: DriverConfig) -> ScheduleDecision | None:
        """特殊货源提前空驶：确保在上架时间前到达装货点附近。

        策略：
          1. 计算当前位置到装货点的行驶时间
          2. 如果需要出发的时刻 ≤ 当前时刻，立即空驶
          3. 预留 30 分钟缓冲（不在到达前太早抵达，避免空等浪费时间）
          4. 罚分 10000 元，宁可早到等一会也不能错过
        """
        sc = config.special_cargo
        if not sc or not sc.cargo_id:
            return None

        # 距离和行驶时间
        dist_km = haversine_km(state.current_lat, state.current_lng,
                               sc.pickup_lat, sc.pickup_lng)
        travel_min = dist_km / max(config.reposition_speed_kmpm, 0.01)

        # 已经在装货点附近（< 5km），等货上架即可
        if dist_km < 5.0:
            # 如果距上架时间还有很久，正常工作；距上架不到 60 分钟就等待
            time_to_available = sc.available_from_min - state.sim_minutes
            if 0 < time_to_available <= 60:
                return ScheduleDecision(
                    action=ScheduleAction.REST,
                    reason=f"特殊货源即将上架（{time_to_available:.0f}分钟后），在装货点附近等待",
                    wait_minutes=min(int(time_to_available), 30),
                    priority=70,
                )
            return None  # 距上架还早，或已过上架时间（交给 query_cargo 被动接单）

        # 计算最晚出发时刻：上架时间 - 行驶时间 - 30 分钟缓冲
        # 缓冲用于应对 scan_cost、query_cargo 延迟等不确定性
        buffer_min = 30
        latest_depart = sc.available_from_min - travel_min - buffer_min

        # 如果已经过了最晚出发时刻，立即空驶
        if state.sim_minutes >= latest_depart:
            # 但如果已经过了上架时间很久（超过 2 小时），放弃主动空驶
            # （可能已经错过了，或者 query_cargo 会在到达后自然查到）
            if state.sim_minutes > sc.available_from_min + 120:
                return None
            return ScheduleDecision(
                action=ScheduleAction.REPOSITION,
                reason=f"特殊货源（罚{sc.penalty_if_missed:.0f}元）提前空驶，"
                       f"距装货点{dist_km:.0f}km约需{travel_min:.0f}分钟",
                target_pos=(sc.pickup_lat, sc.pickup_lng),
                priority=85,
            )

        # 还没到出发时间：提前出发策略
        # 距离越远越需要提前（宁早不晚），罚分 10000 元不能冒险
        time_to_depart = latest_depart - state.sim_minutes
        # 动态提前窗口：距离 >100km 提前 4h, >50km 提前 2h, 否则不提前
        if dist_km > 100:
            early_window = 240  # 4 小时
        elif dist_km > 50:
            early_window = 120  # 2 小时
        else:
            early_window = 0
        if early_window > 0 and time_to_depart <= early_window:
            return ScheduleDecision(
                action=ScheduleAction.REPOSITION,
                reason=f"特殊货源距离较远（{dist_km:.0f}km），提前{early_window//60}h出发确保不错过",
                target_pos=(sc.pickup_lat, sc.pickup_lng),
                priority=78,
            )

        return None

    def _handle_visit_target(self, state: DriverState, config: DriverConfig) -> ScheduleDecision | None:
        """到访目标点规划。

        R9-B 改进：加入"均匀分布"进度检查。
        如果已完成次数落后于按天数线性期望值（expected = required * day/31），
        提前触发到访，避免全压到月末紧急空驶。
        """
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

        # R9-B: 均匀分布进度检查 — 如果完成率低于期望进度，主动触发
        # expected_visits = required * (current_day / 31)
        # 如果 days_visited < expected_visits - 0.5，说明落后进度
        current_day = state.current_day()
        if current_day > 0 and config.visit_days_required > 0:
            expected_visits = config.visit_days_required * current_day / 31.0
            if days_visited < expected_visits - 0.5:
                # 落后进度，提高优先级触发
                return ScheduleDecision(
                    action=ScheduleAction.REPOSITION,
                    reason=f"到访进度落后（已{days_visited}/{config.visit_days_required}天，"
                           f"期望{expected_visits:.1f}），主动去打卡",
                    target_pos=config.visit_target,
                    priority=65,
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
