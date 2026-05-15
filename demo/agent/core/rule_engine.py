"""规则引擎：对候选货源进行硬约束过滤，无需 LLM。

每条规则基于 DriverConfig + DriverState，判定某条 cargo 是否违反硬约束。
通过后的货源进入 CargoScorer 进行评分排序。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from agent.config.driver_config import DriverConfig, get_config, _parse_datetime_to_sim_minutes
from agent.core.state_tracker import DriverState, haversine_km
from agent.core.timeline_projector import (
    check_go_home_feasible,
    check_family_event_feasible,
    check_load_window_feasible,
    estimate_finish_time,
    project_trip,
    SCAN_COST_MINUTES,
    SAFETY_BUFFER_MINUTES,
)


@dataclass
class FilteredCargo:
    """通过规则引擎过滤的候选货源。"""
    cargo: dict[str, Any]
    pickup_km: float       # 当前位置到取货点的距离
    haul_km: float         # 取货点到卸货点的距离
    is_soft_violated: bool = False  # 有软约束违反（会被罚分但不绝对禁止）
    violation_note: str = ""
    soft_violation_amount: float = 0.0  # 实际每单罚分金额（0 表示使用默认值）


class RuleEngine:
    """基于规则的硬约束过滤器。"""

    def __init__(self) -> None:
        self._consecutive_empty_rounds: dict[str, int] = {}  # driver_id -> 连续全过滤轮次
        self._DEGRADATION_THRESHOLD = 10  # 连续 N 轮全过滤后触发降级

    def filter_cargos(self, cargos: list[dict[str, Any]],
                      state: DriverState, config: DriverConfig) -> list[FilteredCargo]:
        """过滤候选货源，返回满足硬约束的列表。

        安全阀机制：当连续多轮全部被过滤时，自动降级到最小硬约束集
        （仅保留禁止品类 + 月末完成时间），避免司机整月零收入。
        """
        driver_id = state.driver_id
        results: list[FilteredCargo] = []

        # 正常过滤
        for cargo in cargos:
            result = self._evaluate_cargo(cargo, state, config)
            if result is not None:
                results.append(result)

        if results:
            self._consecutive_empty_rounds[driver_id] = 0
            return results

        # 全部被过滤：累加连续空轮次
        self._consecutive_empty_rounds[driver_id] = (
            self._consecutive_empty_rounds.get(driver_id, 0) + 1
        )

        # 达到降级阈值：使用最小硬约束集重新过滤
        if self._consecutive_empty_rounds[driver_id] >= self._DEGRADATION_THRESHOLD:
            results = []
            for cargo in cargos:
                result = self._evaluate_cargo_minimal(cargo, state, config)
                if result is not None:
                    results.append(result)
            if results:
                import logging
                logging.getLogger("agent.rule_engine").warning(
                    "driver=%s: filter degraded after %d empty rounds, "
                    "minimal filter passed %d/%d cargos",
                    driver_id, self._consecutive_empty_rounds[driver_id],
                    len(results), len(cargos))
                # 降级成功后重置计数器（但保持在阈值附近，下次仍可快速降级）
                self._consecutive_empty_rounds[driver_id] = self._DEGRADATION_THRESHOLD - 2

        return results

    def _evaluate_cargo(self, cargo: dict[str, Any],
                        state: DriverState, config: DriverConfig) -> FilteredCargo | None:
        """评估单条 cargo，返回 FilteredCargo 或 None（被过滤掉）。"""

        # 解析货源基本信息
        pickup_lat = float(cargo.get("pickup_lat", 0.0))
        pickup_lng = float(cargo.get("pickup_lng", 0.0))
        delivery_lat = float(cargo.get("delivery_lat", 0.0))
        delivery_lng = float(cargo.get("delivery_lng", 0.0))
        category = str(cargo.get("category", ""))

        # 计算距离
        pickup_km = haversine_km(state.current_lat, state.current_lng, pickup_lat, pickup_lng)
        haul_km = haversine_km(pickup_lat, pickup_lng, delivery_lat, delivery_lng)

        soft_violated = False
        violation_note = ""

        # ===== 硬约束检查 =====

        # 1) 禁止品类
        if category in config.forbidden_categories:
            return None

        # 2) 最大运距
        if config.max_haul_km is not None and haul_km > config.max_haul_km:
            return None

        # 3) 最大空驶距离
        if config.max_pickup_km is not None and pickup_km > config.max_pickup_km:
            return None

        # 4) 每日最大接单数
        if config.max_daily_orders is not None and state.today_order_count >= config.max_daily_orders:
            return None

        # 5) 首单时间限制
        # 注意：偏好含义是"首单不晚于 X 点"，过了 deadline 会被罚分，
        # 但不应阻止接单——过了 deadline 更应该尽快接单减少损失，
        # 而不是完全不接单浪费剩余时间。此处不做过滤，罚分风险
        # 由 CargoScorer 的 soft_violation 机制处理。

        # 6) 地理围栏
        if config.geo_fence:
            fence = config.geo_fence
            # 取货点和卸货点都必须在围栏内
            if not self._in_fence(pickup_lat, pickup_lng, fence):
                return None
            if not self._in_fence(delivery_lat, delivery_lng, fence):
                return None

        # 7) 禁止区域
        if config.forbidden_zone:
            center = config.forbidden_zone["center"]
            radius = config.forbidden_zone["radius_km"]
            # 卸货点不能进入禁区
            if haversine_km(delivery_lat, delivery_lng, center[0], center[1]) <= radius:
                return None
            # 取货点也不能在禁区
            if haversine_km(pickup_lat, pickup_lng, center[0], center[1]) <= radius:
                return None

        # 8) 月度空驶总量限制
        if config.max_monthly_deadhead_km is not None:
            projected_deadhead = state.total_deadhead_km + pickup_km
            if projected_deadhead > config.max_monthly_deadhead_km:
                return None

        # 8.5) 空驶配额动态收紧：配额使用超过 50% 后，限制 pickup 距离
        if config.max_monthly_deadhead_km is not None:
            remaining_budget = config.max_monthly_deadhead_km - state.total_deadhead_km
            budget_usage = 1.0 - remaining_budget / max(config.max_monthly_deadhead_km, 1.0)
            if budget_usage > 0.5:
                # 配额使用超过 50%：动态收紧 pickup 上限
                # 50%→25km, 75%→12km, 100%→5km
                max_pickup = max(5.0, 25.0 * (1.0 - budget_usage) / 0.5)
                if pickup_km > max_pickup:
                    return None

        # 9) 回家约束：接了这单后能否在 deadline 前回家
        #    使用 TimelineProjector 统一推演（修复 P0-Bug1: D009 跨天 deadline 计算）
        if not check_go_home_feasible(state, config, cargo, pickup_km, haul_km):
            return None

        # 10) 装货时间窗检查：到达时已过装货窗口则排除
        #     使用 TimelineProjector 统一推演（修复 P0-Bug3: D003 装货窗口缺 scan_cost）
        if not check_load_window_feasible(state, config, cargo, pickup_km):
            return None

        # 11) 月末 income_eligible 检查：接单后完成时间超月末则不计收入
        #     使用 TimelineProjector 统一推演
        finish_estimate = estimate_finish_time(state, config, cargo, pickup_km, haul_km)
        if finish_estimate > 31 * 1440:
            return None

        # 11.5) 休息余量前瞻：如果运输跨越完整自然日，中间日无法安排休息
        if config.min_continuous_rest_minutes > 0:
            trip_duration = finish_estimate - state.sim_minutes
            if trip_duration > 1440:
                # 运输超过24小时，检查中间跨越的每个完整自然日
                start_day = int(state.sim_minutes) // 1440
                end_day = int(finish_estimate) // 1440
                # 中间完整跨越的天数（不含起始日和结束日）
                full_days_crossed = end_day - start_day - 1
                if full_days_crossed > 0:
                    # 中间有完整自然日被运输覆盖，这些天 rest=0 必定违规
                    # 仅在 rest_weekday_only 时检查是否为工作日
                    if config.rest_weekday_only:
                        from agent.core.state_tracker import _calendar_weekday
                        for d in range(start_day + 1, end_day):
                            if _calendar_weekday(d) < 5:  # 工作日
                                return None
                    else:
                        return None

        # 11.6) 当日休息余量前瞻（软约束，由 CargoScorer 的 rest_lookahead_penalty 处理）
        # 不做硬过滤，避免过度限制长途单导致毛收入大幅下降

        # 12) 安静窗口时间推演：接单后整个执行区间不得与安静窗口重叠
        #     R8.3 修正：恢复全区间检查（R8.2 只查 action_start 导致 D007 29 次违规），
        #     但去掉过度保守的 10%+30min 余量，改为固定 15min。
        if config.quiet_window:
            trip = project_trip(state, config, cargo, pickup_km, haul_km)
            action_start_min = int(trip.action_start_min)
            action_end_min = int(trip.arrive_at_delivery_min) + SCAN_COST_MINUTES
            # R8.4: 固定 30min 余量（15min 不足导致 D003/D004/D007 大量违规）
            # 覆盖 scan_cost 误差 + ceil 误差 + 装卸时间偏差
            action_end_min += 30
            if self._overlaps_quiet_window(action_start_min, action_end_min, config.quiet_window):
                return None

        # 13) 特殊货源路径保护：使用 TimelineProjector 统一推演
        if config.special_cargo and not state.special_cargo_taken:
            sc = config.special_cargo
            time_to_available = sc.available_from_min - state.sim_minutes
            if 0 < time_to_available <= 240:
                trip = project_trip(state, config, cargo, pickup_km, haul_km)
                delivery_to_sc = haversine_km(
                    trip.delivery_lat, trip.delivery_lng,
                    sc.pickup_lat, sc.pickup_lng)
                speed = max(config.reposition_speed_kmpm, 0.01)
                time_to_reach_sc = trip.arrive_at_delivery_min + delivery_to_sc / speed
                if time_to_reach_sc > sc.available_from_min + 60:
                    return None

        # 13.5) 家事约束前瞻：接单后能否赶上家事 deadline
        #        修复 P0-Bug2: D010 家事前瞻触发太晚
        if not check_family_event_feasible(state, config, cargo, pickup_km, haul_km):
            return None

        # ===== 软约束检查 =====

        # 14) 软禁止品类（接了会罚分）
        soft_violation_amount = 0.0
        if category in config.soft_forbidden_categories:
            soft_violated = True
            violation_note = f"软禁止品类: {category}"
            soft_violation_amount = config.penalty_weights.get("category", 0.0)

        # 13) 空驶配额渐进预警（软约束）
        #     当月度空驶余额不足时，不直接拒绝而是标记为软违反，
        #     让 CargoScorer 通过罚分来权衡。
        if config.max_monthly_deadhead_km is not None:
            remaining_budget = config.max_monthly_deadhead_km - state.total_deadhead_km
            # 余额低于 pickup_km 的 5 倍时开始预警（比硬约束更早触发）
            if remaining_budget < pickup_km * 5 and remaining_budget >= pickup_km:
                soft_violated = True
                budget_pct = remaining_budget / config.max_monthly_deadhead_km * 100
                violation_note += f" 空驶配额预警({budget_pct:.0f}%剩余)"

        # 14) 装货等待过长降级（软约束）
        #     如果预计装货等待超过 60 分钟，标记为软违反
        load_time_check = cargo.get("load_time")
        if load_time_check and isinstance(load_time_check, list) and len(load_time_check) == 2:
            try:
                load_start_min = _parse_datetime_to_sim_minutes(str(load_time_check[0]))
                travel_to_pickup_est = pickup_km / max(config.reposition_speed_kmpm, 0.01)
                arrival_est = state.sim_minutes + travel_to_pickup_est
                wait_at_pickup = max(0, load_start_min - arrival_est)
                if wait_at_pickup > 60:
                    soft_violated = True
                    violation_note += f" 装货等待过长({wait_at_pickup:.0f}min)"
            except (ValueError, IndexError):
                pass

        return FilteredCargo(
            cargo=cargo,
            pickup_km=pickup_km,
            haul_km=haul_km,
            is_soft_violated=soft_violated,
            violation_note=violation_note,
            soft_violation_amount=soft_violation_amount,
        )

    def _evaluate_cargo_minimal(self, cargo: dict[str, Any],
                               state: DriverState, config: DriverConfig) -> FilteredCargo | None:
        """最小硬约束过滤：仅保留绝对不可违反的规则。

        用于安全阀降级模式。仅检查：
          1. 禁止品类（硬违规，罚分极高）
          2. 月末完成时间（无法计入收入）
          3. 禁入区域（如有配置）
          4. 安静窗口（每次违规 200 元，每天累计，必须检查）
          5. 地理围栏（违规罚分高，必须检查）
        其余约束（距离、空驶配额等）全部放宽，标记为软违反。
        """
        pickup_lat = float(cargo.get("pickup_lat", 0.0))
        pickup_lng = float(cargo.get("pickup_lng", 0.0))
        delivery_lat = float(cargo.get("delivery_lat", 0.0))
        delivery_lng = float(cargo.get("delivery_lng", 0.0))
        category = str(cargo.get("category", ""))

        pickup_km = haversine_km(state.current_lat, state.current_lng, pickup_lat, pickup_lng)
        haul_km = haversine_km(pickup_lat, pickup_lng, delivery_lat, delivery_lng)

        # 硬红线 1: 禁止品类
        if category in config.forbidden_categories:
            return None

        # 硬红线 2: 月末完成时间
        cost_time = float(cargo.get("cost_time_minutes", 0))
        if cost_time > 0:
            pickup_travel = pickup_km / max(config.reposition_speed_kmpm, 0.01)
            finish_estimate = state.sim_minutes + pickup_travel + cost_time
        else:
            total_trip_km = pickup_km + haul_km
            finish_estimate = state.sim_minutes + total_trip_km / max(config.reposition_speed_kmpm, 0.01)
        if finish_estimate > 31 * 1440:
            return None

        # 硬红线 3: 禁入区域
        if config.forbidden_zone:
            center = config.forbidden_zone["center"]
            radius = config.forbidden_zone["radius_km"]
            if haversine_km(delivery_lat, delivery_lng, center[0], center[1]) <= radius:
                return None
            if haversine_km(pickup_lat, pickup_lng, center[0], center[1]) <= radius:
                return None

        # 硬红线 4: 安静窗口（降级模式也必须检查，违规罚分高且每天累计）
        # R8.5: 与正常模式一致，全区间 + 30min 余量
        if config.quiet_window:
            scan_cost_buffer = SCAN_COST_MINUTES
            action_start_min = state.sim_minutes + scan_cost_buffer
            action_end_min = int(finish_estimate) + scan_cost_buffer
            action_end_min += 30
            if self._overlaps_quiet_window(action_start_min, action_end_min, config.quiet_window):
                return None

        # 硬红线 5: 地理围栏（降级模式也必须检查，违规罚分高）
        if config.geo_fence:
            fence = config.geo_fence
            if not self._in_fence(pickup_lat, pickup_lng, fence):
                return None
            if not self._in_fence(delivery_lat, delivery_lng, fence):
                return None

        # 硬红线 6: 回家约束（每次违规 900 元，降级也不能放过）
        # R8.5: D009 因降级放行导致 14 次违规（12,600 罚分），必须在降级中也检查
        if not check_go_home_feasible(state, config, cargo, pickup_km, haul_km):
            return None

        # 硬红线 7: 家事约束（违规罚分极高：5 元/分钟，可达万元级）
        # R8.5: D010 因降级放行导致 2172 分钟缺席（10,860 罚分）
        if not check_family_event_feasible(state, config, cargo, pickup_km, haul_km):
            return None

        # 其余约束全部视为软违反
        return FilteredCargo(
            cargo=cargo,
            pickup_km=pickup_km,
            haul_km=haul_km,
            is_soft_violated=True,
            violation_note="降级过滤：部分约束已放宽",
        )

    def _overlaps_quiet_window(self, action_start: int, action_end: int,
                               quiet_window: "QuietWindow") -> bool:
        """检查 [action_start, action_end] 是否与任何一天的安静窗口有重叠。

        安静窗口每天重复，所以需要检查 action 区间跨越的所有天的安静窗口。
        对于跨天窗口（如 23:00-06:00，end=1800），每天的窗口覆盖：
          day N: [N*1440 + start, N*1440 + end]
        评测公式：max(a_start, b_start) < min(a_end, b_end)
        """
        from agent.config.driver_config import QuietWindow
        # 计算需要检查的天数范围（包含前一天的跨天窗口）
        start_day = action_start // 1440
        end_day = action_end // 1440
        # 从 start_day-1 开始检查（因为前一天的跨天窗口可能覆盖到 start_day 凌晨）
        check_from = max(0, start_day - 1) if quiet_window.end > 1440 else start_day
        for day in range(check_from, end_day + 1):
            # 计算这一天的安静窗口绝对分钟区间
            qw_abs_start = day * 1440 + quiet_window.start
            qw_abs_end = day * 1440 + quiet_window.end
            # 检查重叠（仅在有效范围内）
            effective_start = max(0, qw_abs_start)
            if max(action_start, effective_start) < min(action_end, qw_abs_end):
                return True
        return False

    def _in_fence(self, lat: float, lng: float, fence: dict[str, float]) -> bool:
        """检查坐标是否在地理围栏内。"""
        return (fence["lat_min"] <= lat <= fence["lat_max"] and
                fence["lng_min"] <= lng <= fence["lng_max"])

    def should_take_special_cargo(self, cargo_id: str, state: DriverState,
                                  config: DriverConfig) -> bool:
        """检查是否为必接的特殊货源。"""
        if config.special_cargo and cargo_id == config.special_cargo.cargo_id:
            if not state.special_cargo_taken:
                return True
        return False
