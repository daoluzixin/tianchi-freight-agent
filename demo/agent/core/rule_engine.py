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


@dataclass
class FilteredCargo:
    """通过规则引擎过滤的候选货源。"""
    cargo: dict[str, Any]
    pickup_km: float       # 当前位置到取货点的距离
    haul_km: float         # 取货点到卸货点的距离
    is_soft_violated: bool = False  # 有软约束违反（会被罚分但不绝对禁止）
    violation_note: str = ""


class RuleEngine:
    """基于规则的硬约束过滤器。"""

    def filter_cargos(self, cargos: list[dict[str, Any]],
                      state: DriverState, config: DriverConfig) -> list[FilteredCargo]:
        """过滤候选货源，返回满足硬约束的列表。"""
        results: list[FilteredCargo] = []

        for cargo in cargos:
            result = self._evaluate_cargo(cargo, state, config)
            if result is not None:
                results.append(result)

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

        # 9) 回家约束：接了这单后能否在 deadline 前回家
        if config.must_return_home and config.home_pos:
            # 使用 cost_time_minutes 估算运输耗时（比纯距离/速度更准确）
            cost_time = float(cargo.get("cost_time_minutes", 0))
            pickup_travel = pickup_km / max(config.reposition_speed_kmpm, 0.01)
            if cost_time > 0:
                arrive_time = state.sim_minutes + pickup_travel + cost_time
            else:
                total_trip_km = pickup_km + haul_km
                arrive_time = state.sim_minutes + total_trip_km / max(config.reposition_speed_kmpm, 0.01)

            # 卸货后到家的距离和时间
            home_dist = haversine_km(delivery_lat, delivery_lng,
                                     config.home_pos[0], config.home_pos[1])
            home_time = home_dist / max(config.reposition_speed_kmpm, 0.01)
            final_arrive = arrive_time + home_time

            # 检查是否能在今天 deadline 前到家
            deadline_today = (state.current_day() + 1) * 1440 - (24 - config.home_deadline_hour) * 60
            if final_arrive > deadline_today:
                return None

        # 10) 装货时间窗检查：到达时已过装货窗口则排除
        load_time = cargo.get("load_time")
        if load_time and isinstance(load_time, list) and len(load_time) == 2:
            try:
                load_end_min = _parse_datetime_to_sim_minutes(str(load_time[1]))
                travel_to_pickup = pickup_km / max(config.reposition_speed_kmpm, 0.01)
                arrival_min = state.sim_minutes + travel_to_pickup
                if arrival_min > load_end_min:
                    return None
            except (ValueError, IndexError):
                pass  # 解析失败不过滤

        # 11) 月末 income_eligible 检查：接单后完成时间超月末则不计收入
        cost_time = float(cargo.get("cost_time_minutes", 0))
        if cost_time > 0:
            pickup_travel = pickup_km / max(config.reposition_speed_kmpm, 0.01)
            finish_estimate = state.sim_minutes + pickup_travel + cost_time
        else:
            total_trip_km = pickup_km + haul_km
            finish_estimate = state.sim_minutes + total_trip_km / max(config.reposition_speed_kmpm, 0.01)
        if finish_estimate > 31 * 1440:
            return None

        # ===== 软约束检查 =====

        # 12) 软禁止品类（接了会罚分）
        if category in config.soft_forbidden_categories:
            soft_violated = True
            violation_note = f"软禁止品类: {category}"

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
        )

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
