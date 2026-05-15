"""统一时间推演引擎：所有模块共享的时间/位置/约束推演逻辑。

解决的根本问题：
  rule_engine (45min buffer)、cargo_scorer (60min buffer)、model_decision_service (40min buffer)
  三处时间推演各自独立，安全余量不一致，导致"过滤放行了但评分给高罚分"或反过来的边缘误判。

设计思路：
  提供一组确定性推演函数，输入为 (state, config, cargo)，输出为精确的时间/位置预测。
  所有需要时间推演的模块（RuleEngine / CargoScorer / SchedulePlanner / ModelDecisionService）
  统一调用此处的函数，不再各自估算。

Bug 修复清单：
  P0-Bug1: D009 跨天 deadline — 用 arrive_day 而非 delivery_day 计算 go_home deadline
  P0-Bug2: D010 家事前瞻 — 提供 check_family_event_conflict() 供全链路使用
  P0-Bug3: D003 装货窗口 — arrive_at_pickup 加入 scan_cost
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from agent.config.driver_config import (
    DriverConfig, QuietWindow, FamilyEvent,
    _parse_datetime_to_sim_minutes,
)
from agent.core.state_tracker import DriverState, haversine_km


# ===========================================================================
# 统一常量：全模块共享，不再各自定义
# ===========================================================================

# scan_cost 保守估计（query_cargo 推进仿真时间，取决于返回 items 数量）
# 评测实际 scan_cost ≈ 5-15min，取 10min 作为标准估计
SCAN_COST_MINUTES: int = 10

# 安全余量：用于 SchedulePlanner / ModelDecisionService 的回家/安静窗口触发阈值
# R8.3 修正：从 45 降到 30。
#   45 过度保守导致接单被误拦→级联恶化；
#   15 太小导致许多“刚好能回家”的单被放行但实际回不了（D009 17次违规）。
#   30 = SCAN_COST(10) + CEIL_ERROR(10) + reposition 延迟(10)
SAFETY_BUFFER_MINUTES: int = 30

# ceil 误差上限：仿真引擎对 reposition 执行时间做 ceil(dist/speed)
# 多段行程累积可能达到 ±2min/段，3 段约 ±6min，安全取 10min
CEIL_ERROR_MINUTES: int = 10


# ===========================================================================
# 推演结果数据结构
# ===========================================================================

@dataclass
class TripProjection:
    """一次接单的完整时间推演结果。"""
    # 阶段时间点（仿真绝对分钟）
    action_start_min: float       # 实际动作开始 = sim_minutes + scan_cost
    arrive_at_pickup_min: float   # 到达装货点
    load_ready_min: float         # 装货窗口开始（含等待）
    depart_pickup_min: float      # 从装货点出发 = max(arrive, load_start)
    arrive_at_delivery_min: float # 到达卸货点（运输完成）

    # 距离
    pickup_km: float
    haul_km: float

    # 卸货后的位置
    delivery_lat: float
    delivery_lng: float

    # 卸货完成时刻的日历信息
    delivery_day: int             # 所在仿真日 (0-based)


@dataclass
class GoHomeProjection:
    """接单后回家的时间推演。"""
    home_dist_km: float           # 卸货点到家的距离
    home_travel_min: float        # 回家行驶时间（含 ceil 误差）
    arrive_home_min: float        # 预计到家时间（含安全余量）
    deadline_min: float           # 当天回家 deadline（绝对分钟）
    is_feasible: bool             # 能否在 deadline 前到家
    overshoot_min: float          # 超时分钟数（负值=有余量）


@dataclass
class FamilyEventProjection:
    """接单后对家事的影响推演。"""
    finish_min: float             # 运输完成时刻
    chain_travel_min: float       # 从卸货点到家事链路的总行驶时间
    arrive_at_event_min: float    # 预计到达家事地点时间
    event_deadline_min: float     # 家事 deadline
    is_feasible: bool             # 能否在 deadline 前到达
    overshoot_min: float          # 超时分钟数


# ===========================================================================
# 核心推演函数
# ===========================================================================

def project_trip(
    state: DriverState,
    config: DriverConfig,
    cargo: dict[str, Any],
    pickup_km: float,
    haul_km: float,
) -> TripProjection:
    """推演一次接单的完整时间轨迹。

    这是所有约束检查的基础函数。统一处理：
      - scan_cost（query_cargo 推进时间）
      - 空驶到装货点
      - 装货窗口等待
      - 运输时间（优先用 cost_time_minutes，fallback 用距离/速度）
    """
    speed = max(config.reposition_speed_kmpm, 0.01)

    # Step 1: action_start（考虑 scan_cost）
    action_start = state.sim_minutes + SCAN_COST_MINUTES

    # Step 2: 空驶到装货点
    pickup_travel = pickup_km / speed
    arrive_at_pickup = action_start + pickup_travel

    # Step 3: 装货窗口等待
    load_time = cargo.get("load_time")
    load_start_min = 0.0
    if load_time and isinstance(load_time, list) and len(load_time) == 2:
        try:
            load_start_min = float(_parse_datetime_to_sim_minutes(str(load_time[0])))
        except (ValueError, IndexError):
            pass
    depart_pickup = max(arrive_at_pickup, load_start_min)

    # Step 4: 运输时间
    cost_time = float(cargo.get("cost_time_minutes", 0))
    if cost_time > 0:
        # 使用货源自带的运输耗时（更准确，含路网距离信息）
        # 关键修复 R8.6: 必须从 depart_pickup（含装货窗口等待）算起，
        # 而非 action_start + pickup_travel（忽略了装货窗口等待时间）。
        # 仿真引擎逻辑: finish = max(arrival, load_start) + cost_time_minutes
        # 旧代码: arrive_at_delivery = action_start + pickup_travel + cost_time
        #   → 当有装货窗口且需等待时严重低估（如 D009 cargo 224568: 
        #     估计 1309min vs 实际 2265min，相差 16 小时）
        arrive_at_delivery = depart_pickup + cost_time
    else:
        # fallback: 用直线距离 / 速度
        arrive_at_delivery = depart_pickup + haul_km / speed

    delivery_lat = float(cargo.get("delivery_lat", 0.0))
    delivery_lng = float(cargo.get("delivery_lng", 0.0))
    delivery_day = int(arrive_at_delivery) // 1440

    return TripProjection(
        action_start_min=action_start,
        arrive_at_pickup_min=arrive_at_pickup,
        load_ready_min=load_start_min,
        depart_pickup_min=depart_pickup,
        arrive_at_delivery_min=arrive_at_delivery,
        pickup_km=pickup_km,
        haul_km=haul_km,
        delivery_lat=delivery_lat,
        delivery_lng=delivery_lng,
        delivery_day=delivery_day,
    )


def project_go_home(
    trip: TripProjection,
    config: DriverConfig,
    current_sim_minutes: float = 0.0,
) -> GoHomeProjection:
    """推演接单后能否在 deadline 前回家。

    P0-Bug1 修复核心（v2 回退修正）：
      v1 逻辑用 arrive_home_day 计算 deadline —— 如果接单跨天，deadline 从
      "今天 23:00" 滑到 "明天 23:00"，导致超长单被放行。

      原始 rule_engine 的正确逻辑是：deadline 基于「当前仿真日」（即 current_day），
      含义是"今天必须回家"。如果一单卸货+回家超过今天的 deadline，直接拦截。
      这是保守策略，会误杀部分跨天但次日能回家的单，但有效防止连续多天回不了家。

      修复方案：deadline = (current_day + 1) * 1440 - (24 - home_deadline_hour) * 60
      即用当前仿真日推算今天的回家 deadline，与原始 rule_engine 行为完全一致。
    """
    if not config.home_pos:
        return GoHomeProjection(
            home_dist_km=0, home_travel_min=0,
            arrive_home_min=trip.arrive_at_delivery_min,
            deadline_min=0, is_feasible=True, overshoot_min=0)

    speed = max(config.reposition_speed_kmpm, 0.01)

    # 卸货后到家的距离和时间
    home_dist = haversine_km(
        trip.delivery_lat, trip.delivery_lng,
        config.home_pos[0], config.home_pos[1])
    # 用 ceil 模拟仿真引擎的 reposition 执行时间
    home_travel = math.ceil(home_dist / speed)

    # 预计到家时间 = 卸货完成 + 回家行驶 + scan_cost + ceil误差
    # R8.3: 用 SCAN_COST + CEIL_ERROR = 20min（比 SAFETY_BUFFER_MINUTES 更精确）
    # 过滤场景需要精确，避免误杀好单；调度场景用 SAFETY_BUFFER 提供额外安全边际
    arrive_home = trip.arrive_at_delivery_min + home_travel + SCAN_COST_MINUTES + CEIL_ERROR_MINUTES

    # 关键修复 v2：deadline 基于「当前仿真日」（与原始 rule_engine 一致）
    # current_day = int(current_sim_minutes) // 1440
    # deadline = (current_day + 1) * 1440 - (24 - home_deadline_hour) * 60
    #          = current_day * 1440 + home_deadline_hour * 60
    current_day = int(current_sim_minutes) // 1440
    deadline = current_day * 1440 + config.home_deadline_hour * 60

    is_feasible = arrive_home <= deadline
    overshoot = arrive_home - deadline

    return GoHomeProjection(
        home_dist_km=home_dist,
        home_travel_min=home_travel,
        arrive_home_min=arrive_home,
        deadline_min=deadline,
        is_feasible=is_feasible,
        overshoot_min=overshoot,
    )


def project_family_event(
    trip: TripProjection,
    config: DriverConfig,
) -> FamilyEventProjection | None:
    """推演接单后能否赶上家事。

    P0-Bug2 修复核心：
      旧逻辑只在 SchedulePlanner 的家事 trigger_min 附近才检查，
      D010 在 3/9 21:16 接了 24.6h 的长途单直接跨过 3/10 10:00 deadline。
      
      修复方案：在接单前就推演「卸货完成时间 + 从卸货点到家事链路的行驶时间」，
      如果超过家事 deadline 则拦截。这个检查在 RuleEngine 层执行，
      比 SchedulePlanner 提前 24-48 小时。
    """
    fe = config.family_event
    if not fe:
        return None

    speed = max(config.reposition_speed_kmpm, 0.01)

    # 计算从卸货点到家事链路的完整行程时间
    waypoints = fe.waypoints or []
    if waypoints:
        # 有途经点：卸货点 → 第一个途经点 → 等待 → 家
        first_wp = waypoints[0]
        wp_lat, wp_lng = float(first_wp["lat"]), float(first_wp["lng"])
        wp_wait = int(first_wp.get("wait_minutes", 10))

        dist_to_wp = haversine_km(
            trip.delivery_lat, trip.delivery_lng, wp_lat, wp_lng)
        dist_wp_to_home = haversine_km(
            wp_lat, wp_lng, fe.home_pos[0], fe.home_pos[1])
        chain_travel = (dist_to_wp + dist_wp_to_home) / speed + wp_wait
    else:
        # 无途经点：卸货点 → 家
        dist_to_home = haversine_km(
            trip.delivery_lat, trip.delivery_lng,
            fe.home_pos[0], fe.home_pos[1])
        chain_travel = dist_to_home / speed

    # 预计到达家事地点时间 = 卸货完成 + 链路行程 + scan_cost + ceil误差
    # R8.3: 与 go_home 同理，用 SCAN_COST + CEIL_ERROR
    arrive_at_event = trip.arrive_at_delivery_min + chain_travel + SCAN_COST_MINUTES + CEIL_ERROR_MINUTES

    # 家事 deadline = home_deadline_min（如果是 0 则用 trigger_min 作为 deadline）
    event_deadline = fe.home_deadline_min if fe.home_deadline_min > 0 else fe.trigger_min

    is_feasible = arrive_at_event <= event_deadline
    overshoot = arrive_at_event - event_deadline

    return FamilyEventProjection(
        finish_min=trip.arrive_at_delivery_min,
        chain_travel_min=chain_travel,
        arrive_at_event_min=arrive_at_event,
        event_deadline_min=event_deadline,
        is_feasible=is_feasible,
        overshoot_min=overshoot,
    )


# ===========================================================================
# 约束检查便捷函数（供 RuleEngine 调用）
# ===========================================================================

def check_go_home_feasible(
    state: DriverState,
    config: DriverConfig,
    cargo: dict[str, Any],
    pickup_km: float,
    haul_km: float,
) -> bool:
    """检查接单后能否在 deadline 前回家。True=可行。"""
    if not config.must_return_home or not config.home_pos:
        return True
    trip = project_trip(state, config, cargo, pickup_km, haul_km)
    go_home = project_go_home(trip, config, current_sim_minutes=state.sim_minutes)
    return go_home.is_feasible


def check_family_event_feasible(
    state: DriverState,
    config: DriverConfig,
    cargo: dict[str, Any],
    pickup_km: float,
    haul_km: float,
) -> bool:
    """检查接单后能否赶上家事。True=可行或无家事。"""
    if not config.family_event:
        return True
    # 家事已结束
    if state.family_phase == "done":
        return True
    # 家事已触发并在执行中（此时不应该在接单）
    if state.family_phase not in ("idle",):
        return True

    trip = project_trip(state, config, cargo, pickup_km, haul_km)
    projection = project_family_event(trip, config)
    if projection is None:
        return True
    return projection.is_feasible


def check_load_window_feasible(
    state: DriverState,
    config: DriverConfig,
    cargo: dict[str, Any],
    pickup_km: float,
) -> bool:
    """检查到达装货点时是否还在装货窗口内。

    P0-Bug3 修复核心：
      旧逻辑 `arrival_min = state.sim_minutes + travel_to_pickup` 没加 scan_cost，
      导致多次"空驶到了但货已过期"。
      
      修复方案：统一用 project_trip 计算 arrive_at_pickup（已含 scan_cost），
      与装货窗口 end 比较。
    """
    load_time = cargo.get("load_time")
    if not load_time or not isinstance(load_time, list) or len(load_time) != 2:
        return True
    try:
        load_end_min = _parse_datetime_to_sim_minutes(str(load_time[1]))
    except (ValueError, IndexError):
        return True

    speed = max(config.reposition_speed_kmpm, 0.01)
    # 到达装货点时间 = 当前时间 + scan_cost + 空驶时间
    arrive_at_pickup = state.sim_minutes + SCAN_COST_MINUTES + pickup_km / speed

    return arrive_at_pickup <= load_end_min


def estimate_finish_time(
    state: DriverState,
    config: DriverConfig,
    cargo: dict[str, Any],
    pickup_km: float,
    haul_km: float,
) -> float:
    """估算运输完成时间（绝对分钟），供多处复用。"""
    trip = project_trip(state, config, cargo, pickup_km, haul_km)
    return trip.arrive_at_delivery_min


def compute_go_home_penalty_score(
    state: DriverState,
    config: DriverConfig,
    cargo: dict[str, Any],
    pickup_km: float,
    haul_km: float,
) -> float:
    """计算回家前瞻的罚分评估值（供 CargoScorer 使用）。

    返回 0 表示无风险，正值表示预估罚分。
    """
    if not config.must_return_home or not config.home_pos:
        return 0.0

    trip = project_trip(state, config, cargo, pickup_km, haul_km)
    go_home = project_go_home(trip, config, current_sim_minutes=state.sim_minutes)

    if go_home.is_feasible:
        return 0.0

    # 超时越多罚分越重
    overshoot = go_home.overshoot_min
    penalty_per_day = config.penalty_weights.get("go_home", 900)
    return penalty_per_day * min(3.0, 1.0 + overshoot / 60)
