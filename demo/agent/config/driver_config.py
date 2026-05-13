"""司机策略配置：纯动态，从 ParsedPreferences 生成 DriverConfig。

核心数据结构 DriverConfig 在运行时由 build_config_from_parsed() 动态构建，
不包含任何硬编码的司机偏好规则。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# 仿真时间常量
SIM_EPOCH_MINUTES = 0
MONTH_END_MINUTES = 31 * 1440  # 44640


@dataclass
class QuietWindow:
    """每天的禁止活动时段（接单/空驶），用 day 内偏移分钟表示。"""
    start: int  # 日内偏移分钟
    end: int    # 日内偏移分钟（跨天时 end > 1440）

    def is_active(self, sim_minutes: int) -> bool:
        day = sim_minutes // 1440
        day_offset = sim_minutes - day * 1440
        if self.end <= 1440:
            return self.start <= day_offset < self.end
        if day_offset >= self.start:
            return True
        if day_offset < (self.end - 1440):
            return True
        return False

    def minutes_until_end(self, sim_minutes: int) -> int:
        day = sim_minutes // 1440
        day_offset = sim_minutes - day * 1440
        if self.end <= 1440:
            return self.end - day_offset if day_offset < self.end else 0
        if day_offset >= self.start:
            return (self.end - 1440) + (1440 - day_offset)
        return (self.end - 1440) - day_offset


@dataclass
class FamilyEvent:
    """家事事件配置。"""
    trigger_min: int = 0
    home_deadline_min: int = 0
    end_min: int = 0
    waypoints: list[dict[str, Any]] = field(default_factory=list)  # [{"lat", "lng", "wait_minutes"}]
    home_pos: tuple[float, float] = (0.0, 0.0)
    penalty_per_minute: float = 5.0
    penalty_once_if_failed: float = 9000.0


@dataclass
class SpecialCargo:
    """必接特殊货源。"""
    cargo_id: str = ""
    available_from_min: int = 0
    pickup_lat: float = 0.0
    pickup_lng: float = 0.0
    penalty_if_missed: float = 10000.0


@dataclass
class DriverConfig:
    """单个司机的完整策略配置（运行时动态构建）。"""
    driver_id: str
    cost_per_km: float = 1.5
    reposition_speed_kmpm: float = 0.8  # 空驶速度 km/min（默认48km/h）

    # 休息约束
    min_continuous_rest_minutes: int = 0
    rest_weekday_only: bool = False
    monthly_off_days_required: int = 0

    # 禁止活动时段
    quiet_window: QuietWindow | None = None

    # 货源过滤
    forbidden_categories: set[str] = field(default_factory=set)
    max_haul_km: float | None = None
    max_pickup_km: float | None = None
    max_daily_orders: int | None = None
    first_order_deadline_hour: int | None = None

    # 地理约束
    geo_fence: dict[str, float] | None = None
    forbidden_zone: dict[str, Any] | None = None
    max_monthly_deadhead_km: float | None = None

    # 回家约束
    must_return_home: bool = False
    home_pos: tuple[float, float] | None = None
    home_deadline_hour: int = 23
    home_quiet_start: int = 23
    home_quiet_end: int = 8

    # 特殊事件
    family_event: FamilyEvent | None = None
    special_cargo: SpecialCargo | None = None

    # 目标点到访
    visit_target: tuple[float, float] | None = None
    visit_days_required: int = 0

    # 建议休息时段
    suggested_rest_start_hour: int = 22
    suggested_rest_end_hour: int = 6

    # 软约束品类
    soft_forbidden_categories: set[str] = field(default_factory=set)

    # 罚分权重（用于评分器评估风险）
    penalty_weights: dict[str, float] = field(default_factory=dict)


# ===========================================================================
# 从 ParsedPreferences 动态构建 DriverConfig
# ===========================================================================

def _parse_datetime_to_sim_minutes(dt_str: str) -> int:
    """将 '2026-03-DD HH:MM:SS' 转为仿真分钟数。"""
    try:
        parts = dt_str.strip().split(" ")
        date_parts = parts[0].split("-")
        time_parts = parts[1].split(":") if len(parts) > 1 else ["0", "0", "0"]
        day = int(date_parts[2]) - 1  # 0-based (3月1日=day0)
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        return day * 1440 + hour * 60 + minute
    except (IndexError, ValueError):
        return 0


def build_config_from_parsed(parsed: Any) -> DriverConfig:
    """从 ParsedPreferences 动态构建 DriverConfig。

    这是泛化的核心：无论什么司机，只要偏好文本被正确解析为 ParsedPreferences，
    就能自动生成对应的 DriverConfig。
    """
    from agent.config.preference_parser import ParsedPreferences
    if not isinstance(parsed, ParsedPreferences):
        return DriverConfig(driver_id="unknown")

    config = DriverConfig(
        driver_id=parsed.driver_id,
        cost_per_km=parsed.cost_per_km,
    )

    # 1) 休息约束：取最大的 min_hours
    if parsed.rest_constraints:
        max_rest = max(parsed.rest_constraints, key=lambda r: r.min_hours)
        config.min_continuous_rest_minutes = int(max_rest.min_hours * 60)
        config.rest_weekday_only = max_rest.weekday_only
        config.penalty_weights["rest"] = max_rest.penalty_per_day

        # 根据休息时长推算建议休息时段
        if config.min_continuous_rest_minutes >= 480:
            config.suggested_rest_start_hour = 22
            config.suggested_rest_end_hour = 6
        elif config.min_continuous_rest_minutes >= 300:
            config.suggested_rest_start_hour = 23
            config.suggested_rest_end_hour = 4
        else:
            config.suggested_rest_start_hour = 0
            config.suggested_rest_end_hour = 3

    # 2) 安静窗口：取最早/最宽的
    if parsed.quiet_windows:
        qw = parsed.quiet_windows[0]
        start_min = qw.start_hour * 60 + qw.start_minute
        end_min = qw.end_hour * 60 + qw.end_minute
        # 处理跨天
        if end_min <= start_min:
            end_min += 1440
        config.quiet_window = QuietWindow(start=start_min, end=end_min)
        config.suggested_rest_start_hour = qw.start_hour
        config.suggested_rest_end_hour = qw.end_hour
        config.penalty_weights["quiet_window"] = qw.penalty_per_day

    # 3) 禁止品类
    for fc in parsed.forbidden_categories:
        if fc.is_soft:
            config.soft_forbidden_categories.update(fc.categories)
        else:
            config.forbidden_categories.update(fc.categories)
        config.penalty_weights["category"] = fc.penalty_per_order

    # 4) 距离限制
    for md in parsed.max_distances:
        if md.constraint_type == "haul":
            config.max_haul_km = md.max_km
        elif md.constraint_type == "pickup":
            config.max_pickup_km = md.max_km
        elif md.constraint_type == "monthly_deadhead":
            config.max_monthly_deadhead_km = md.max_km
        config.penalty_weights[f"distance_{md.constraint_type}"] = md.penalty_per_violation

    # 5) 每日最大接单数
    if parsed.max_orders:
        config.max_daily_orders = parsed.max_orders[0].max_per_day
        config.penalty_weights["max_orders"] = parsed.max_orders[0].penalty_per_extra

    # 6) 首单 deadline
    if parsed.first_order_deadline:
        config.first_order_deadline_hour = parsed.first_order_deadline[0].deadline_hour
        config.penalty_weights["first_order"] = parsed.first_order_deadline[0].penalty_per_day

    # 7) Off-days
    if parsed.off_days:
        config.monthly_off_days_required = max(od.min_days for od in parsed.off_days)
        config.penalty_weights["off_days"] = parsed.off_days[0].penalty_once

    # 8) 地理围栏
    if parsed.geo_fences:
        gf = parsed.geo_fences[0]
        config.geo_fence = {
            "lat_min": gf.lat_min,
            "lat_max": gf.lat_max,
            "lng_min": gf.lng_min,
            "lng_max": gf.lng_max,
        }
        config.penalty_weights["geo_fence"] = gf.penalty_once

    # 9) 禁入区域
    if parsed.forbidden_zones:
        fz = parsed.forbidden_zones[0]
        config.forbidden_zone = {
            "center": (fz.center_lat, fz.center_lng),
            "radius_km": fz.radius_km,
        }
        config.penalty_weights["forbidden_zone"] = fz.penalty_per_entry

    # 10) 回家约束
    if parsed.go_home:
        gh = parsed.go_home[0]
        config.must_return_home = True
        config.home_pos = (gh.home_lat, gh.home_lng)
        config.home_deadline_hour = gh.deadline_hour
        config.home_quiet_start = gh.quiet_start_hour
        config.home_quiet_end = gh.quiet_end_hour
        config.penalty_weights["go_home"] = gh.penalty_per_day

        # 回家约束隐含安静窗口
        if not config.quiet_window:
            start_min = gh.quiet_start_hour * 60
            end_min = gh.quiet_end_hour * 60
            if end_min <= start_min:
                end_min += 1440
            config.quiet_window = QuietWindow(start=start_min, end=end_min)

    # 11) 特殊货源
    if parsed.special_cargos:
        sc = parsed.special_cargos[0]
        config.special_cargo = SpecialCargo(
            cargo_id=sc.cargo_id,
            available_from_min=_parse_datetime_to_sim_minutes(sc.available_from),
            pickup_lat=sc.pickup_lat,
            pickup_lng=sc.pickup_lng,
            penalty_if_missed=sc.penalty_if_missed,
        )

    # 12) 家事事件
    if parsed.family_events:
        fe = parsed.family_events[0]
        waypoints = []
        for wp in fe.waypoints:
            waypoints.append({
                "lat": float(wp.get("lat", 0)),
                "lng": float(wp.get("lng", 0)),
                "wait_minutes": int(wp.get("wait_minutes", 10)),
            })
        config.family_event = FamilyEvent(
            trigger_min=_parse_datetime_to_sim_minutes(fe.trigger_time),
            home_deadline_min=_parse_datetime_to_sim_minutes(fe.home_deadline),
            end_min=_parse_datetime_to_sim_minutes(fe.stay_until),
            waypoints=waypoints,
            home_pos=(fe.home_lat, fe.home_lng),
            penalty_per_minute=fe.penalty_per_minute_late,
            penalty_once_if_failed=fe.penalty_once_if_failed,
        )

    # 13) 到访目标
    if parsed.visit_targets:
        vt = parsed.visit_targets[0]
        config.visit_target = (vt.target_lat, vt.target_lng)
        config.visit_days_required = vt.min_days
        config.penalty_weights["visit_target"] = vt.penalty_once

    return config


# ===========================================================================
# 配置管理器（运行时动态注册 + fallback）
# ===========================================================================

# 动态配置注册表
_dynamic_configs: dict[str, DriverConfig] = {}


def register_config(driver_id: str, config: DriverConfig) -> None:
    """注册动态生成的配置。"""
    _dynamic_configs[driver_id] = config


def get_config(driver_id: str) -> DriverConfig:
    """获取司机配置：优先动态配置 → fallback 到默认。"""
    if driver_id in _dynamic_configs:
        return _dynamic_configs[driver_id]
    # 未注册的司机：返回无约束的默认配置
    return DriverConfig(driver_id=driver_id)


def clear_configs() -> None:
    """清空动态配置（测试用）。"""
    _dynamic_configs.clear()
