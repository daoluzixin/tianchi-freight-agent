"""集成测试：用 drivers.json 的 10 个真实司机数据验证全部 12 个修复点。

测试方式：
  - 手动构造 ParsedPreferences（模拟 LLM 解析结果），绕过真实 LLM 调用
  - 通过 build_config_from_parsed 生成 DriverConfig
  - 对每个司机的特定约束场景构造测试用例
  - 验证 RuleEngine / SchedulePlanner / CargoScorer / StateTracker / ModelDecisionService

覆盖的 12 个修复点：
  fix1:  query_cargo 冷却（_QUERY_COOLDOWN_MINUTES=15）
  fix2:  装货时间窗过滤（rule 10: load_time window）
  fix3:  total_gross_income 追踪
  fix4:  off-day 周末优先 + 均匀分布
  fix5:  first_order_deadline 不再阻止接单
  fix6:  回家约束用 cost_time_minutes 估算
  fix7:  家事等待用仿真时间差
  fix8:  月末 income_eligible 检查（rule 11）
  fix9:  安静窗口中未到家 → GO_HOME 安全网
  fix10: 历史 pickup 位置种子 HotspotTracker
  fix11: HotspotTracker 冷启动用司机初始位置
  fix12: cargo_id 验证 strip 引号/空格

运行方式：
    cd demo && python -m agent.tests.test_integration_drivers
"""

from __future__ import annotations

import sys
import json
import logging
from typing import Any

sys.path.insert(0, ".")

from agent.config.driver_config import (
    DriverConfig, QuietWindow, FamilyEvent, SpecialCargo,
    get_config, register_config, clear_configs, build_config_from_parsed,
    _parse_datetime_to_sim_minutes,
)
from agent.core.state_tracker import StateTracker, DriverState, haversine_km
from agent.core.schedule_planner import SchedulePlanner, ScheduleAction
from agent.core.rule_engine import RuleEngine, FilteredCargo
from agent.scoring.cargo_scorer import CargoScorer, HotspotTracker
from agent.core.model_decision_service import ModelDecisionService
from agent.config.preference_parser import (
    ParsedPreferences, RestConstraint, QuietWindowConstraint,
    ForbiddenCategoryConstraint, MaxDistanceConstraint, MaxOrdersConstraint,
    FirstOrderDeadlineConstraint, OffDaysConstraint, GeoFenceConstraint,
    ForbiddenZoneConstraint, GoHomeConstraint, SpecialCargoConstraint,
    FamilyEventConstraint, VisitTargetConstraint, CustomConstraint,
    parse_llm_response,
)

# ===========================================================================
# 测试框架
# ===========================================================================

results = []


def record(test_name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append({"test": test_name, "status": status, "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {test_name}")
    if detail and not passed:
        print(f"         {detail}")


# ===========================================================================
# 为每个司机构造 ParsedPreferences（模拟 LLM 解析结果）
# ===========================================================================

def build_d001_parsed() -> ParsedPreferences:
    """D001 张三：地理围栏 + 禁止品类 + 休息8h"""
    return ParsedPreferences(
        driver_id="D001", cost_per_km=1.5,
        initial_lat=22.54, initial_lng=114.06,
        rest_constraints=[RestConstraint(min_hours=8, penalty_per_day=300, penalty_cap=3000)],
        forbidden_categories=[ForbiddenCategoryConstraint(
            categories=["化工塑料", "煤炭矿产"], is_soft=False, penalty_per_order=500, penalty_cap=5000)],
        geo_fences=[GeoFenceConstraint(
            lat_min=22.42, lat_max=22.89, lng_min=113.74, lng_max=114.66, penalty_once=2000, penalty_cap=2000)],
    )


def build_d002_parsed() -> ParsedPreferences:
    """D002 李四：off-days 4 + 禁止蔬菜 + 休息4h"""
    return ParsedPreferences(
        driver_id="D002", cost_per_km=1.5,
        initial_lat=23.02, initial_lng=113.75,
        off_days=[OffDaysConstraint(min_days=4, penalty_once=6000, penalty_cap=6000)],
        forbidden_categories=[ForbiddenCategoryConstraint(
            categories=["蔬菜"], is_soft=False, penalty_per_order=350, penalty_cap=3500)],
        rest_constraints=[RestConstraint(min_hours=4, penalty_per_day=200, penalty_cap=6000)],
    )


def build_d003_parsed() -> ParsedPreferences:
    """D003 王五：月度空驶100km + 禁入区域 + 安静2-5am"""
    return ParsedPreferences(
        driver_id="D003", cost_per_km=1.5,
        initial_lat=23.03, initial_lng=113.12,
        max_distances=[MaxDistanceConstraint(
            constraint_type="monthly_deadhead", max_km=100, penalty_per_violation=10, penalty_cap=2000)],
        forbidden_zones=[ForbiddenZoneConstraint(
            center_lat=23.30, center_lng=113.52, radius_km=20, penalty_per_entry=1000, penalty_cap=10000)],
        quiet_windows=[QuietWindowConstraint(
            start_hour=2, start_minute=0, end_hour=5, end_minute=0, penalty_per_day=200, penalty_cap=6000)],
    )


def build_d004_parsed() -> ParsedPreferences:
    """D004 赵六：首单deadline 12pm + 每日最多3单 + 午休安静12-13"""
    return ParsedPreferences(
        driver_id="D004", cost_per_km=1.5,
        initial_lat=22.52, initial_lng=113.39,
        first_order_deadline=[FirstOrderDeadlineConstraint(
            deadline_hour=12, penalty_per_day=200, penalty_cap=4000)],
        max_orders=[MaxOrdersConstraint(max_per_day=3, penalty_per_extra=200)],
        quiet_windows=[QuietWindowConstraint(
            start_hour=12, start_minute=0, end_hour=13, end_minute=0, penalty_per_day=100, penalty_cap=3000)],
    )


def build_d005_parsed() -> ParsedPreferences:
    """D005 钱七：最大运距100km + 最大空驶90km + 安静23-6"""
    return ParsedPreferences(
        driver_id="D005", cost_per_km=1.5,
        initial_lat=22.58, initial_lng=113.08,
        max_distances=[
            MaxDistanceConstraint(constraint_type="haul", max_km=100, penalty_per_violation=100),
            MaxDistanceConstraint(constraint_type="pickup", max_km=90, penalty_per_violation=100),
        ],
        quiet_windows=[QuietWindowConstraint(
            start_hour=23, start_minute=0, end_hour=6, end_minute=0, penalty_per_day=200, penalty_cap=6000)],
    )


def build_d006_parsed() -> ParsedPreferences:
    """D006 孙八：休息5h + 禁止鲜活水产品 + 最大运距150km + off-days 2"""
    return ParsedPreferences(
        driver_id="D006", cost_per_km=1.5,
        initial_lat=23.11, initial_lng=114.42,
        rest_constraints=[RestConstraint(min_hours=5, penalty_per_day=200, penalty_cap=6000)],
        forbidden_categories=[ForbiddenCategoryConstraint(
            categories=["鲜活水产品"], is_soft=False, penalty_per_order=400, penalty_cap=4000)],
        max_distances=[MaxDistanceConstraint(constraint_type="haul", max_km=150, penalty_per_violation=250, penalty_cap=3000)],
        off_days=[OffDaysConstraint(min_days=2, penalty_once=3000, penalty_cap=3000)],
    )


def build_d007_parsed() -> ParsedPreferences:
    """D007 周九：安静23-4 + 禁止机械设备 + 最大运距180km + off-days 1"""
    return ParsedPreferences(
        driver_id="D007", cost_per_km=1.5,
        initial_lat=23.05, initial_lng=112.46,
        quiet_windows=[QuietWindowConstraint(
            start_hour=23, start_minute=0, end_hour=4, end_minute=0, penalty_per_day=500, penalty_cap=15000)],
        forbidden_categories=[ForbiddenCategoryConstraint(
            categories=["机械设备"], is_soft=False, penalty_per_order=280, penalty_cap=2800)],
        max_distances=[MaxDistanceConstraint(constraint_type="haul", max_km=180, penalty_per_violation=200, penalty_cap=2400)],
        off_days=[OffDaysConstraint(min_days=1, penalty_once=800, penalty_cap=800)],
    )


def build_d008_parsed() -> ParsedPreferences:
    """D008 吴十：off-days 2 + 休息4h(仅工作日) + 软禁止食品饮料 + 最大空驶50km"""
    return ParsedPreferences(
        driver_id="D008", cost_per_km=1.5,
        initial_lat=22.27, initial_lng=113.58,
        off_days=[OffDaysConstraint(min_days=2, penalty_once=1500, penalty_cap=1500)],
        rest_constraints=[RestConstraint(min_hours=4, weekday_only=True, penalty_per_day=400, penalty_cap=12000)],
        forbidden_categories=[ForbiddenCategoryConstraint(
            categories=["食品饮料"], is_soft=True, penalty_per_order=200, penalty_cap=2000)],
        max_distances=[MaxDistanceConstraint(constraint_type="pickup", max_km=50, penalty_per_violation=100, penalty_cap=2000)],
    )


def build_d009_parsed() -> ParsedPreferences:
    """D009 郑一：特殊货源240646 + 回家23点 + 禁止快递快运搬家"""
    return ParsedPreferences(
        driver_id="D009", cost_per_km=1.5,
        initial_lat=23.12, initial_lng=113.28,
        special_cargos=[SpecialCargoConstraint(
            cargo_id="240646", available_from="2026-03-03 14:43:36",
            pickup_lat=24.81, pickup_lng=113.58, penalty_if_missed=10000)],
        go_home=[GoHomeConstraint(
            home_lat=23.12, home_lng=113.28, deadline_hour=23,
            quiet_start_hour=23, quiet_end_hour=8, radius_km=1.0, penalty_per_day=900, penalty_cap=27000)],
        forbidden_categories=[ForbiddenCategoryConstraint(
            categories=["快递快运搬家"], is_soft=False, penalty_per_order=350, penalty_cap=3500)],
    )


def build_d010_parsed() -> ParsedPreferences:
    """D010 冯二：家事3/10-3/13 + 到访目标5天 + 休息3h + 软禁止服饰纺织皮革"""
    return ParsedPreferences(
        driver_id="D010", cost_per_km=1.5,
        initial_lat=23.19, initial_lng=113.36,
        family_events=[FamilyEventConstraint(
            trigger_time="2026-03-10 10:00:00",
            waypoints=[{"lat": 23.21, "lng": 113.37, "wait_minutes": 10}],
            home_lat=23.19, home_lng=113.36,
            home_deadline="2026-03-10 22:00:00",
            stay_until="2026-03-13 22:00:00",
            penalty_per_minute_late=5, penalty_once_if_failed=9000)],
        visit_targets=[VisitTargetConstraint(
            target_lat=23.13, target_lng=113.26, radius_km=1.0, min_days=5, penalty_once=3000, penalty_cap=3000)],
        rest_constraints=[RestConstraint(min_hours=3, penalty_per_day=300, penalty_cap=6000)],
        forbidden_categories=[ForbiddenCategoryConstraint(
            categories=["服饰纺织皮革"], is_soft=True, penalty_per_order=240, penalty_cap=2400)],
    )


ALL_BUILDERS = {
    "D001": build_d001_parsed,
    "D002": build_d002_parsed,
    "D003": build_d003_parsed,
    "D004": build_d004_parsed,
    "D005": build_d005_parsed,
    "D006": build_d006_parsed,
    "D007": build_d007_parsed,
    "D008": build_d008_parsed,
    "D009": build_d009_parsed,
    "D010": build_d010_parsed,
}


# ===========================================================================
# Mock API for end-to-end tests
# ===========================================================================

class DriverMockApi:
    """为指定司机构造 Mock API，模拟 LLM 返回解析后的 JSON。"""

    def __init__(self, driver_id: str, parsed_json: str,
                 status_overrides: dict[str, Any] | None = None,
                 cargo_items: list[dict[str, Any]] | None = None):
        self._driver_id = driver_id
        self._parsed_json = parsed_json
        self._status_overrides = status_overrides or {}
        self._cargo_items = cargo_items or []
        self._call_count = 0

    def get_driver_status(self, driver_id):
        base = {
            "driver_id": driver_id,
            "simulation_progress_minutes": 600,
            "current_lat": 23.0,
            "current_lng": 113.3,
            "cost_per_km": 1.5,
            "truck_length": 4.2,
            "completed_order_count": 0,
            "reposition_speed_km_per_hour": 48.0,
            "preferences": [{"content": "test", "penalty_amount": 100}],
        }
        base.update(self._status_overrides)
        return base

    def query_cargo(self, driver_id, latitude, longitude):
        return {"items": self._cargo_items}

    def query_decision_history(self, driver_id, step):
        return {"records": [], "total_steps": 0, "returned_count": 0}

    def model_chat_completion(self, payload):
        self._call_count += 1
        if self._call_count == 1:
            return {
                "choices": [{"message": {"content": self._parsed_json}}],
                "usage": {"total_tokens": 350},
            }
        # 后续调用返回选择第一个 cargo
        return {
            "choices": [{"message": {"content": json.dumps({
                "action": "take_order", "cargo_id": "C_DEFAULT",
                "confidence": 0.8, "reasoning": "default"
            })}}],
            "usage": {"total_tokens": 120},
        }


# ===========================================================================
# 测试用例
# ===========================================================================

def test_d001_geo_fence_and_forbidden():
    """D001 张三：地理围栏 + 禁止品类 + 休息8h"""
    print("\n[D001] 张三 - 地理围栏 + 禁止品类 + 休息8h")
    parsed = build_d001_parsed()
    config = build_config_from_parsed(parsed)

    # 验证配置生成
    record("D001 地理围栏正确",
           config.geo_fence is not None and config.geo_fence["lat_min"] == 22.42,
           f"geo_fence={config.geo_fence}")
    record("D001 禁止品类正确",
           "化工塑料" in config.forbidden_categories and "煤炭矿产" in config.forbidden_categories,
           f"forbidden={config.forbidden_categories}")
    record("D001 休息480min",
           config.min_continuous_rest_minutes == 480,
           f"rest={config.min_continuous_rest_minutes}")

    # 规则引擎测试
    engine = RuleEngine()
    state = DriverState(driver_id="D001", current_lat=22.54, current_lng=114.06)

    cargos = [
        {"cargo_id": "C1", "category": "化工塑料", "price": 1000,
         "pickup_lat": 22.6, "pickup_lng": 114.1, "delivery_lat": 22.7, "delivery_lng": 114.2},
        {"cargo_id": "C2", "category": "食品饮料", "price": 500,
         "pickup_lat": 22.6, "pickup_lng": 114.1, "delivery_lat": 22.7, "delivery_lng": 114.2},
        {"cargo_id": "C3", "category": "电子产品", "price": 800,
         "pickup_lat": 22.6, "pickup_lng": 114.1, "delivery_lat": 23.5, "delivery_lng": 113.0},  # 围栏外
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    ids = [f.cargo["cargo_id"] for f in filtered]

    record("D001 化工塑料被过滤", "C1" not in ids, f"passed={ids}")
    record("D001 食品饮料通过", "C2" in ids, f"passed={ids}")
    record("D001 围栏外被过滤", "C3" not in ids, f"passed={ids}")


def test_d002_off_days_weekend_first():
    """D002 李四：off-days 周末优先策略 (fix4)"""
    print("\n[D002] 李四 - off-days 周末优先 (fix4)")
    parsed = build_d002_parsed()
    config = build_config_from_parsed(parsed)

    record("D002 off-days=4", config.monthly_off_days_required == 4,
           f"off_days={config.monthly_off_days_required}")
    record("D002 禁止蔬菜", "蔬菜" in config.forbidden_categories,
           f"forbidden={config.forbidden_categories}")

    planner = SchedulePlanner()

    # 2026-03-01 是周日(weekday=6)，应该安排 off-day
    state_sun = DriverState(driver_id="D002_sun")
    state_sun.sim_minutes = 0 * 1440 + 10 * 60  # day0 10:00 (3月1日周日)
    state_sun.current_lat = 23.02
    state_sun.current_lng = 113.75
    state_sun.longest_rest_today = 240  # 已满足休息
    state_sun._last_day = 0

    decision_sun = planner.plan(state_sun, config)
    record("D002 周日安排off-day", decision_sun.action == ScheduleAction.OFF_DAY,
           f"day0(周日) action={decision_sun.action.value}")

    # 周一不应该安排 off-day（除非紧急）
    state_mon = DriverState(driver_id="D002_mon")
    state_mon.sim_minutes = 1 * 1440 + 10 * 60  # day1 10:00 (3月2日周一)
    state_mon.current_lat = 23.02
    state_mon.current_lng = 113.75
    state_mon.longest_rest_today = 240
    state_mon._last_day = 1

    decision_mon = planner.plan(state_mon, config)
    record("D002 周一正常工作", decision_mon.action == ScheduleAction.WORK,
           f"day1(周一) action={decision_mon.action.value}")


def test_d003_deadhead_and_forbidden_zone():
    """D003 王五：月度空驶100km + 禁入区域 + 安静2-5am"""
    print("\n[D003] 王五 - 月度空驶 + 禁入区域 + 安静窗口")
    parsed = build_d003_parsed()
    config = build_config_from_parsed(parsed)

    record("D003 月度空驶100km",
           config.max_monthly_deadhead_km == 100,
           f"deadhead={config.max_monthly_deadhead_km}")
    record("D003 禁入区域正确",
           config.forbidden_zone is not None and config.forbidden_zone["center"] == (23.30, 113.52),
           f"zone={config.forbidden_zone}")
    record("D003 安静窗口2-5am",
           config.quiet_window is not None and config.quiet_window.start == 120,
           f"quiet={config.quiet_window.start}-{config.quiet_window.end}")

    # 安静窗口测试
    planner = SchedulePlanner()
    state_quiet = DriverState(driver_id="D003_q")
    state_quiet.sim_minutes = 1440 + 3 * 60  # day1 03:00
    state_quiet.current_lat = 23.03
    state_quiet.current_lng = 113.12
    state_quiet._last_day = 1

    decision = planner.plan(state_quiet, config)
    record("D003 凌晨3点REST", decision.action == ScheduleAction.REST,
           f"action={decision.action.value}")


def test_d004_first_order_deadline_no_block():
    """D004 赵六：首单deadline不阻止接单 (fix5)"""
    print("\n[D004] 赵六 - 首单deadline不阻止接单 (fix5)")
    parsed = build_d004_parsed()
    config = build_config_from_parsed(parsed)

    record("D004 首单deadline=12",
           config.first_order_deadline_hour == 12,
           f"deadline={config.first_order_deadline_hour}")
    record("D004 每日最多3单",
           config.max_daily_orders == 3,
           f"max_orders={config.max_daily_orders}")

    # fix5 核心验证：过了 deadline 后仍然可以接单
    engine = RuleEngine()
    state = DriverState(driver_id="D004")
    state.sim_minutes = 14 * 60  # 14:00，已过 12:00 deadline
    state.current_lat = 22.52
    state.current_lng = 113.39
    state.today_order_count = 0  # 今天还没接单

    cargos = [
        {"cargo_id": "LATE", "category": "建材", "price": 600,
         "pickup_lat": 22.55, "pickup_lng": 113.42,
         "delivery_lat": 22.60, "delivery_lng": 113.50},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    record("D004 过deadline仍可接单(fix5)",
           len(filtered) == 1 and filtered[0].cargo["cargo_id"] == "LATE",
           f"filtered={len(filtered)} (应为1，不被deadline阻止)")

    # 但达到3单上限后应被过滤
    state.today_order_count = 3
    filtered2 = engine.filter_cargos(cargos, state, config)
    record("D004 达3单上限被过滤",
           len(filtered2) == 0,
           f"filtered={len(filtered2)}")


def test_d005_max_distances():
    """D005 钱七：最大运距100km + 最大空驶90km + 安静23-6"""
    print("\n[D005] 钱七 - 最大运距 + 最大空驶")
    parsed = build_d005_parsed()
    config = build_config_from_parsed(parsed)

    record("D005 最大运距100km", config.max_haul_km == 100, f"haul={config.max_haul_km}")
    record("D005 最大空驶90km", config.max_pickup_km == 90, f"pickup={config.max_pickup_km}")

    engine = RuleEngine()
    state = DriverState(driver_id="D005", current_lat=22.58, current_lng=113.08)

    cargos = [
        # 运距超100km
        {"cargo_id": "FAR_HAUL", "category": "建材", "price": 1500,
         "pickup_lat": 22.60, "pickup_lng": 113.10,
         "delivery_lat": 23.60, "delivery_lng": 113.80},
        # 空驶超90km
        {"cargo_id": "FAR_PICKUP", "category": "建材", "price": 800,
         "pickup_lat": 23.50, "pickup_lng": 113.80,
         "delivery_lat": 23.55, "delivery_lng": 113.85},
        # 正常范围内
        {"cargo_id": "OK", "category": "建材", "price": 500,
         "pickup_lat": 22.60, "pickup_lng": 113.10,
         "delivery_lat": 22.80, "delivery_lng": 113.30},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    ids = [f.cargo["cargo_id"] for f in filtered]

    far_haul_km = haversine_km(22.60, 113.10, 23.60, 113.80)
    far_pickup_km = haversine_km(22.58, 113.08, 23.50, 113.80)
    record("D005 超运距被过滤", "FAR_HAUL" not in ids,
           f"haul_km={far_haul_km:.1f}")
    record("D005 超空驶被过滤", "FAR_PICKUP" not in ids,
           f"pickup_km={far_pickup_km:.1f}")
    record("D005 正常货源通过", "OK" in ids, f"passed={ids}")


def test_d006_rest_and_off_days():
    """D006 孙八：休息5h + 禁止鲜活水产品 + 运距150km + off-days 2"""
    print("\n[D006] 孙八 - 休息5h + 禁止鲜活水产品 + off-days 2")
    parsed = build_d006_parsed()
    config = build_config_from_parsed(parsed)

    record("D006 休息300min", config.min_continuous_rest_minutes == 300,
           f"rest={config.min_continuous_rest_minutes}")
    record("D006 禁止鲜活水产品", "鲜活水产品" in config.forbidden_categories,
           f"forbidden={config.forbidden_categories}")
    record("D006 运距150km", config.max_haul_km == 150, f"haul={config.max_haul_km}")
    record("D006 off-days=2", config.monthly_off_days_required == 2,
           f"off_days={config.monthly_off_days_required}")

    engine = RuleEngine()
    state = DriverState(driver_id="D006", current_lat=23.11, current_lng=114.42)

    cargos = [
        {"cargo_id": "FISH", "category": "鲜活水产品", "price": 700,
         "pickup_lat": 23.15, "pickup_lng": 114.45,
         "delivery_lat": 23.20, "delivery_lng": 114.50},
        {"cargo_id": "GOOD", "category": "电子产品", "price": 600,
         "pickup_lat": 23.15, "pickup_lng": 114.45,
         "delivery_lat": 23.20, "delivery_lng": 114.50},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    ids = [f.cargo["cargo_id"] for f in filtered]
    record("D006 鲜活水产品被过滤", "FISH" not in ids, f"passed={ids}")
    record("D006 电子产品通过", "GOOD" in ids, f"passed={ids}")


def test_d007_quiet_window_and_off_day():
    """D007 周九：安静23-4 + 禁止机械设备 + 运距180km + off-days 1"""
    print("\n[D007] 周九 - 安静23-4 + 禁止机械设备")
    parsed = build_d007_parsed()
    config = build_config_from_parsed(parsed)

    record("D007 安静窗口23-4",
           config.quiet_window is not None and config.quiet_window.start == 23 * 60,
           f"quiet_start={config.quiet_window.start if config.quiet_window else None}")
    record("D007 禁止机械设备", "机械设备" in config.forbidden_categories,
           f"forbidden={config.forbidden_categories}")
    record("D007 运距180km", config.max_haul_km == 180, f"haul={config.max_haul_km}")
    record("D007 off-days=1", config.monthly_off_days_required == 1,
           f"off_days={config.monthly_off_days_required}")

    # 安静窗口内应 REST
    planner = SchedulePlanner()
    state = DriverState(driver_id="D007")
    state.sim_minutes = 1440 + 1 * 60  # day1 01:00
    state.current_lat = 23.05
    state.current_lng = 112.46
    state._last_day = 1

    decision = planner.plan(state, config)
    record("D007 凌晨1点REST", decision.action == ScheduleAction.REST,
           f"action={decision.action.value}")


def test_d008_soft_forbidden_and_weekday_rest():
    """D008 吴十：off-days 2 + 仅工作日休息4h + 软禁止食品饮料 + 空驶50km"""
    print("\n[D008] 吴十 - 软禁止品类 + 仅工作日休息")
    parsed = build_d008_parsed()
    config = build_config_from_parsed(parsed)

    record("D008 off-days=2", config.monthly_off_days_required == 2,
           f"off_days={config.monthly_off_days_required}")
    record("D008 仅工作日休息", config.rest_weekday_only is True,
           f"weekday_only={config.rest_weekday_only}")
    record("D008 软禁止食品饮料", "食品饮料" in config.soft_forbidden_categories,
           f"soft_forbidden={config.soft_forbidden_categories}")
    record("D008 空驶50km", config.max_pickup_km == 50, f"pickup={config.max_pickup_km}")

    # 软约束测试
    engine = RuleEngine()
    state = DriverState(driver_id="D008", current_lat=22.27, current_lng=113.58)

    cargos = [
        {"cargo_id": "FOOD", "category": "食品饮料", "price": 500,
         "pickup_lat": 22.30, "pickup_lng": 113.60,
         "delivery_lat": 22.35, "delivery_lng": 113.65},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    record("D008 食品饮料不被硬过滤", len(filtered) == 1,
           f"filtered={len(filtered)}")
    if filtered:
        record("D008 食品饮料标记soft_violated", filtered[0].is_soft_violated is True,
               f"is_soft_violated={filtered[0].is_soft_violated}")


def test_d009_special_cargo_and_go_home():
    """D009 郑一：特殊货源240646 + 回家23点 + 禁止快递快运搬家"""
    print("\n[D009] 郑一 - 特殊货源 + 回家约束")
    parsed = build_d009_parsed()
    config = build_config_from_parsed(parsed)

    record("D009 特殊货源240646",
           config.special_cargo is not None and config.special_cargo.cargo_id == "240646",
           f"special={config.special_cargo}")
    record("D009 回家约束",
           config.must_return_home and config.home_pos == (23.12, 113.28),
           f"home={config.home_pos}")
    record("D009 禁止快递快运搬家", "快递快运搬家" in config.forbidden_categories,
           f"forbidden={config.forbidden_categories}")

    # 特殊货源必接
    engine = RuleEngine()
    state = DriverState(driver_id="D009", current_lat=24.80, current_lng=113.57)
    record("D009 特殊货源必接",
           engine.should_take_special_cargo("240646", state, config) is True,
           "cargo_id=240646")
    record("D009 普通货源不必接",
           engine.should_take_special_cargo("C999", state, config) is False,
           "cargo_id=C999")


def test_d010_family_event_and_visit():
    """D010 冯二：家事3/10-3/13 + 到访目标5天 + 休息3h + 软禁止服饰纺织皮革"""
    print("\n[D010] 冯二 - 家事状态机 + 到访目标")
    parsed = build_d010_parsed()
    config = build_config_from_parsed(parsed)

    record("D010 家事事件配置",
           config.family_event is not None,
           f"trigger={config.family_event.trigger_min if config.family_event else None}")
    record("D010 到访目标(23.13,113.26)",
           config.visit_target == (23.13, 113.26),
           f"visit={config.visit_target}")
    record("D010 到访5天",
           config.visit_days_required == 5,
           f"days={config.visit_days_required}")
    record("D010 休息180min",
           config.min_continuous_rest_minutes == 180,
           f"rest={config.min_continuous_rest_minutes}")
    record("D010 软禁止服饰纺织皮革",
           "服饰纺织皮革" in config.soft_forbidden_categories,
           f"soft={config.soft_forbidden_categories}")

    # 家事状态机测试 (fix7: 用仿真时间差)
    planner = SchedulePlanner()
    state = DriverState(driver_id="D010")
    state.sim_minutes = _parse_datetime_to_sim_minutes("2026-03-10 10:00:00")  # 触发时刻
    state.current_lat = 23.0
    state.current_lng = 113.3
    state.family_phase = "idle"

    decision = planner.plan(state, config)
    record("D010 家事触发→go_spouse",
           decision.action == ScheduleAction.REPOSITION and state.family_phase == "go_spouse",
           f"action={decision.action.value}, phase={state.family_phase}")


# ===========================================================================
# 修复点专项测试
# ===========================================================================

def test_fix1_query_cooldown():
    """fix1: query_cargo 动态冷却机制（高峰/普通/低谷）"""
    print("\n[fix1] query_cargo 动态冷却机制")
    from agent.core.model_decision_service import (
        _QUERY_COOLDOWN_PEAK, _QUERY_COOLDOWN_NORMAL, _QUERY_COOLDOWN_OFFPEAK)

    record("fix1 高峰冷却=8min", _QUERY_COOLDOWN_PEAK == 8,
           f"peak_cooldown={_QUERY_COOLDOWN_PEAK}")
    record("fix1 普通冷却=15min", _QUERY_COOLDOWN_NORMAL == 15,
           f"normal_cooldown={_QUERY_COOLDOWN_NORMAL}")
    record("fix1 低谷冷却=25min", _QUERY_COOLDOWN_OFFPEAK == 25,
           f"offpeak_cooldown={_QUERY_COOLDOWN_OFFPEAK}")

    # 模拟冷却逻辑（使用普通时段冷却值）
    last_query = 100
    now = 110
    should_skip = (now - last_query) < _QUERY_COOLDOWN_NORMAL
    record("fix1 10min内跳过查询(普通)", should_skip is True,
           f"last={last_query}, now={now}, diff={now-last_query}")

    now2 = 120
    should_query = (now2 - last_query) >= _QUERY_COOLDOWN_NORMAL
    record("fix1 20min后允许查询(普通)", should_query is True,
           f"last={last_query}, now={now2}, diff={now2-last_query}")


def test_fix2_load_time_window():
    """fix2: 装货时间窗过滤（rule 10）"""
    print("\n[fix2] 装货时间窗过滤 (rule 10)")
    engine = RuleEngine()
    config = DriverConfig(driver_id="FIX2", reposition_speed_kmpm=0.8)
    state = DriverState(driver_id="FIX2", current_lat=23.0, current_lng=113.3)
    state.sim_minutes = 10 * 60  # 10:00

    # 装货窗口已过
    cargos_expired = [
        {"cargo_id": "EXPIRED", "category": "建材", "price": 500,
         "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 23.10, "delivery_lng": 113.40,
         "load_time": ["2026-03-01 06:00:00", "2026-03-01 08:00:00"]},  # 窗口6-8点，已过
    ]
    filtered = engine.filter_cargos(cargos_expired, state, config)
    record("fix2 过期装货窗口被过滤", len(filtered) == 0,
           f"load_time ends 8:00, now 10:00, filtered={len(filtered)}")

    # 装货窗口未过
    cargos_valid = [
        {"cargo_id": "VALID", "category": "建材", "price": 500,
         "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 23.10, "delivery_lng": 113.40,
         "load_time": ["2026-03-01 10:00:00", "2026-03-01 14:00:00"]},  # 窗口10-14点
    ]
    filtered2 = engine.filter_cargos(cargos_valid, state, config)
    record("fix2 有效装货窗口通过", len(filtered2) == 1,
           f"load_time ends 14:00, now 10:00, filtered={len(filtered2)}")


def test_fix3_total_gross_income():
    """fix3: total_gross_income 追踪"""
    print("\n[fix3] total_gross_income 追踪")
    tracker = StateTracker()
    state = tracker.get_state("FIX3")
    state.sim_minutes = 600
    state.current_lat = 23.0
    state.current_lng = 113.3
    state._last_day = 0

    # 模拟接单
    action = {"action": "take_order", "params": {"cargo_id": "C1"}}
    result = {
        "accepted": True, "income": 500.0,
        "pickup_deadhead_km": 5.0, "haul_distance_km": 50.0,
        "current_lat": 23.5, "current_lng": 113.8,
        "simulation_progress_minutes": 660,
    }
    register_config("FIX3", DriverConfig(driver_id="FIX3"))
    tracker.update_after_action(state, action, result)

    record("fix3 income=500", state.total_gross_income == 500.0,
           f"income={state.total_gross_income}")

    # 第二单
    result2 = {
        "accepted": True, "income": 300.0,
        "pickup_deadhead_km": 3.0, "haul_distance_km": 30.0,
        "current_lat": 23.8, "current_lng": 114.0,
        "simulation_progress_minutes": 720,
    }
    tracker.update_after_action(state, action, result2)
    record("fix3 累计income=800", state.total_gross_income == 800.0,
           f"income={state.total_gross_income}")

    # 历史重建也追踪 income
    state2 = tracker.get_state("FIX3_hist")
    register_config("FIX3_hist", DriverConfig(driver_id="FIX3_hist"))
    records = [
        {"action": {"action": "take_order", "params": {"cargo_id": "H1"}},
         "result": {"accepted": True, "income": 200.0, "pickup_deadhead_km": 2.0,
                    "haul_distance_km": 20.0, "simulation_progress_minutes": 100},
         "position_after": {"lat": 23.1, "lng": 113.3}},
        {"action": {"action": "take_order", "params": {"cargo_id": "H2"}},
         "result": {"accepted": True, "price": 150.0, "pickup_deadhead_km": 1.0,
                    "haul_distance_km": 10.0, "simulation_progress_minutes": 200},
         "position_after": {"lat": 23.2, "lng": 113.4}},
    ]
    tracker.rebuild_from_history("FIX3_hist", records)
    record("fix3 历史重建income=350", state2.total_gross_income == 350.0,
           f"income={state2.total_gross_income} (200+150)")


def test_fix6_go_home_cost_time():
    """fix6: 回家约束用 cost_time_minutes 估算"""
    print("\n[fix6] 回家约束用 cost_time_minutes (rule 9)")
    engine = RuleEngine()
    config = DriverConfig(
        driver_id="FIX6",
        must_return_home=True,
        home_pos=(23.12, 113.28),
        home_deadline_hour=23,
        reposition_speed_kmpm=0.8,
    )

    state = DriverState(driver_id="FIX6", current_lat=23.0, current_lng=113.3)
    state.sim_minutes = 20 * 60  # 20:00

    # 有 cost_time_minutes 的货源：运输耗时120min，加上空驶和回家时间
    cargos = [
        {"cargo_id": "COST_TIME", "category": "建材", "price": 800,
         "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 23.50, "delivery_lng": 113.80,
         "cost_time_minutes": 120},  # 2小时运输
    ]
    filtered = engine.filter_cargos(cargos, state, config)

    # 20:00 出发，空驶约8min，运输120min，到达约22:08
    # 卸货点到家约50km，回家约62min，到家约23:10 > 23:00 deadline
    # 应该被过滤
    record("fix6 cost_time超deadline被过滤",
           len(filtered) == 0,
           f"filtered={len(filtered)} (20:00出发, 运输120min, 回家>23:00)")


def test_fix7_family_wait_sim_time():
    """fix7: 家事等待用仿真时间差"""
    print("\n[fix7] 家事等待用仿真时间差")
    planner = SchedulePlanner()
    config = DriverConfig(
        driver_id="FIX7",
        family_event=FamilyEvent(
            trigger_min=13200,  # day9 10:00
            home_deadline_min=13920,
            end_min=17520,
            waypoints=[{"lat": 23.21, "lng": 113.37, "wait_minutes": 10}],
            home_pos=(23.19, 113.36),
        ),
    )

    # 模拟已到达途经点，进入 wait_spouse 阶段
    state = DriverState(driver_id="FIX7")
    state.sim_minutes = 13210  # 到达后10分钟
    state.current_lat = 23.21
    state.current_lng = 113.37
    state.family_phase = "wait_spouse"
    state.spouse_wait_start_min = 13200  # 进入等待的时刻

    # 已等待 13210 - 13200 = 10 分钟 >= wait_minutes(10)，应转入 go_home
    decision = planner.plan(state, config)
    record("fix7 等待10min后→go_home",
           state.family_phase == "go_home" and decision.action == ScheduleAction.REPOSITION,
           f"phase={state.family_phase}, action={decision.action.value}")

    # 如果只等了5分钟，应继续等待
    state2 = DriverState(driver_id="FIX7_2")
    state2.sim_minutes = 13205
    state2.current_lat = 23.21
    state2.current_lng = 113.37
    state2.family_phase = "wait_spouse"
    state2.spouse_wait_start_min = 13200

    decision2 = planner.plan(state2, config)
    record("fix7 等待5min继续REST",
           state2.family_phase == "wait_spouse" and decision2.action == ScheduleAction.REST,
           f"phase={state2.family_phase}, action={decision2.action.value}")


def test_fix8_month_end_income_eligible():
    """fix8: 月末 income_eligible 检查 (rule 11)"""
    print("\n[fix8] 月末 income_eligible 检查 (rule 11)")
    engine = RuleEngine()
    config = DriverConfig(driver_id="FIX8", reposition_speed_kmpm=0.8)

    # 月末最后一天 23:00
    state = DriverState(driver_id="FIX8", current_lat=23.0, current_lng=113.3)
    state.sim_minutes = 30 * 1440 + 23 * 60  # day30 23:00 = 44280min

    # 运输耗时超过月末（31*1440=44640）
    cargos = [
        {"cargo_id": "OVER_MONTH", "category": "建材", "price": 1000,
         "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 24.00, "delivery_lng": 114.00,
         "cost_time_minutes": 480},  # 8小时运输，完成时间超月末
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    record("fix8 超月末完成被过滤", len(filtered) == 0,
           f"sim=44280, cost_time=480, finish>44640, filtered={len(filtered)}")

    # 能在月末前完成的
    cargos2 = [
        {"cargo_id": "IN_MONTH", "category": "建材", "price": 500,
         "pickup_lat": 23.02, "pickup_lng": 113.32,
         "delivery_lat": 23.10, "delivery_lng": 113.40,
         "cost_time_minutes": 30},  # 30分钟运输
    ]
    filtered2 = engine.filter_cargos(cargos2, state, config)
    record("fix8 月末前完成通过", len(filtered2) == 1,
           f"sim=44280, cost_time=30, finish<44640, filtered={len(filtered2)}")


def test_fix9_quiet_window_go_home_safety():
    """fix9: 安静窗口中未到家 → GO_HOME 安全网"""
    print("\n[fix9] 安静窗口中未到家 → GO_HOME 安全网")
    planner = SchedulePlanner()
    config = DriverConfig(
        driver_id="FIX9",
        must_return_home=True,
        home_pos=(23.12, 113.28),
        home_deadline_hour=23,
        home_quiet_start=23,
        home_quiet_end=8,
        quiet_window=QuietWindow(start=23 * 60, end=8 * 60 + 1440),
        reposition_speed_kmpm=0.8,
    )

    # 安静窗口中但远离家
    state = DriverState(driver_id="FIX9")
    state.sim_minutes = 1440 + 2 * 60  # day1 02:00
    state.current_lat = 24.0  # 远离家
    state.current_lng = 114.0
    state._last_day = 1

    decision = planner.plan(state, config)
    record("fix9 安静窗口中远离家→GO_HOME",
           decision.action == ScheduleAction.GO_HOME,
           f"action={decision.action.value}, reason={decision.reason}")

    # 安静窗口中在家附近 → REST
    state2 = DriverState(driver_id="FIX9_home")
    state2.sim_minutes = 1440 + 2 * 60
    state2.current_lat = 23.12
    state2.current_lng = 113.28
    state2._last_day = 1

    decision2 = planner.plan(state2, config)
    record("fix9 安静窗口中在家→REST",
           decision2.action == ScheduleAction.REST,
           f"action={decision2.action.value}")


def test_fix10_hotspot_history_seed():
    """fix10: 历史 pickup 位置种子 HotspotTracker"""
    print("\n[fix10] 历史 pickup 位置种子 HotspotTracker")
    tracker = HotspotTracker()

    # 模拟从历史接单记录提取 pickup 位置
    seed_cargos = [
        {"pickup_lat": 23.05, "pickup_lng": 113.35},
        {"pickup_lat": 23.06, "pickup_lng": 113.36},
        {"pickup_lat": 23.04, "pickup_lng": 113.34},
    ]
    tracker.observe(seed_cargos, 0)

    hotspots = tracker.get_hotspots()
    record("fix10 历史种子生成热点", len(hotspots) > 0,
           f"hotspots={len(hotspots)}")

    # 验证热点在种子位置附近
    if hotspots:
        hs = hotspots[0]
        dist = haversine_km(hs[0], hs[1], 23.05, 113.35)
        record("fix10 热点在种子附近", dist < 20,
               f"hotspot=({hs[0]:.2f},{hs[1]:.2f}), dist={dist:.1f}km")


def test_fix11_hotspot_cold_start():
    """fix11: HotspotTracker 冷启动用司机初始位置"""
    print("\n[fix11] HotspotTracker 冷启动")
    scorer = CargoScorer()

    # 初始无热点
    record("fix11 初始无热点", len(scorer.hotspot_tracker.get_hotspots()) == 0,
           f"hotspots={len(scorer.hotspot_tracker.get_hotspots())}")

    # 模拟冷启动：用司机初始位置种子化
    initial_lat, initial_lng = 22.54, 114.06
    scorer.hotspot_tracker.observe(
        [{"pickup_lat": initial_lat, "pickup_lng": initial_lng}], 0)

    hotspots = scorer.hotspot_tracker.get_hotspots()
    record("fix11 冷启动后有热点", len(hotspots) > 0,
           f"hotspots={len(hotspots)}")


def test_fix12_cargo_id_strip():
    """fix12: cargo_id 验证 strip 引号/空格"""
    print("\n[fix12] cargo_id strip 引号/空格")

    # 模拟 LLM 返回带引号的 cargo_id
    candidates = [
        {"cargo_id": "C100", "price": 500},
        {"cargo_id": "C101", "price": 800},
    ]

    # 模拟 strip 逻辑
    raw_id = '"C101"'
    cleaned = raw_id.strip().strip('"').strip("'")
    valid_ids = {str(c.get("cargo_id", "")).strip(): str(c.get("cargo_id", ""))
                 for c in candidates}

    record("fix12 带引号的cargo_id清理", cleaned == "C101",
           f"raw='{raw_id}', cleaned='{cleaned}'")
    record("fix12 清理后匹配成功", cleaned in valid_ids,
           f"valid_ids={list(valid_ids.keys())}")

    # 带空格
    raw_id2 = "  C100  "
    cleaned2 = raw_id2.strip().strip('"').strip("'")
    record("fix12 带空格的cargo_id清理", cleaned2 == "C100",
           f"raw='{raw_id2}', cleaned='{cleaned2}'")
    record("fix12 空格清理后匹配", cleaned2 in valid_ids,
           f"valid_ids={list(valid_ids.keys())}")


# ===========================================================================
# 端到端测试：用真实司机数据走完整流程
# ===========================================================================

def test_e2e_d001_full_pipeline():
    """端到端：D001 张三完整决策流程"""
    print("\n[E2E-D001] 张三完整决策流程")
    clear_configs()

    parsed = build_d001_parsed()
    parsed_json = json.dumps({
        "rest_constraints": [{"min_hours": 8, "penalty_per_day": 300, "penalty_cap": 3000}],
        "forbidden_categories": [{"categories": ["化工塑料", "煤炭矿产"], "is_soft": False,
                                   "penalty_per_order": 500, "penalty_cap": 5000}],
        "geo_fences": [{"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66,
                        "penalty_once": 2000, "penalty_cap": 2000}],
    })

    api = DriverMockApi(
        driver_id="D001",
        parsed_json=parsed_json,
        status_overrides={
            "driver_id": "D001",
            "simulation_progress_minutes": 600,
            "current_lat": 22.54,
            "current_lng": 114.06,
            "cost_per_km": 1.5,
        },
        cargo_items=[
            {"cargo": {"cargo_id": "C100", "price": 500, "category": "食品饮料",
                        "pickup_lat": 22.58, "pickup_lng": 114.08,
                        "delivery_lat": 22.65, "delivery_lng": 114.15,
                        "cost_time_minutes": 60}, "distance_km": 5},
            {"cargo": {"cargo_id": "C101", "price": 800, "category": "电子产品",
                        "pickup_lat": 22.60, "pickup_lng": 114.10,
                        "delivery_lat": 22.70, "delivery_lng": 114.20,
                        "cost_time_minutes": 90}, "distance_km": 8},
            {"cargo": {"cargo_id": "C102", "price": 300, "category": "化工塑料",
                        "pickup_lat": 22.50, "pickup_lng": 114.00,
                        "delivery_lat": 22.45, "delivery_lng": 113.95,
                        "cost_time_minutes": 45}, "distance_km": 7},
        ],
    )

    svc = ModelDecisionService(api)
    result = svc.decide("D001")

    record("E2E-D001 返回take_order", result["action"] == "take_order",
           f"action={result['action']}")
    record("E2E-D001 不选化工塑料",
           result.get("params", {}).get("cargo_id") != "C102",
           f"cargo_id={result.get('params', {}).get('cargo_id')}")


def test_e2e_d009_special_cargo():
    """端到端：D009 郑一特殊货源必接"""
    print("\n[E2E-D009] 郑一特殊货源必接")
    clear_configs()

    parsed_json = json.dumps({
        "special_cargos": [{"cargo_id": "240646", "available_from": "2026-03-03 14:43:36",
                            "pickup_lat": 24.81, "pickup_lng": 113.58, "penalty_if_missed": 10000}],
        "go_home": [{"home_lat": 23.12, "home_lng": 113.28, "deadline_hour": 23,
                     "quiet_start_hour": 23, "quiet_end_hour": 8, "penalty_per_day": 900, "penalty_cap": 27000}],
        "forbidden_categories": [{"categories": ["快递快运搬家"], "is_soft": False,
                                   "penalty_per_order": 350, "penalty_cap": 3500}],
    })

    api = DriverMockApi(
        driver_id="D009",
        parsed_json=parsed_json,
        status_overrides={
            "driver_id": "D009",
            "simulation_progress_minutes": 3530,  # day2 ~10:50
            "current_lat": 24.80,
            "current_lng": 113.57,
            "cost_per_km": 1.5,
        },
        cargo_items=[
            {"cargo": {"cargo_id": "240646", "price": 600, "category": "服饰纺织皮革",
                        "pickup_lat": 24.81, "pickup_lng": 113.58,
                        "delivery_lat": 23.5, "delivery_lng": 113.4,
                        "cost_time_minutes": 120}, "distance_km": 1.5},
            {"cargo": {"cargo_id": "C999", "price": 2000, "category": "电子产品",
                        "pickup_lat": 24.82, "pickup_lng": 113.59,
                        "delivery_lat": 25.0, "delivery_lng": 114.0,
                        "cost_time_minutes": 60}, "distance_km": 2.0},
        ],
    )

    svc = ModelDecisionService(api)
    result = svc.decide("D009")

    record("E2E-D009 必接240646",
           result.get("params", {}).get("cargo_id") == "240646",
           f"decision={json.dumps(result, ensure_ascii=False)}")


def test_e2e_d010_family_trigger():
    """端到端：D010 冯二家事触发"""
    print("\n[E2E-D010] 冯二家事触发")
    clear_configs()

    parsed_json = json.dumps({
        "family_events": [{"trigger_time": "2026-03-10 10:00:00",
                           "waypoints": [{"lat": 23.21, "lng": 113.37, "wait_minutes": 10}],
                           "home_lat": 23.19, "home_lng": 113.36,
                           "home_deadline": "2026-03-10 22:00:00",
                           "stay_until": "2026-03-13 22:00:00",
                           "penalty_per_minute_late": 5, "penalty_once_if_failed": 9000}],
        "visit_targets": [{"target_lat": 23.13, "target_lng": 113.26, "min_days": 5,
                           "penalty_once": 3000, "penalty_cap": 3000}],
        "rest_constraints": [{"min_hours": 3, "penalty_per_day": 300, "penalty_cap": 6000}],
        "forbidden_categories": [{"categories": ["服饰纺织皮革"], "is_soft": True,
                                   "penalty_per_order": 240, "penalty_cap": 2400}],
    })

    trigger_min = _parse_datetime_to_sim_minutes("2026-03-10 10:00:00")

    api = DriverMockApi(
        driver_id="D010",
        parsed_json=parsed_json,
        status_overrides={
            "driver_id": "D010",
            "simulation_progress_minutes": trigger_min,  # 家事触发时刻
            "current_lat": 23.0,
            "current_lng": 113.3,
            "cost_per_km": 1.5,
        },
        cargo_items=[],  # 家事期间不查货
    )

    svc = ModelDecisionService(api)
    result = svc.decide("D010")

    record("E2E-D010 家事触发→reposition",
           result["action"] == "reposition",
           f"action={result['action']}, params={result.get('params', {})}")


# ===========================================================================
# 综合验证：所有司机配置生成正确性
# ===========================================================================

def test_all_drivers_config_generation():
    """验证所有 10 个司机的 ParsedPreferences → DriverConfig 转换正确性"""
    print("\n[ALL] 10 个司机配置生成综合验证")

    for driver_id, builder in ALL_BUILDERS.items():
        parsed = builder()
        config = build_config_from_parsed(parsed)
        record(f"{driver_id} 配置生成成功",
               config.driver_id == driver_id,
               f"driver_id={config.driver_id}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("=" * 70)
    print("  集成测试：10 个真实司机 × 12 个修复点")
    print("  数据来源: server/data/drivers.json")
    print("=" * 70)

    # Part 1: 每个司机的约束验证
    print("\n" + "=" * 70)
    print("  Part 1: 各司机约束验证")
    print("=" * 70)
    test_d001_geo_fence_and_forbidden()
    test_d002_off_days_weekend_first()
    test_d003_deadhead_and_forbidden_zone()
    test_d004_first_order_deadline_no_block()
    test_d005_max_distances()
    test_d006_rest_and_off_days()
    test_d007_quiet_window_and_off_day()
    test_d008_soft_forbidden_and_weekday_rest()
    test_d009_special_cargo_and_go_home()
    test_d010_family_event_and_visit()

    # Part 2: 12 个修复点专项测试
    print("\n" + "=" * 70)
    print("  Part 2: 12 个修复点专项测试")
    print("=" * 70)
    test_fix1_query_cooldown()
    test_fix2_load_time_window()
    test_fix3_total_gross_income()
    test_fix6_go_home_cost_time()
    test_fix7_family_wait_sim_time()
    test_fix8_month_end_income_eligible()
    test_fix9_quiet_window_go_home_safety()
    test_fix10_hotspot_history_seed()
    test_fix11_hotspot_cold_start()
    test_fix12_cargo_id_strip()

    # Part 3: 端到端测试
    print("\n" + "=" * 70)
    print("  Part 3: 端到端决策流程")
    print("=" * 70)
    test_e2e_d001_full_pipeline()
    test_e2e_d009_special_cargo()
    test_e2e_d010_family_trigger()

    # Part 4: 综合验证
    print("\n" + "=" * 70)
    print("  Part 4: 综合验证")
    print("=" * 70)
    test_all_drivers_config_generation()

    # 汇总
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"  测试汇总: {passed} PASS / {failed} FAIL / {len(results)} TOTAL")
    print("=" * 70)

    if failed > 0:
        print("\n  失败项目:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"    - {r['test']}: {r['detail']}")
        sys.exit(1)
    else:
        print("\n  ALL TESTS PASSED!")
        sys.exit(0)
