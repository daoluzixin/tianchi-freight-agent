"""Agent 决策流水线端到端测试脚本（v3 纯动态版本）。

测试目标：验证 StateTracker / SchedulePlanner / RuleEngine / CargoScorer / ModelDecisionService
在各典型场景下的行为正确性。

测试方式：
  - 子模块测试（test_02~12, 14~17）：直接构造 DriverConfig 对象传入子模块，
    不依赖全局注册表（验证算法逻辑本身）。
  - 端到端测试（test_07, test_13）：让 Mock API 返回偏好文本 + 对应 parser JSON，
    走完整的"偏好解析 → 配置生成 → 决策"流水线（验证泛化机制可用）。

运行方式：
    cd demo && python -m agent.tests.test_pipeline
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
)
from agent.core.state_tracker import StateTracker, DriverState, haversine_km
from agent.core.schedule_planner import SchedulePlanner, ScheduleAction
from agent.core.rule_engine import RuleEngine, FilteredCargo
from agent.scoring.cargo_scorer import CargoScorer
from agent.core.model_decision_service import ModelDecisionService


# ===========================================================================
# Mock API（端到端测试用，模拟 LLM 返回解析后的 JSON）
# ===========================================================================

# D001 偏好解析结果（LLM 返回的 JSON）
D001_PARSED_JSON = json.dumps({
    "rest_constraints": [{"min_hours": 8, "weekday_only": False, "penalty_per_day": 300}],
    "quiet_windows": [{"start_hour": 23, "start_minute": 0, "end_hour": 6, "end_minute": 0, "penalty_per_day": 200}],
    "forbidden_categories": [{"categories": ["化工塑料", "煤炭矿产"], "is_soft": False, "penalty_per_order": 500}],
    "geo_fences": [{"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66, "penalty_once": 2000}],
})

# D009 偏好解析结果
D009_PARSED_JSON = json.dumps({
    "go_home": [{"home_lat": 23.12, "home_lng": 113.28, "deadline_hour": 23,
                 "quiet_start_hour": 23, "quiet_end_hour": 8, "radius_km": 1.0, "penalty_per_day": 900}],
    "special_cargos": [{"cargo_id": "240646", "available_from": "2026-03-03 14:43:36",
                        "pickup_lat": 24.81, "pickup_lng": 113.58, "penalty_if_missed": 10000}],
    "quiet_windows": [{"start_hour": 23, "start_minute": 0, "end_hour": 8, "end_minute": 0, "penalty_per_day": 200}],
})


class MockApi:
    """默认 Mock：模拟 D001 场景（深圳 day0 10:00）。
    model_chat_completion 的第一次调用返回偏好解析 JSON，
    后续调用返回 cargo 选择 JSON。
    """

    def __init__(self, parsed_json: str = D001_PARSED_JSON):
        self._parsed_json = parsed_json
        self._call_count = 0

    def get_driver_status(self, driver_id):
        return {
            "driver_id": driver_id,
            "simulation_progress_minutes": 600,
            "current_lat": 22.55,
            "current_lng": 114.05,
            "cost_per_km": 1.5,
            "truck_length": 9.6,
            "completed_order_count": 2,
            "preferences": [
                {"content": "不接化工塑料和煤炭矿产", "penalty_amount": 500, "penalty_cap": 5000},
            ],
        }

    def query_cargo(self, driver_id, latitude, longitude):
        return {
            "items": [
                {
                    "cargo": {
                        "cargo_id": "C100",
                        "price": 500,
                        "category": "食品饮料",
                        "pickup_lat": 22.58,
                        "pickup_lng": 114.08,
                        "delivery_lat": 22.65,
                        "delivery_lng": 114.15,
                        "cost_time_minutes": 60,
                    },
                    "distance_km": 5.2,
                },
                {
                    "cargo": {
                        "cargo_id": "C101",
                        "price": 800,
                        "category": "电子产品",
                        "pickup_lat": 22.60,
                        "pickup_lng": 114.10,
                        "delivery_lat": 22.70,
                        "delivery_lng": 114.20,
                        "cost_time_minutes": 90,
                    },
                    "distance_km": 8.1,
                },
                {
                    "cargo": {
                        "cargo_id": "C102",
                        "price": 300,
                        "category": "化工塑料",
                        "pickup_lat": 22.50,
                        "pickup_lng": 114.00,
                        "delivery_lat": 22.45,
                        "delivery_lng": 113.95,
                        "cost_time_minutes": 45,
                    },
                    "distance_km": 7.5,
                },
            ]
        }

    def query_decision_history(self, driver_id, step):
        return {"records": [], "total_steps": 0, "returned_count": 0}

    def model_chat_completion(self, payload):
        self._call_count += 1
        # 第一次调用是偏好解析，后续是 cargo tiebreak
        if self._call_count == 1:
            return {
                "choices": [{"message": {"content": self._parsed_json}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 150, "total_tokens": 350},
            }
        return {
            "choices": [{"message": {"content": json.dumps({"cargo_id": "C101"})}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }


# ===========================================================================
# 测试用例
# ===========================================================================

results = []


def record(test_name: str, passed: bool, detail: str):
    status = "PASS" if passed else "FAIL"
    results.append({"test": test_name, "status": status, "detail": detail})
    print(f"  [{status}] {test_name}")
    if detail:
        print(f"         {detail}")


def test_01_build_config_from_parsed():
    """测试1：build_config_from_parsed 动态配置生成。"""
    print("\n[Test 01] 动态配置生成（build_config_from_parsed）")
    from agent.config.preference_parser import (
        ParsedPreferences, RestConstraint, QuietWindowConstraint,
        ForbiddenCategoryConstraint, MaxDistanceConstraint, GoHomeConstraint,
    )

    parsed = ParsedPreferences(
        driver_id="D_DYN",
        cost_per_km=2.0,
        rest_constraints=[RestConstraint(min_hours=8, weekday_only=False, penalty_per_day=300)],
        quiet_windows=[QuietWindowConstraint(start_hour=22, end_hour=6, penalty_per_day=200)],
        forbidden_categories=[ForbiddenCategoryConstraint(categories=["化工塑料"], is_soft=False, penalty_per_order=500)],
        max_distances=[MaxDistanceConstraint(constraint_type="haul", max_km=100, penalty_per_violation=100)],
        go_home=[GoHomeConstraint(home_lat=23.12, home_lng=113.28, deadline_hour=23,
                                  quiet_start_hour=23, quiet_end_hour=8, penalty_per_day=900)],
    )

    config = build_config_from_parsed(parsed)

    record("driver_id正确", config.driver_id == "D_DYN", f"got: {config.driver_id}")
    record("cost_per_km正确", config.cost_per_km == 2.0, f"got: {config.cost_per_km}")
    record("休息分钟数正确", config.min_continuous_rest_minutes == 480,
           f"got: {config.min_continuous_rest_minutes}")
    record("安静窗口正确", config.quiet_window is not None and config.quiet_window.start == 22*60,
           f"start={config.quiet_window.start if config.quiet_window else None}")
    record("禁止品类正确", "化工塑料" in config.forbidden_categories,
           f"got: {config.forbidden_categories}")
    record("最大运距正确", config.max_haul_km == 100, f"got: {config.max_haul_km}")
    record("回家约束正确", config.must_return_home and config.home_pos == (23.12, 113.28),
           f"home_pos={config.home_pos}")


def test_02_forbidden_category():
    """测试2：禁止品类过滤。"""
    print("\n[Test 02] 禁止品类过滤")
    engine = RuleEngine()
    tracker = StateTracker()
    state = tracker.get_state("T02")
    state.current_lat = 22.55
    state.current_lng = 114.05

    # 直接构造配置，不依赖注册表
    config = DriverConfig(
        driver_id="T02",
        forbidden_categories={"化工塑料", "煤炭矿产"},
        geo_fence={"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66},
    )

    cargos = [
        {"cargo_id": "A1", "category": "化工塑料", "pickup_lat": 22.6, "pickup_lng": 114.1,
         "delivery_lat": 22.7, "delivery_lng": 114.2, "price": 1000},
        {"cargo_id": "A2", "category": "煤炭矿产", "pickup_lat": 22.6, "pickup_lng": 114.1,
         "delivery_lat": 22.7, "delivery_lng": 114.2, "price": 900},
        {"cargo_id": "A3", "category": "食品饮料", "pickup_lat": 22.6, "pickup_lng": 114.1,
         "delivery_lat": 22.7, "delivery_lng": 114.2, "price": 500},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    passed_ids = [f.cargo["cargo_id"] for f in filtered]

    record("化工塑料被过滤", "A1" not in passed_ids, f"通过列表: {passed_ids}")
    record("煤炭矿产被过滤", "A2" not in passed_ids, f"通过列表: {passed_ids}")
    record("食品饮料通过", "A3" in passed_ids, f"通过列表: {passed_ids}")


def test_03_geo_fence():
    """测试3：地理围栏。"""
    print("\n[Test 03] 地理围栏")
    engine = RuleEngine()
    tracker = StateTracker()
    state = tracker.get_state("T03")
    state.current_lat = 22.55
    state.current_lng = 114.05

    config = DriverConfig(
        driver_id="T03",
        geo_fence={"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66},
    )

    cargos = [
        {"cargo_id": "IN", "category": "电子产品", "pickup_lat": 22.6, "pickup_lng": 114.1,
         "delivery_lat": 22.7, "delivery_lng": 114.2, "price": 600},
        {"cargo_id": "OUT", "category": "电子产品", "pickup_lat": 22.6, "pickup_lng": 114.1,
         "delivery_lat": 23.13, "delivery_lng": 113.26, "price": 1200},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    passed_ids = [f.cargo["cargo_id"] for f in filtered]

    record("围栏内货源通过", "IN" in passed_ids, "")
    record("围栏外货源被过滤", "OUT" not in passed_ids,
           "卸货点(23.13,113.26)超出围栏")


def test_04_forbidden_zone():
    """测试4：禁止区域。"""
    print("\n[Test 04] 禁止区域")
    engine = RuleEngine()
    tracker = StateTracker()
    state = tracker.get_state("T04")
    state.current_lat = 23.0
    state.current_lng = 113.3

    config = DriverConfig(
        driver_id="T04",
        forbidden_zone={"center": (23.30, 113.52), "radius_km": 20.0},
        max_monthly_deadhead_km=100.0,
    )

    cargos = [
        {"cargo_id": "ZONE", "category": "建材", "pickup_lat": 23.0, "pickup_lng": 113.3,
         "delivery_lat": 23.30, "delivery_lng": 113.52, "price": 700},
        {"cargo_id": "SAFE", "category": "建材", "pickup_lat": 23.0, "pickup_lng": 113.3,
         "delivery_lat": 22.8, "delivery_lng": 113.1, "price": 500},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    passed_ids = [f.cargo["cargo_id"] for f in filtered]

    record("禁区内卸货被过滤", "ZONE" not in passed_ids,
           "卸货点(23.30,113.52)在禁区中心")
    record("禁区外卸货通过", "SAFE" in passed_ids, "")


def test_05_max_daily_orders():
    """测试5：每日最大接单数。"""
    print("\n[Test 05] 每日最大接单数")
    engine = RuleEngine()
    tracker = StateTracker()
    state = tracker.get_state("T05")
    state.current_lat = 23.0
    state.current_lng = 113.3
    state.today_order_count = 3

    config = DriverConfig(driver_id="T05", max_daily_orders=3)

    cargos = [
        {"cargo_id": "OVER", "category": "建材", "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 23.1, "delivery_lng": 113.4, "price": 600},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    record("达到3单上限后过滤", len(filtered) == 0,
           f"today_order_count=3, filtered={len(filtered)}")

    state.today_order_count = 2
    filtered2 = engine.filter_cargos(cargos, state, config)
    record("未达上限时通过", len(filtered2) == 1,
           f"today_order_count=2, filtered={len(filtered2)}")


def test_06_max_haul_km():
    """测试6：最大运距限制。"""
    print("\n[Test 06] 最大运距限制")
    engine = RuleEngine()
    tracker = StateTracker()
    state = tracker.get_state("T06")
    state.current_lat = 23.0
    state.current_lng = 113.3

    config = DriverConfig(driver_id="T06", max_haul_km=100.0)

    cargos = [
        {"cargo_id": "FAR", "category": "建材", "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 24.3, "delivery_lng": 113.8, "price": 1500},
        {"cargo_id": "NEAR", "category": "建材", "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 23.2, "delivery_lng": 113.5, "price": 400},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    passed_ids = [f.cargo["cargo_id"] for f in filtered]

    far_haul = haversine_km(23.05, 113.35, 24.3, 113.8)
    near_haul = haversine_km(23.05, 113.35, 23.2, 113.5)
    record("超100km运距被过滤", "FAR" not in passed_ids,
           f"运距={far_haul:.1f}km > 100km")
    record("100km内运距通过", "NEAR" in passed_ids,
           f"运距={near_haul:.1f}km <= 100km")


def test_07_special_cargo_e2e():
    """测试7：端到端 - 特殊货源必接（走完整 parser 链路）。"""
    print("\n[Test 07] 端到端特殊货源必接")

    class D009Api(MockApi):
        def __init__(self):
            super().__init__(parsed_json=D009_PARSED_JSON)

        def get_driver_status(self, driver_id):
            return {
                "driver_id": driver_id,
                "simulation_progress_minutes": 3530,
                "current_lat": 24.80,
                "current_lng": 113.57,
                "cost_per_km": 1.5,
                "truck_length": 9.6,
                "completed_order_count": 5,
                "preferences": [
                    {"content": "每天23点前回家(23.12,113.28)，23-08不活动", "penalty_amount": 900, "penalty_cap": 27000},
                    {"content": "必接货源240646", "penalty_amount": 10000, "penalty_cap": None},
                ],
            }

        def query_cargo(self, driver_id, latitude, longitude):
            return {
                "items": [
                    {
                        "cargo": {
                            "cargo_id": "240646",
                            "price": 600,
                            "category": "建材",
                            "pickup_lat": 24.81,
                            "pickup_lng": 113.58,
                            "delivery_lat": 23.5,
                            "delivery_lng": 113.4,
                            "cost_time_minutes": 120,
                        },
                        "distance_km": 1.5,
                    },
                    {
                        "cargo": {
                            "cargo_id": "C999",
                            "price": 2000,
                            "category": "电子产品",
                            "pickup_lat": 24.82,
                            "pickup_lng": 113.59,
                            "delivery_lat": 25.0,
                            "delivery_lng": 114.0,
                            "cost_time_minutes": 60,
                        },
                        "distance_km": 2.0,
                    },
                ]
            }

    clear_configs()  # 确保无预注册
    svc = ModelDecisionService(D009Api())
    result = svc.decide("D009")
    record("必接熟货240646", result["params"].get("cargo_id") == "240646",
           f"decision={json.dumps(result, ensure_ascii=False)}")


def test_08_go_home_at_night():
    """测试8：夜间远离家时触发回家（子模块测试）。"""
    print("\n[Test 08] 夜间回家")
    tracker = StateTracker()
    planner = SchedulePlanner()

    config = DriverConfig(
        driver_id="T08",
        must_return_home=True,
        home_pos=(23.12, 113.28),
        home_deadline_hour=23,
        home_quiet_start=23,
        home_quiet_end=8,
        quiet_window=QuietWindow(start=23*60, end=8*60+1440),
    )

    # 远离家
    state = tracker.get_state("T08_far")
    state.sim_minutes = 1440 * 2 + 23 * 60 + 30  # day2 23:30
    state.current_lat = 23.5
    state.current_lng = 113.6
    decision = planner.plan(state, config)
    record("远离家时GO_HOME", decision.action == ScheduleAction.GO_HOME,
           f"action={decision.action.value}, reason={decision.reason}")

    # 在家
    state2 = tracker.get_state("T08_home")
    state2.sim_minutes = 1440 * 2 + 23 * 60 + 30
    state2.current_lat = 23.12
    state2.current_lng = 113.28
    decision2 = planner.plan(state2, config)
    record("在家时REST", decision2.action == ScheduleAction.REST,
           f"action={decision2.action.value}, reason={decision2.reason}")


def test_09_daily_rest():
    """测试9：每日连续休息约束。"""
    print("\n[Test 09] 每日休息约束")
    tracker = StateTracker()
    planner = SchedulePlanner()

    config = DriverConfig(
        driver_id="T09",
        min_continuous_rest_minutes=480,
        suggested_rest_start_hour=22,
        suggested_rest_end_hour=6,
    )

    state = tracker.get_state("T09")
    state.sim_minutes = 180  # day0 03:00（在休息时段内）
    state.current_lat = 22.55
    state.current_lng = 114.05
    state.longest_rest_today = 0
    state.current_rest_streak = 0

    decision = planner.plan(state, config)
    record("休息未满足时强制REST", decision.action == ScheduleAction.REST,
           f"action={decision.action.value}, 已休息0/480min")

    state.longest_rest_today = 480
    state.sim_minutes = 600  # day0 10:00
    decision2 = planner.plan(state, config)
    record("休息满足后WORK", decision2.action == ScheduleAction.WORK,
           f"action={decision2.action.value}")


def test_10_family_event():
    """测试10：家事状态机触发。"""
    print("\n[Test 10] 家事状态机")
    tracker = StateTracker()
    planner = SchedulePlanner()

    config = DriverConfig(
        driver_id="T10",
        family_event=FamilyEvent(
            trigger_min=13200,
            home_deadline_min=13920,
            end_min=17520,
            waypoints=[{"lat": 23.21, "lng": 113.37, "wait_minutes": 10}],
            home_pos=(23.19, 113.36),
            penalty_per_minute=5.0,
            penalty_once_if_failed=9000.0,
        ),
    )

    state = tracker.get_state("T10")
    state.sim_minutes = 13200
    state.current_lat = 23.0
    state.current_lng = 113.3
    state.family_phase = "idle"

    decision = planner.plan(state, config)
    record("家事触发→go_spouse",
           decision.action == ScheduleAction.REPOSITION and state.family_phase == "go_spouse",
           f"action={decision.action.value}, phase={state.family_phase}")


def test_11_deadhead_budget():
    """测试11：月度空驶配额。"""
    print("\n[Test 11] 空驶配额限制")
    engine = RuleEngine()
    tracker = StateTracker()
    state = tracker.get_state("T11")
    state.current_lat = 23.0
    state.current_lng = 113.3
    state.total_deadhead_km = 90.0

    config = DriverConfig(
        driver_id="T11",
        forbidden_zone={"center": (23.30, 113.52), "radius_km": 20.0},
        max_monthly_deadhead_km=100.0,
    )

    cargos = [
        {"cargo_id": "OVER_BUDGET", "category": "建材", "pickup_lat": 23.1, "pickup_lng": 113.4,
         "delivery_lat": 22.8, "delivery_lng": 113.1, "price": 500},
        {"cargo_id": "IN_BUDGET", "category": "建材", "pickup_lat": 23.02, "pickup_lng": 113.32,
         "delivery_lat": 22.8, "delivery_lng": 113.1, "price": 400},
    ]
    filtered = engine.filter_cargos(cargos, state, config)
    passed_ids = [f.cargo["cargo_id"] for f in filtered]

    record("超空驶预算被过滤", "OVER_BUDGET" not in passed_ids,
           "已用90km, 该单空驶>10km")
    record("空驶预算内通过", "IN_BUDGET" in passed_ids,
           "该单空驶约5km, 总计95km")


def test_12_cargo_scorer_ranking():
    """测试12：评分器排序验证。"""
    print("\n[Test 12] 货源评分排序")
    scorer = CargoScorer()
    tracker = StateTracker()
    state = tracker.get_state("T12")
    state.current_lat = 23.0
    state.current_lng = 113.3

    config = DriverConfig(driver_id="T12", max_haul_km=200.0)

    candidates = [
        FilteredCargo(
            cargo={"cargo_id": "HIGH", "price": 800, "delivery_lat": 23.13, "delivery_lng": 113.26},
            pickup_km=5.0, haul_km=50.0),
        FilteredCargo(
            cargo={"cargo_id": "MED", "price": 400, "delivery_lat": 24.0, "delivery_lng": 114.0},
            pickup_km=2.0, haul_km=80.0),
        FilteredCargo(
            cargo={"cargo_id": "LOW", "price": 300, "delivery_lat": 22.55, "delivery_lng": 114.06},
            pickup_km=30.0, haul_km=20.0),
    ]
    scored = scorer.score_and_rank(candidates, state, config)
    ranking = [s.cargo["cargo_id"] for s in scored]

    record("高价短距排第一", ranking[0] == "HIGH",
           f"排序: {ranking}, 分数: {[f'{s.score:.1f}' for s in scored]}")
    record("高空驶比排最后", ranking[-1] == "LOW",
           "LOW 空驶比=30/20=1.5, 罚分严重")


def test_13_end_to_end_d001():
    """测试13：端到端 D001 完整决策流程（走 parser 链路）。"""
    print("\n[Test 13] 端到端 D001 决策")
    clear_configs()
    svc = ModelDecisionService(MockApi(parsed_json=D001_PARSED_JSON))
    result = svc.decide("D001")

    record("返回take_order", result["action"] == "take_order",
           f"action={result['action']}")
    record("选择C101（最高分合规货源）",
           result["params"].get("cargo_id") == "C101",
           f"cargo_id={result['params'].get('cargo_id')}, "
           f"C102被禁(化工塑料), C101(800元)>C100(500元)")


def test_14_quiet_window():
    """测试14：安静窗口验证。"""
    print("\n[Test 14] 安静窗口验证")
    planner = SchedulePlanner()
    tracker = StateTracker()

    config = DriverConfig(
        driver_id="T14",
        quiet_window=QuietWindow(start=23*60, end=6*60+1440),  # 23:00-06:00 跨天
        suggested_rest_start_hour=23,
        suggested_rest_end_hour=6,
    )

    state = tracker.get_state("T14_quiet")
    state.sim_minutes = 1440 + 2 * 60  # day1 02:00
    state.current_lat = 23.0
    state.current_lng = 113.3
    decision = planner.plan(state, config)
    record("02:00 在安静窗口内→REST",
           decision.action == ScheduleAction.REST,
           f"23:00-06:00窗口, action={decision.action.value}")

    state2 = tracker.get_state("T14_work")
    state2.sim_minutes = 1440 + 10 * 60  # day1 10:00
    state2.current_lat = 23.0
    state2.current_lng = 113.3
    decision2 = planner.plan(state2, config)
    record("10:00 正常工作",
           decision2.action == ScheduleAction.WORK,
           f"action={decision2.action.value}")


def test_15_soft_constraint():
    """测试15：软约束品类标记。"""
    print("\n[Test 15] 软约束品类标记")
    engine = RuleEngine()
    tracker = StateTracker()
    state = tracker.get_state("T15")
    state.current_lat = 23.0
    state.current_lng = 113.3

    config = DriverConfig(
        driver_id="T15",
        soft_forbidden_categories={"服饰纺织皮革"},
    )

    cargos = [
        {"cargo_id": "SOFT", "category": "服饰纺织皮革", "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 23.1, "delivery_lng": 113.4, "price": 500},
        {"cargo_id": "CLEAN", "category": "建材", "pickup_lat": 23.05, "pickup_lng": 113.35,
         "delivery_lat": 23.1, "delivery_lng": 113.4, "price": 500},
    ]
    filtered = engine.filter_cargos(cargos, state, config)

    soft_item = next((f for f in filtered if f.cargo["cargo_id"] == "SOFT"), None)
    clean_item = next((f for f in filtered if f.cargo["cargo_id"] == "CLEAN"), None)

    record("软禁止品类不被过滤", soft_item is not None, "服饰纺织皮革通过但标记")
    record("软禁止品类标记violated", soft_item is not None and soft_item.is_soft_violated,
           f"is_soft_violated={soft_item.is_soft_violated if soft_item else 'N/A'}")
    record("普通品类无违反标记", clean_item is not None and not clean_item.is_soft_violated, "")


def test_16_preference_parser_json():
    """测试16：偏好解析器 JSON 解析正确性。"""
    print("\n[Test 16] 偏好解析器 JSON 解析")
    from agent.config.preference_parser import parse_llm_response

    driver_status = {
        "driver_id": "D_PARSE",
        "cost_per_km": 1.8,
        "current_lat": 22.55,
        "current_lng": 114.05,
        "preferences": [{"content": "不接化工", "penalty_amount": 500}],
    }

    llm_output = json.dumps({
        "rest_constraints": [{"min_hours": 6, "weekday_only": True, "penalty_per_day": 200}],
        "forbidden_categories": [{"categories": ["化工塑料", "煤炭矿产"], "is_soft": False, "penalty_per_order": 500}],
        "max_orders": [{"max_per_day": 4, "penalty_per_extra": 300}],
        "geo_fences": [{"lat_min": 22.0, "lat_max": 23.0, "lng_min": 113.5, "lng_max": 114.5, "penalty_once": 2000}],
    })

    result = parse_llm_response(driver_status, llm_output)

    record("解析rest_constraints",
           len(result.rest_constraints) == 1 and result.rest_constraints[0].min_hours == 6,
           f"got {len(result.rest_constraints)} items")
    record("解析forbidden_categories",
           len(result.forbidden_categories) == 1 and "化工塑料" in result.forbidden_categories[0].categories,
           f"categories={result.forbidden_categories[0].categories if result.forbidden_categories else []}")
    record("解析max_orders",
           len(result.max_orders) == 1 and result.max_orders[0].max_per_day == 4,
           f"max_per_day={result.max_orders[0].max_per_day if result.max_orders else None}")
    record("解析geo_fences",
           len(result.geo_fences) == 1 and result.geo_fences[0].lat_min == 22.0,
           f"lat_min={result.geo_fences[0].lat_min if result.geo_fences else None}")


def test_17_unknown_driver_default():
    """测试17：未知司机 - 无约束的默认配置。"""
    print("\n[Test 17] 未知司机默认配置")

    config = DriverConfig(driver_id="D_UNKNOWN")
    record("默认无禁止品类", len(config.forbidden_categories) == 0, "")
    record("默认无地理围栏", config.geo_fence is None, "")
    record("默认无回家约束", config.must_return_home is False, "")
    record("默认无运距限制", config.max_haul_km is None, "")
    record("默认无安静窗口", config.quiet_window is None, "")


def test_18_token_budget_manager():
    """测试18：Token 预算管理器。"""
    print("\n[Test 18] Token 预算管理器")
    from agent.strategy.token_budget import TokenBudgetManager

    budget = TokenBudgetManager(total_budget=4_000_000)

    record("初始预算4M", budget.total_remaining == 4_000_000,
           f"remaining={budget.total_remaining}")

    budget.record_usage("parse", 4000)
    record("消耗后余额减少", budget.total_remaining == 3_996_000,
           f"remaining={budget.total_remaining}")

    record("可支出判断正常", budget.can_spend("decision", 1500) is True, "")
    record("超额支出判断正常", budget.can_spend("decision", 999_999_999) is False, "")

    # 测试 should_use_llm_for_decision
    should = budget.should_use_llm_for_decision({
        "cargo_count": 5,
        "score_gap": 3.0,
        "best_score": 10,
        "has_custom_constraints": False,
        "current_day": 15,
        "steps_without_order": 0,
    })
    record("多货源分差小触发LLM", should is True, "cargo_count=5, gap=3.0")

    should_not = budget.should_use_llm_for_decision({
        "cargo_count": 1,
        "score_gap": 999,
        "best_score": 100,
        "has_custom_constraints": False,
        "current_day": 15,
        "steps_without_order": 0,
    })
    record("单货源高分不触发LLM", should_not is False, "cargo_count=1, gap=999")


def test_19_strategy_advisor_params():
    """测试19：策略顾问参数管理。"""
    print("\n[Test 19] 策略顾问参数")
    from agent.strategy.strategy_advisor import StrategyAdvisor, StrategyParams

    advisor = StrategyAdvisor()
    params = advisor.get_params("D_TEST")

    record("默认激进度0.5", params.aggression == 0.5,
           f"aggression={params.aggression}")
    record("默认等待乘数0.6", params.wait_value_multiplier == 0.6,
           f"wait_value_multiplier={params.wait_value_multiplier}")

    # 测试 fallback 调整
    advisor._fallback_day_adjust(params, 28)
    record("月末冲刺aggression=0.8", params.aggression == 0.8,
           f"day=28, aggression={params.aggression}")
    record("月末max_wait=15", params.max_wait_minutes == 15,
           f"max_wait_minutes={params.max_wait_minutes}")

    # 月初
    advisor._fallback_day_adjust(params, 2)
    record("月初探索aggression=0.4", params.aggression == 0.4,
           f"day=2, aggression={params.aggression}")


def test_20_smart_wait_timing():
    """测试20：智能等待时段感知。"""
    print("\n[Test 20] 智能等待时段感知")
    from agent.strategy.strategy_advisor import StrategyParams

    clear_configs()
    svc = ModelDecisionService(MockApi())

    params = StrategyParams(
        max_wait_minutes=30,
        min_wait_minutes=10,
        aggression=0.5,
        peak_hours=[(8, 11), (14, 18)],
    )

    tracker = StateTracker()
    # 高峰时段 10:00
    state_peak = tracker.get_state("T20_peak")
    state_peak.sim_minutes = 600  # 10:00
    wait_peak = svc._smart_wait(state_peak, params)

    # 低谷时段 03:00
    state_off = tracker.get_state("T20_off")
    state_off.sim_minutes = 180  # 03:00
    wait_off = svc._smart_wait(state_off, params)

    record("高峰等待时间短", wait_peak < wait_off,
           f"peak_wait={wait_peak}min, offpeak_wait={wait_off}min")
    record("高峰等待≥最小值", wait_peak >= params.min_wait_minutes,
           f"wait={wait_peak}, min={params.min_wait_minutes}")
    record("低谷等待不超60", wait_off <= 60,
           f"wait={wait_off}")


def test_21_token_budget_daily_review():
    """测试21：Token 预算的每日回顾控制。"""
    print("\n[Test 21] Token 预算每日回顾控制")
    from agent.strategy.token_budget import TokenBudgetManager

    budget = TokenBudgetManager(total_budget=4_000_000)

    record("day0可做回顾", budget.should_do_daily_review(0, -1) is True, "")
    record("同一天不重复回顾", budget.should_do_daily_review(0, 0) is False, "")
    record("新一天可以回顾", budget.should_do_daily_review(1, 0) is True, "")

    # 消耗所有 daily_review 预算
    for _ in range(40):
        budget.record_usage("daily_review", 3000)
    # daily_review 预算 100K，用了 120K → 超了，但总预算还有 → 仍可以
    can = budget.should_do_daily_review(5, 4)
    record("总预算充足仍可回顾", can is True,
           f"daily_used={budget._categories['daily_review'].used}")


def test_22_e2e_with_llm_enhance():
    """测试22：端到端 v4 - LLM 增强决策流程验证。"""
    print("\n[Test 22] 端到端 v4 LLM 增强决策")

    # 构造一个会触发 LLM enhance 的场景：多货源、分差小
    class EnhanceApi(MockApi):
        def __init__(self):
            super().__init__(parsed_json=D001_PARSED_JSON)
            self._call_count = 0

        def query_cargo(self, driver_id, latitude, longitude):
            """返回多个分差很小的货源，触发 LLM 增强。"""
            return {
                "items": [
                    {"cargo": {"cargo_id": "E1", "price": 500, "category": "食品饮料",
                               "pickup_lat": 22.58, "pickup_lng": 114.08,
                               "delivery_lat": 22.65, "delivery_lng": 114.15}, "distance_km": 5},
                    {"cargo": {"cargo_id": "E2", "price": 505, "category": "电子产品",
                               "pickup_lat": 22.57, "pickup_lng": 114.07,
                               "delivery_lat": 22.64, "delivery_lng": 114.14}, "distance_km": 4.8},
                    {"cargo": {"cargo_id": "E3", "price": 498, "category": "建材",
                               "pickup_lat": 22.59, "pickup_lng": 114.09,
                               "delivery_lat": 22.66, "delivery_lng": 114.16}, "distance_km": 5.5},
                ]
            }

        def model_chat_completion(self, payload):
            self._call_count += 1
            if self._call_count == 1:
                # 偏好解析
                return {"choices": [{"message": {"content": self._parsed_json}}],
                        "usage": {"total_tokens": 350}}
            # 后续调用（daily_review / enhance_decision）都返回一个选择
            return {"choices": [{"message": {"content": json.dumps({
                "action": "take_order", "cargo_id": "E2",
                "confidence": 0.8, "reasoning": "E2性价比最优"
            })}}], "usage": {"total_tokens": 150}}

    clear_configs()
    svc = ModelDecisionService(EnhanceApi())
    result = svc.decide("D_ENH")

    record("v4返回有效决策", result["action"] == "take_order",
           f"action={result['action']}")
    record("v4选择了某个有效货源",
           result["params"].get("cargo_id") in ("E1", "E2", "E3"),
           f"cargo_id={result['params'].get('cargo_id')}")

    # 验证 token budget 有消耗记录
    record("token已消耗", svc._budget.total_used > 0,
           f"total_used={svc._budget.total_used}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("=" * 70)
    print("  Agent 决策流水线 - 完整测试（v4 LLM增强版，零硬编码）")
    print("  测试时间: 运行时生成")
    print("=" * 70)

    test_01_build_config_from_parsed()
    test_02_forbidden_category()
    test_03_geo_fence()
    test_04_forbidden_zone()
    test_05_max_daily_orders()
    test_06_max_haul_km()
    test_07_special_cargo_e2e()
    test_08_go_home_at_night()
    test_09_daily_rest()
    test_10_family_event()
    test_11_deadhead_budget()
    test_12_cargo_scorer_ranking()
    test_13_end_to_end_d001()
    test_14_quiet_window()
    test_15_soft_constraint()
    test_16_preference_parser_json()
    test_17_unknown_driver_default()
    test_18_token_budget_manager()
    test_19_strategy_advisor_params()
    test_20_smart_wait_timing()
    test_21_token_budget_daily_review()
    test_22_e2e_with_llm_enhance()

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
