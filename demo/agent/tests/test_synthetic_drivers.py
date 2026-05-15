"""飞轮合成数据集测试：验证 rule_based_parser 对 L1/L2/L3 层合成司机偏好的解析能力。

测试目标：
  L1（措辞变体）  → 期望 100% 识别率（0 custom）
  L2（极端组合）  → 期望 100% 识别率（0 custom）
  L3（对抗探针）  → 允许部分进入 custom（新约束类型），但已知类型应正确识别

运行方式：
  cd demo && python -m pytest agent/tests/test_synthetic_drivers.py -v
  或：
  cd demo && python agent/tests/test_synthetic_drivers.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import fields

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.config.rule_based_parser import rule_based_parse
from agent.config.preference_parser import ParsedPreferences


# ===========================================================================
# 数据加载
# ===========================================================================

SYNTHETIC_DIR = Path(__file__).parent / "synthetic_drivers"


def load_layer(filename: str) -> list[dict]:
    filepath = SYNTHETIC_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


L1_DRIVERS = load_layer("l1_variants.json")
L2_DRIVERS = load_layer("l2_hard_combos.json")
L3_DRIVERS = load_layer("l3_adversarial.json")


# ===========================================================================
# 解析结果统计工具
# ===========================================================================

CONSTRAINT_FIELDS = [
    "rest_constraints", "quiet_windows", "forbidden_categories",
    "max_distances", "max_orders", "first_order_deadline",
    "off_days", "geo_fences", "forbidden_zones", "go_home",
    "special_cargos", "family_events", "visit_targets",
]


def count_recognized(result: ParsedPreferences) -> int:
    """统计被识别的约束数量（非 custom）。"""
    total = 0
    for field_name in CONSTRAINT_FIELDS:
        total += len(getattr(result, field_name, []))
    return total


def count_custom(result: ParsedPreferences) -> int:
    return len(result.custom)


def get_constraint_summary(result: ParsedPreferences) -> dict[str, int]:
    """获取各类型约束数量汇总。"""
    summary = {}
    for field_name in CONSTRAINT_FIELDS:
        count = len(getattr(result, field_name, []))
        if count > 0:
            summary[field_name] = count
    custom_count = len(result.custom)
    if custom_count > 0:
        summary["custom"] = custom_count
    return summary


# ===========================================================================
# L1 层测试：措辞变体 → 期望全部识别
# ===========================================================================

class TestL1Variants:
    """L1 层：相同语义不同措辞的变体，rule_based_parser 应全部识别。"""

    def test_all_recognized(self):
        """L1 所有偏好都应被识别为非 custom 类型。"""
        failures = []
        for driver in L1_DRIVERS:
            driver_id = driver["driver_id"]
            result = rule_based_parse(driver)
            custom_count = count_custom(result)
            if custom_count > 0:
                customs = [c.original_text[:50] for c in result.custom]
                failures.append(f"{driver_id}: {custom_count} unrecognized → {customs}")

        assert len(failures) == 0, (
            f"L1 层有 {len(failures)} 个司机存在未识别偏好:\n" +
            "\n".join(f"  - {f}" for f in failures)
        )

    def test_constraint_count_matches(self):
        """L1 识别的约束数量应等于偏好条数。"""
        for driver in L1_DRIVERS:
            driver_id = driver["driver_id"]
            total_prefs = len(driver["preferences"])
            result = rule_based_parse(driver)
            recognized = count_recognized(result)
            custom = count_custom(result)
            assert recognized + custom == total_prefs, (
                f"{driver_id}: prefs={total_prefs}, recognized={recognized}, custom={custom}"
            )

    def test_s001_rest_and_geofence(self):
        """S001 (D001变体): 8h休息 + 禁品类 + 地理围栏。"""
        driver = next(d for d in L1_DRIVERS if d["driver_id"] == "S001")
        result = rule_based_parse(driver)
        assert len(result.rest_constraints) == 1
        assert result.rest_constraints[0].min_hours == 8.0
        assert len(result.forbidden_categories) == 1
        assert "化工塑料" in result.forbidden_categories[0].categories
        assert len(result.geo_fences) == 1
        gf = result.geo_fences[0]
        assert gf.lat_min == 22.42
        assert gf.lat_max == 22.89

    def test_s004_first_order_and_max_orders(self):
        """S004 (D004变体): 首单时限 + 接单上限 + 午休静默。"""
        driver = next(d for d in L1_DRIVERS if d["driver_id"] == "S004")
        result = rule_based_parse(driver)
        assert len(result.first_order_deadline) == 1
        assert result.first_order_deadline[0].deadline_hour == 12
        assert len(result.max_orders) == 1
        assert result.max_orders[0].max_per_day == 3
        assert len(result.quiet_windows) == 1

    def test_s007_special_cargo_and_go_home(self):
        """S007 (D009变体): 熟货必接 + 回家 + 禁品类。"""
        driver = next(d for d in L1_DRIVERS if d["driver_id"] == "S007")
        result = rule_based_parse(driver)
        assert len(result.special_cargos) == 1
        assert result.special_cargos[0].cargo_id == "240646"
        assert result.special_cargos[0].penalty_if_missed == 10000.0
        assert len(result.go_home) == 1
        assert result.go_home[0].home_lat == 23.12
        assert result.go_home[0].deadline_hour == 23
        assert len(result.forbidden_categories) == 1

    def test_s008_family_event(self):
        """S008 (D010变体): 家事事件解析。"""
        driver = next(d for d in L1_DRIVERS if d["driver_id"] == "S008")
        result = rule_based_parse(driver)
        assert len(result.family_events) == 1
        fe = result.family_events[0]
        assert fe.trigger_time == "2026-03-10 10:00:00"
        assert fe.home_lat == 23.19
        assert fe.penalty_once_if_failed == 9000.0


# ===========================================================================
# L2 层测试：极端约束组合 → 期望全部识别
# ===========================================================================

class TestL2HardCombos:
    """L2 层：极端约束组合，测试解析器在高压场景下的正确性。"""

    def test_all_recognized(self):
        """L2 所有偏好都应被识别为非 custom 类型。"""
        failures = []
        for driver in L2_DRIVERS:
            driver_id = driver["driver_id"]
            result = rule_based_parse(driver)
            custom_count = count_custom(result)
            if custom_count > 0:
                customs = [c.original_text[:50] for c in result.custom]
                failures.append(f"{driver_id}: {custom_count} unrecognized → {customs}")

        assert len(failures) == 0, (
            f"L2 层有 {len(failures)} 个司机存在未识别偏好:\n" +
            "\n".join(f"  - {f}" for f in failures)
        )

    def test_h001_six_constraints(self):
        """H001: 6个约束全部被正确识别。"""
        driver = next(d for d in L2_DRIVERS if d["driver_id"] == "H001")
        result = rule_based_parse(driver)
        assert count_recognized(result) == 6
        assert count_custom(result) == 0
        assert len(result.rest_constraints) == 1
        assert len(result.forbidden_zones) == 1
        assert len(result.quiet_windows) == 1
        assert len(result.max_distances) == 1
        assert len(result.forbidden_categories) == 1
        assert len(result.off_days) == 1

    def test_h002_geo_plus_forbidden_zone(self):
        """H002: 地理围栏+禁区叠加，两者不冲突。"""
        driver = next(d for d in L2_DRIVERS if d["driver_id"] == "H002")
        result = rule_based_parse(driver)
        assert len(result.geo_fences) == 1
        assert len(result.forbidden_zones) == 1
        assert result.geo_fences[0].lat_min == 22.55
        assert result.forbidden_zones[0].center_lat == 22.75

    def test_h003_high_penalty_no_cap(self):
        """H003: 所有约束都无上限罚金。"""
        driver = next(d for d in L2_DRIVERS if d["driver_id"] == "H003")
        result = rule_based_parse(driver)
        assert len(result.go_home) == 1
        assert result.go_home[0].penalty_cap is None
        assert len(result.max_orders) == 1
        assert result.max_orders[0].penalty_cap is None

    def test_h005_special_cargo_with_distance(self):
        """H005: 特殊货源（远距离）+ 空驶限制的冲突场景。"""
        driver = next(d for d in L2_DRIVERS if d["driver_id"] == "H005")
        result = rule_based_parse(driver)
        assert len(result.special_cargos) == 1
        assert result.special_cargos[0].cargo_id == "350812"
        assert result.special_cargos[0].penalty_if_missed == 15000.0
        assert len(result.visit_targets) == 1
        assert result.visit_targets[0].min_days == 8
        # 两种距离限制
        pickup_dists = [d for d in result.max_distances if d.constraint_type == "pickup"]
        monthly_dists = [d for d in result.max_distances if d.constraint_type == "monthly_deadhead"]
        assert len(pickup_dists) == 1
        assert len(monthly_dists) == 1

    def test_h006_family_event_variant(self):
        """H006: 家事事件变体（不同时间、不同等待时长）。"""
        driver = next(d for d in L2_DRIVERS if d["driver_id"] == "H006")
        result = rule_based_parse(driver)
        assert len(result.family_events) == 1
        fe = result.family_events[0]
        assert fe.trigger_time == "2026-03-15 14:00:00"
        assert len(fe.waypoints) == 1
        assert fe.waypoints[0]["wait_minutes"] == 15
        assert fe.home_deadline == "2026-03-15 20:00:00"
        assert fe.stay_until == "2026-03-18 08:00:00"
        assert fe.penalty_once_if_failed == 12000.0

    def test_h008_multiple_soft_categories(self):
        """H008: 多个品类限制（硬+软）同时存在。"""
        driver = next(d for d in L2_DRIVERS if d["driver_id"] == "H008")
        result = rule_based_parse(driver)
        assert len(result.forbidden_categories) >= 3
        hard_cats = [fc for fc in result.forbidden_categories if not fc.is_soft]
        soft_cats = [fc for fc in result.forbidden_categories if fc.is_soft]
        assert len(hard_cats) >= 1
        assert len(soft_cats) >= 2


# ===========================================================================
# L3 层测试：对抗性探针 → 新类型允许进 custom，已知类型必须识别
# ===========================================================================

class TestL3Adversarial:
    """L3 层：对抗性探针，测试解析器对未知约束类型的处理。"""

    def test_known_types_still_recognized(self):
        """L3 中的已知类型约束（quiet_window、rest、forbidden_category等）仍应被正确识别。"""
        known_recognized = 0
        known_total = 0
        for driver in L3_DRIVERS:
            result = rule_based_parse(driver)
            # 已知类型不应进 custom
            recognized = count_recognized(result)
            known_recognized += recognized

            total_prefs = len(driver["preferences"])
            known_total += total_prefs

        # L3 中约有 60-70% 的偏好是已知类型，应被识别
        recognition_rate = known_recognized / known_total if known_total > 0 else 0
        assert recognition_rate >= 0.5, (
            f"L3 识别率过低: {recognition_rate:.1%} ({known_recognized}/{known_total})"
        )

    def test_novel_types_handled_gracefully(self):
        """新约束类型不应导致解析器崩溃，应进入 custom。"""
        novel_drivers = ["A001", "A002", "A003", "A004", "A005", "A006", "A010"]
        for driver in L3_DRIVERS:
            if driver["driver_id"] in novel_drivers:
                # 不应崩溃
                result = rule_based_parse(driver)
                total = count_recognized(result) + count_custom(result)
                assert total == len(driver["preferences"]), (
                    f"{driver['driver_id']}: 约束丢失"
                )

    def test_a001_weekend_restriction(self):
        """A001: '周六周日全天不接单'应能被识别或优雅降级。"""
        driver = next(d for d in L3_DRIVERS if d["driver_id"] == "A001")
        result = rule_based_parse(driver)
        # 已知类型：rest(4h) + forbidden_category(活禽畜牧) 应被识别
        assert len(result.rest_constraints) == 1
        assert len(result.forbidden_categories) == 1
        # 周末约束可能进 custom 或被识别为 off_days 变体

    def test_a004_order_interval(self):
        """A004: '两单间隔2小时'是新约束，但 off_days 和 forbidden_zone 应正确识别。"""
        driver = next(d for d in L3_DRIVERS if d["driver_id"] == "A004")
        result = rule_based_parse(driver)
        assert len(result.off_days) == 1
        assert result.off_days[0].min_days == 2
        assert len(result.forbidden_zones) == 1
        assert result.forbidden_zones[0].radius_km == 25

    def test_a007_multi_waypoint(self):
        """A007: 多目标点巡回 — rest、max_orders 和多点 visit_target 应被识别。"""
        driver = next(d for d in L3_DRIVERS if d["driver_id"] == "A007")
        result = rule_based_parse(driver)
        assert len(result.rest_constraints) == 1
        assert len(result.max_orders) == 1
        # 多点巡回：3 个 visit_target（A/B/C 三个点）
        assert len(result.visit_targets) == 3
        lats = sorted([vt.target_lat for vt in result.visit_targets])
        assert abs(lats[0] - 22.95) < 0.01
        assert abs(lats[1] - 23.10) < 0.01
        assert abs(lats[2] - 23.20) < 0.01

    def test_a009_noisy_text(self):
        """A009: 噪声文本 — 测试解析器对非标格式的鲁棒性。"""
        driver = next(d for d in L3_DRIVERS if d["driver_id"] == "A009")
        result = rule_based_parse(driver)
        # 即使文本含大量符号/英文/非标格式，基本约束应尽量被识别
        total = count_recognized(result) + count_custom(result)
        assert total == len(driver["preferences"])


# ===========================================================================
# 综合统计报告
# ===========================================================================

def run_report():
    """生成综合统计报告。"""
    print("=" * 70)
    print("  飞轮合成数据集 — Rule-Based Parser 解析覆盖率报告")
    print("=" * 70)

    layers = [
        ("L1 (措辞变体)", L1_DRIVERS),
        ("L2 (极端组合)", L2_DRIVERS),
        ("L3 (对抗探针)", L3_DRIVERS),
    ]

    for layer_name, drivers in layers:
        print(f"\n{'─' * 70}")
        print(f"  {layer_name}")
        print(f"{'─' * 70}")

        total_prefs = 0
        total_recognized = 0
        total_custom = 0

        for driver in drivers:
            driver_id = driver["driver_id"]
            n_prefs = len(driver["preferences"])
            total_prefs += n_prefs

            result = rule_based_parse(driver)
            recognized = count_recognized(result)
            custom = count_custom(result)
            total_recognized += recognized
            total_custom += custom

            summary = get_constraint_summary(result)
            summary_str = ", ".join(f"{k}={v}" for k, v in summary.items())

            icon = "✓" if custom == 0 else "△"
            print(f"  [{icon}] {driver_id}: {n_prefs} prefs → {summary_str}")

            if custom > 0:
                for c in result.custom:
                    print(f"       ⚠ custom: {c.original_text[:60]}...")

        rate = total_recognized / total_prefs * 100 if total_prefs > 0 else 0
        print(f"\n  统计: {total_recognized}/{total_prefs} 识别 ({rate:.1f}%), "
              f"{total_custom} 进入 custom")

    print(f"\n{'=' * 70}")
    print("  测试完成")
    print(f"{'=' * 70}")


# ===========================================================================
# pytest 入口 + 独立运行入口
# ===========================================================================

def test_l1_full_coverage():
    """pytest: L1 全覆盖。"""
    t = TestL1Variants()
    t.test_all_recognized()
    t.test_constraint_count_matches()


def test_l2_full_coverage():
    """pytest: L2 全覆盖。"""
    t = TestL2HardCombos()
    t.test_all_recognized()


def test_l3_graceful_handling():
    """pytest: L3 优雅降级。"""
    t = TestL3Adversarial()
    t.test_known_types_still_recognized()
    t.test_novel_types_handled_gracefully()


if __name__ == "__main__":
    run_report()
