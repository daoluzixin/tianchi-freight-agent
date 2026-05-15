"""测试 rule_based_parser：验证所有 10 个司机的偏好文本能被正确解析。"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from agent.config.rule_based_parser import rule_based_parse


# ===========================================================================
# 所有 10 个司机的偏好原文（来自 server/data/drivers.json）
# ===========================================================================

ALL_DRIVERS = {
    "D001": {
        "driver_id": "D001", "cost_per_km": 1.5,
        "current_lat": 22.54, "current_lng": 114.06,
        "preferences": [
            {"content": "我这人熬不住连轴转，每天至少连续停车熄火休息满8小时。", "penalty_amount": 300, "penalty_cap": 3000},
            {"content": "不接货源品类为「化工塑料」或「煤炭矿产」的订单。", "penalty_amount": 500, "penalty_cap": 5000},
            {"content": "我就在深圳干活，不出市。从 22.54,114.06 这一带出车；跑车或停车时，车辆位置须始终在深圳市范围内（北纬22.42至22.89，东经113.74至114.66）。", "penalty_amount": 2000, "penalty_cap": 2000},
        ]
    },
    "D002": {
        "driver_id": "D002", "cost_per_km": 1.5,
        "current_lat": 23.02, "current_lng": 113.75,
        "preferences": [
            {"content": "自然月内至少要有4个整天不接单。", "penalty_amount": 6000, "penalty_cap": 6000},
            {"content": "不接货源品类为「蔬菜」的订单。", "penalty_amount": 350, "penalty_cap": 3500},
            {"content": "每天至少有一段连着停车歇满4小时（真熄火歇脚）。", "penalty_amount": 200, "penalty_cap": 6000},
        ]
    },
    "D003": {
        "driver_id": "D003", "cost_per_km": 1.5,
        "current_lat": 23.03, "current_lng": 113.12,
        "preferences": [
            {"content": "一个月空驶赶路里程总和不得超过100公里；仅对超出部分按公里计罚。", "penalty_amount": 10, "penalty_cap": 2000},
            {"content": "车辆不得进入以（23.30，113.52）为圆心、半径20公里的区域。", "penalty_amount": 1000, "penalty_cap": 10000},
            {"content": "每天凌晨2点至5点不接单、不空车赶路（从发车赶路或去接单时刻计）。", "penalty_amount": 200, "penalty_cap": 6000},
        ]
    },
    "D004": {
        "driver_id": "D004", "cost_per_km": 1.5,
        "current_lat": 22.52, "current_lng": 113.39,
        "preferences": [
            {"content": "只要这天接了单，首单开工不得晚于当天中午12点。", "penalty_amount": 200, "penalty_cap": 4000},
            {"content": "同一天接单不得超过3单；每多一单按单计罚（无月度封顶）。", "penalty_amount": 200, "penalty_cap": None},
            {"content": "每天中午12点至下午1点吃饭歇脚，不接单、不空车赶路。", "penalty_amount": 100, "penalty_cap": 3000},
        ]
    },
    "D005": {
        "driver_id": "D005", "cost_per_km": 1.5,
        "current_lat": 22.58, "current_lng": 113.08,
        "preferences": [
            {"content": "单笔货装货点至卸货点的距离不得超过100公里。", "penalty_amount": 100, "penalty_cap": None},
            {"content": "接单后赴装货点的空驶距离不得超过90公里。", "penalty_amount": 100, "penalty_cap": None},
            {"content": "每晚23点至次日早6点不接单、不空车赶路。", "penalty_amount": 200, "penalty_cap": 6000},
        ]
    },
    "D006": {
        "driver_id": "D006", "cost_per_km": 1.5,
        "current_lat": 23.11, "current_lng": 114.42,
        "preferences": [
            {"content": "每天至少有一段连着停车休息满5小时（真停车歇着；送货与空跑不算休息）。", "penalty_amount": 200, "penalty_cap": 6000},
            {"content": "不接货源品类为「鲜活水产品」的订单。", "penalty_amount": 400, "penalty_cap": 4000},
            {"content": "单笔货装货点至卸货点距离不得超过150公里。", "penalty_amount": 250, "penalty_cap": 3000},
            {"content": "自然月内至少要有2个整天既不接单也不空车乱跑。", "penalty_amount": 3000, "penalty_cap": 3000},
        ]
    },
    "D007": {
        "driver_id": "D007", "cost_per_km": 1.5,
        "current_lat": 23.05, "current_lng": 112.46,
        "preferences": [
            {"content": "每天23点至次日4点不接单、不空车赶路；路边停车等人不算违规。", "penalty_amount": 500, "penalty_cap": 15000},
            {"content": "不接货源品类为「机械设备」的订单。", "penalty_amount": 280, "penalty_cap": 2800},
            {"content": "单笔货装货点至卸货点距离不得超过180公里。", "penalty_amount": 200, "penalty_cap": 2400},
            {"content": "自然月内至少放空一整天不接单。", "penalty_amount": 800, "penalty_cap": 800},
        ]
    },
    "D008": {
        "driver_id": "D008", "cost_per_km": 1.5,
        "current_lat": 22.27, "current_lng": 113.58,
        "preferences": [
            {"content": "自然月内至少要有2天完全歇着：不接单也不空车乱跑。", "penalty_amount": 1500, "penalty_cap": 1500},
            {"content": "平日每天连续停车休息满4小时。", "penalty_amount": 400, "penalty_cap": 12000},
            {"content": "尽量不拉货源品类为「食品饮料」的订单。", "penalty_amount": 200, "penalty_cap": 2000},
            {"content": "接单后赴装货点空驶距离不得超过50公里。", "penalty_amount": 100, "penalty_cap": 2000},
        ]
    },
    "D009": {
        "driver_id": "D009", "cost_per_km": 1.5,
        "current_lat": 23.12, "current_lng": 113.28,
        "preferences": [
            {"content": "【临时约定·熟货】指定熟货源编号240646（品类「服饰纺织皮革」）：装货地：韶关市浈江区（24.81，113.58）；上架时间：2026-03-03 14:43:36；不接则丧失该老客户信任，经济损失10000元。", "penalty_amount": 10000, "penalty_cap": 10000},
            {"content": "每天23点前车辆须在自家位置（23.12，113.28）一公里内；当天23点至次日8点不接单、不空跑。", "penalty_amount": 900, "penalty_cap": 27000},
            {"content": "不接货源品类为「快递快运搬家」的订单。", "penalty_amount": 350, "penalty_cap": 3500},
        ]
    },
    "D010": {
        "driver_id": "D010", "cost_per_km": 1.5,
        "current_lat": 23.19, "current_lng": 113.36,
        "preferences": [
            {"content": "【临时约定·家事】2026年3月10日10:00，家中急事：须先到（23.21，113.37）接上配偶（原地停留不少于10分钟），再返回老家（23.19，113.36）；须在2026年3月10日22:00前进家门，到家后须在原处静止，至少待到2026年3月13日22:00事情解决方可再出车。解决前（2026年3月10日22:00至2026年3月13日22:00）每迟到或不在家1分钟罚5元、上不封顶。未接到配偶即抵达老家、完全未抵达家中或提前离家，满足任一项额外一次性罚9000元。", "penalty_amount": 9000, "penalty_cap": None},
            {"content": "自然月内至少5个不同的自然日到过（23.13，113.26）一公里内；同日多次只算一天。", "penalty_amount": 3000, "penalty_cap": 3000},
            {"content": "每天连续停车休息至少3小时。", "penalty_amount": 300, "penalty_cap": 6000},
            {"content": "尽量不拉货源品类为「服饰纺织皮革」的订单。", "penalty_amount": 240, "penalty_cap": 2400},
        ]
    },
}


def test_driver(driver_id: str, status: dict) -> tuple[bool, str]:
    """测试单个司机的解析结果。"""
    result = rule_based_parse(status)
    total_prefs = len(status["preferences"])
    custom_count = len(result.custom)

    details = []
    if result.rest_constraints:
        details.append(f"rest={len(result.rest_constraints)}")
    if result.quiet_windows:
        details.append(f"quiet={len(result.quiet_windows)}")
    if result.forbidden_categories:
        details.append(f"forbidden_cat={len(result.forbidden_categories)}")
    if result.max_distances:
        details.append(f"max_dist={len(result.max_distances)}")
    if result.max_orders:
        details.append(f"max_orders={len(result.max_orders)}")
    if result.first_order_deadline:
        details.append(f"first_order={len(result.first_order_deadline)}")
    if result.off_days:
        details.append(f"off_days={len(result.off_days)}")
    if result.geo_fences:
        details.append(f"geo_fence={len(result.geo_fences)}")
    if result.forbidden_zones:
        details.append(f"forbidden_zone={len(result.forbidden_zones)}")
    if result.go_home:
        details.append(f"go_home={len(result.go_home)}")
    if result.special_cargos:
        details.append(f"special={len(result.special_cargos)}")
    if result.family_events:
        details.append(f"family={len(result.family_events)}")
    if result.visit_targets:
        details.append(f"visit={len(result.visit_targets)}")
    if result.custom:
        details.append(f"CUSTOM={custom_count}")

    success = custom_count == 0
    return success, f"{total_prefs} prefs → {', '.join(details)}"


def main():
    print("=" * 60)
    print("Rule-Based Parser Test: All 10 Drivers")
    print("=" * 60)

    all_pass = True
    for driver_id in sorted(ALL_DRIVERS.keys()):
        status = ALL_DRIVERS[driver_id]
        passed, detail = test_driver(driver_id, status)
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] {driver_id}: {detail}")
        if not passed:
            all_pass = False
            # Show custom items for debugging
            result = rule_based_parse(status)
            for c in result.custom:
                print(f"       → UNRECOGNIZED: {c.original_text[:60]}...")

    print()
    if all_pass:
        print("ALL PASS: 全部 10 个司机的偏好都被正确识别！")
    else:
        print("PARTIAL: 部分偏好未被识别（放入了 custom）")

    # 验证关键约束的数值正确性
    print()
    print("-" * 60)
    print("Key Value Checks:")
    print("-" * 60)

    # D009: go_home
    r = rule_based_parse(ALL_DRIVERS["D009"])
    gh = r.go_home[0]
    assert gh.home_lat == 23.12, f"D009 home_lat wrong: {gh.home_lat}"
    assert gh.home_lng == 113.28, f"D009 home_lng wrong: {gh.home_lng}"
    assert gh.deadline_hour == 23, f"D009 deadline wrong: {gh.deadline_hour}"
    assert gh.quiet_start_hour == 23, f"D009 quiet_start wrong: {gh.quiet_start_hour}"
    assert gh.quiet_end_hour == 8, f"D009 quiet_end wrong: {gh.quiet_end_hour}"
    assert gh.penalty_per_day == 900, f"D009 penalty wrong: {gh.penalty_per_day}"
    assert gh.penalty_cap == 27000, f"D009 cap wrong: {gh.penalty_cap}"
    print("  [✓] D009 go_home values correct")

    # D009: special_cargo
    sc = r.special_cargos[0]
    assert sc.cargo_id == "240646", f"D009 cargo_id wrong: {sc.cargo_id}"
    assert sc.pickup_lat == 24.81, f"D009 pickup_lat wrong: {sc.pickup_lat}"
    assert sc.pickup_lng == 113.58, f"D009 pickup_lng wrong: {sc.pickup_lng}"
    assert sc.penalty_if_missed == 10000, f"D009 penalty wrong: {sc.penalty_if_missed}"
    assert sc.available_from == "2026-03-03 14:43:36", f"D009 time wrong: {sc.available_from}"
    print("  [✓] D009 special_cargo values correct")

    # D010: family_event
    r10 = rule_based_parse(ALL_DRIVERS["D010"])
    fe = r10.family_events[0]
    assert fe.trigger_time == "2026-03-10 10:00:00", f"D010 trigger wrong: {fe.trigger_time}"
    assert len(fe.waypoints) == 1, f"D010 waypoints wrong: {fe.waypoints}"
    assert fe.waypoints[0]["lat"] == 23.21, f"D010 wp lat wrong"
    assert fe.waypoints[0]["lng"] == 113.37, f"D010 wp lng wrong"
    assert fe.waypoints[0]["wait_minutes"] == 10, f"D010 wp wait wrong"
    assert fe.home_lat == 23.19, f"D010 home_lat wrong: {fe.home_lat}"
    assert fe.home_lng == 113.36, f"D010 home_lng wrong: {fe.home_lng}"
    assert fe.home_deadline == "2026-03-10 22:00:00", f"D010 deadline wrong: {fe.home_deadline}"
    assert fe.stay_until == "2026-03-13 22:00:00", f"D010 stay wrong: {fe.stay_until}"
    assert fe.penalty_once_if_failed == 9000, f"D010 once penalty wrong: {fe.penalty_once_if_failed}"
    print("  [✓] D010 family_event values correct")

    # D010: visit_target
    vt = r10.visit_targets[0]
    assert vt.target_lat == 23.13, f"D010 target_lat wrong: {vt.target_lat}"
    assert vt.target_lng == 113.26, f"D010 target_lng wrong: {vt.target_lng}"
    assert vt.min_days == 5, f"D010 min_days wrong: {vt.min_days}"
    print("  [✓] D010 visit_target values correct")

    # D001: geo_fence
    r1 = rule_based_parse(ALL_DRIVERS["D001"])
    gf = r1.geo_fences[0]
    assert gf.lat_min == 22.42, f"D001 lat_min wrong: {gf.lat_min}"
    assert gf.lat_max == 22.89, f"D001 lat_max wrong: {gf.lat_max}"
    assert gf.lng_min == 113.74, f"D001 lng_min wrong: {gf.lng_min}"
    assert gf.lng_max == 114.66, f"D001 lng_max wrong: {gf.lng_max}"
    print("  [✓] D001 geo_fence values correct")

    # D003: forbidden_zone
    r3 = rule_based_parse(ALL_DRIVERS["D003"])
    fz = r3.forbidden_zones[0]
    assert fz.center_lat == 23.30, f"D003 center_lat wrong: {fz.center_lat}"
    assert fz.center_lng == 113.52, f"D003 center_lng wrong: {fz.center_lng}"
    assert fz.radius_km == 20, f"D003 radius wrong: {fz.radius_km}"
    print("  [✓] D003 forbidden_zone values correct")

    # D008: soft forbidden + weekday rest
    r8 = rule_based_parse(ALL_DRIVERS["D008"])
    assert any(fc.is_soft for fc in r8.forbidden_categories), "D008 should have soft forbidden"
    assert any(rc.weekday_only for rc in r8.rest_constraints), "D008 should have weekday-only rest"
    print("  [✓] D008 soft_forbidden + weekday_only correct")

    print()
    print("ALL VALUE CHECKS PASSED!")


if __name__ == "__main__":
    main()
