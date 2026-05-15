"""对 L2 仿真结果做逐条偏好违规校验 + 真实净收益核算。

用法：
    cd demo && python agent/tests/check_l2_violations.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent.parent.parent
SERVER_ROOT = DEMO_ROOT / "server"
sys.path.insert(0, str(DEMO_ROOT))
from simkit.simulation_actions import haversine_km

LOG_DIR = DEMO_ROOT / "log" / "20260514_163132"
CARGO_PATH = SERVER_ROOT / "data" / "cargo_dataset.jsonl"
L2_DRIVERS_PATH = DEMO_ROOT / "agent" / "tests" / "synthetic_drivers" / "drivers_l2_sim.json"
COST_PER_KM = 1.5
REPOSITION_SPEED = 60.0  # km/h


# ── helpers ──────────────────────────────────────────────
def load_cargo_map():
    m = {}
    with CARGO_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            cid = str(item["cargo_id"])
            s, e = item["start"], item["end"]
            m[cid] = {
                "cargo_name": item.get("cargo_name", ""),
                "start_lat": float(s["lat"]), "start_lng": float(s["lng"]),
                "end_lat": float(e["lat"]), "end_lng": float(e["lng"]),
                "distance_km": haversine_km(float(s["lat"]), float(s["lng"]),
                                            float(e["lat"]), float(e["lng"])),
            }
    return m


def load_actions(driver_id: str):
    files = list(LOG_DIR.glob(f"actions_202603_{driver_id}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No actions file for {driver_id}")
    rows = []
    prev_end = 0
    with files[0].open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            act = rec["action"]
            res = rec.get("result", {})
            pb = rec.get("position_before", {})
            pa = rec.get("position_after", {})
            qc = int(rec.get("query_scan_cost_minutes", 0))
            ac = int(rec.get("action_exec_cost_minutes", 0))
            end = int(res.get("simulation_progress_minutes", 0))
            rows.append({
                "action": act["action"],
                "params": act.get("params", {}),
                "result": res,
                "step_start": prev_end,
                "action_start": prev_end + qc,
                "action_end": prev_end + qc + ac,
                "step_end": end,
                "exec_cost": ac,
                "before_lat": float(pb.get("lat", 0)),
                "before_lng": float(pb.get("lng", 0)),
                "after_lat": float(pa.get("lat", 0)),
                "after_lng": float(pa.get("lng", 0)),
            })
            prev_end = end
    return rows


def overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def longest_wait_streak(rows, day):
    """day 内最长连续 wait 分钟数。"""
    d0, d1 = day * 1440, (day + 1) * 1440
    intervals = []
    for r in rows:
        if r["action"] != "wait" or r["exec_cost"] <= 0:
            continue
        s = max(r["step_start"], d0)
        e = min(r["step_end"], d1)
        if e > s:
            intervals.append((s, e))
    intervals.sort()
    merged = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return max((e - s for s, e in merged), default=0)


def accepted_orders(rows):
    return [r for r in rows if r["action"] == "take_order" and r["result"].get("accepted")]


def sum_deadhead(rows):
    total = 0.0
    for r in rows:
        if r["action"] == "reposition":
            total += float(r["result"].get("distance_km", 0) or 0)
        elif r["action"] == "take_order" and r["result"].get("accepted"):
            total += float(r["result"].get("pickup_deadhead_km", 0) or 0)
    return total


def sum_distance(rows):
    total = 0.0
    for r in rows:
        if r["action"] == "reposition":
            total += float(r["result"].get("distance_km", 0) or 0)
        elif r["action"] == "take_order" and r["result"].get("accepted"):
            total += float(r["result"].get("pickup_deadhead_km", 0) or 0)
            total += float(r["result"].get("haul_distance_km", 0) or 0)
    return total


def gross_income(rows, cargo_map):
    total = 0.0
    for r in accepted_orders(rows):
        cid = str(r["params"].get("cargo_id", ""))
        # 从 result 里拿 income_eligible
        if r["result"].get("income_eligible", True):
            # 需要从 cargo_dataset 拿 price，但 actions 里没有
            # 用 result 里的信息推算不了，直接用 summary 里的毛收入
            pass
    return total


# ── H001 校验 ────────────────────────────────────────────
def check_h001(rows, cargo_map):
    print("=" * 70)
    print("H001 压力甲 — 6 约束校验")
    print("=" * 70)
    days = list(range(31))
    orders = accepted_orders(rows)
    total_penalty = 0.0

    # 1) rest_constraint: 每天连续停车休息 ≥ 6h
    viol = 0
    for d in days:
        if longest_wait_streak(rows, d) < 360:
            viol += 1
    pen = min(viol * 400, 12000)
    total_penalty += pen
    print(f"  [1] 每日连续休息≥6h: 违规 {viol} 天, 罚 {pen:.0f} (cap 12000)")

    # 2) forbidden_zone: 不进入 (22.95,113.80) 半径 15km
    viol = 0
    for r in rows:
        if haversine_km(r["before_lat"], r["before_lng"], 22.95, 113.80) <= 15:
            viol += 1
        elif haversine_km(r["after_lat"], r["after_lng"], 22.95, 113.80) <= 15:
            viol += 1
    pen = min(viol * 1500, 15000)
    total_penalty += pen
    print(f"  [2] 禁区(22.95,113.80)r=15km: 违规 {viol} 步, 罚 {pen:.0f} (cap 15000)")

    # 3) quiet_window: 01:00-07:00 不接单不空跑
    viol_days = set()
    for r in rows:
        if r["action"] not in ("take_order", "reposition"):
            continue
        for d in days:
            ws = d * 1440 + 60
            we = d * 1440 + 420
            if overlap(r["action_start"], r["action_end"], ws, we):
                viol_days.add(d)
    pen = min(len(viol_days) * 300, 9000)
    total_penalty += pen
    print(f"  [3] 安静窗口01-07: 违规 {len(viol_days)} 天, 罚 {pen:.0f} (cap 9000)")

    # 4) max_haul: 单笔运距 ≤ 120km
    viol = 0
    for r in orders:
        cid = str(r["params"].get("cargo_id", ""))
        cg = cargo_map.get(cid)
        if cg and cg["distance_km"] > 120:
            viol += 1
    pen = min(viol * 150, 3000)
    total_penalty += pen
    print(f"  [4] 单笔运距≤120km: 违规 {viol} 单, 罚 {pen:.0f} (cap 3000)")

    # 5) forbidden_category: 不接建材陶瓷/化工塑料
    forbidden = {"建材陶瓷", "化工塑料"}
    viol = 0
    for r in orders:
        cid = str(r["params"].get("cargo_id", ""))
        cg = cargo_map.get(cid)
        if cg and cg["cargo_name"] in forbidden:
            viol += 1
    pen = min(viol * 600, 6000)
    total_penalty += pen
    print(f"  [5] 禁接建材陶瓷/化工塑料: 违规 {viol} 单, 罚 {pen:.0f} (cap 6000)")

    # 6) off_days: 月内至少 3 天完全歇着
    active_days = set()
    for r in rows:
        if r["action"] in ("take_order", "reposition"):
            active_days.add(r["step_end"] // 1440)
    off = 31 - len(active_days)
    pen = 0 if off >= 3 else 5000
    total_penalty += pen
    print(f"  [6] 月内≥3天休息: 休息 {off} 天, 罚 {pen:.0f} (cap 5000)")

    dist = sum_distance(rows)
    cost = dist * COST_PER_KM
    print(f"  ── 总里程: {dist:.1f} km, 成本: {cost:.1f}")
    print(f"  ── 总罚分: {total_penalty:.0f}")
    return total_penalty, cost


# ── H002 校验 ────────────────────────────────────────────
def check_h002(rows, cargo_map):
    print("=" * 70)
    print("H002 压力乙 — 4 约束校验")
    print("=" * 70)
    days = list(range(31))
    orders = accepted_orders(rows)
    total_penalty = 0.0

    # 1) geo_fence: 北纬22.55-22.92, 东经113.68-114.25
    viol = 0
    for r in rows:
        for lat, lng in [(r["before_lat"], r["before_lng"]), (r["after_lat"], r["after_lng"])]:
            if not (22.55 <= lat <= 22.92 and 113.68 <= lng <= 114.25):
                viol += 1
                break
    pen = min(viol * 2500, 2500)
    total_penalty += pen
    print(f"  [1] 地理围栏(22.55-22.92, 113.68-114.25): 违规 {viol} 步, 罚 {pen:.0f} (cap 2500)")

    # 2) forbidden_zone: 不进入 (22.75,113.90) 半径 10km
    viol = 0
    for r in rows:
        if haversine_km(r["before_lat"], r["before_lng"], 22.75, 113.90) <= 10:
            viol += 1
        elif haversine_km(r["after_lat"], r["after_lng"], 22.75, 113.90) <= 10:
            viol += 1
    pen = min(viol * 800, 8000)
    total_penalty += pen
    print(f"  [2] 禁区(22.75,113.90)r=10km: 违规 {viol} 步, 罚 {pen:.0f} (cap 8000)")

    # 3) max_pickup: 空驶距离 ≤ 30km
    viol = 0
    for r in orders:
        pk = float(r["result"].get("pickup_deadhead_km", 0) or 0)
        if pk > 30:
            viol += 1
    # penalty_cap = null → 无封顶
    pen = viol * 200
    total_penalty += pen
    print(f"  [3] 赴装货点空驶≤30km: 违规 {viol} 单, 罚 {pen:.0f} (无封顶)")

    # 4) quiet_window: 14:00-15:30 不接单不空跑
    viol_days = set()
    for r in rows:
        if r["action"] not in ("take_order", "reposition"):
            continue
        for d in days:
            ws = d * 1440 + 14 * 60
            we = d * 1440 + 15 * 60 + 30
            if overlap(r["action_start"], r["action_end"], ws, we):
                viol_days.add(d)
    pen = min(len(viol_days) * 150, 4500)
    total_penalty += pen
    print(f"  [4] 午休14:00-15:30: 违规 {len(viol_days)} 天, 罚 {pen:.0f} (cap 4500)")

    dist = sum_distance(rows)
    cost = dist * COST_PER_KM
    print(f"  ── 总里程: {dist:.1f} km, 成本: {cost:.1f}")
    print(f"  ── 总罚分: {total_penalty:.0f}")
    return total_penalty, cost


# ── H006 校验 ────────────────────────────────────────────
def check_h006(rows, cargo_map):
    print("=" * 70)
    print("H006 压力己 — 3 约束校验")
    print("=" * 70)
    days = list(range(31))
    orders = accepted_orders(rows)
    total_penalty = 0.0

    # 1) family_event: 3/15 14:00 触发
    #    须先到(22.97,113.72)接配偶(停留≥15min)，再回老家(22.95,113.70)
    #    须在 3/15 20:00 前进家门
    #    到家后静止至 3/18 08:00
    #    迟到或不在家 1min 罚 8 元无封顶
    #    未接配偶/未到家/提前离家 额外一次性罚 12000
    family_start = 14 * 1440 + 14 * 60  # 3/15 14:00 = day14 14:00
    home_deadline = 14 * 1440 + 20 * 60  # 3/15 20:00
    stay_end = 17 * 1440 + 8 * 60       # 3/18 08:00

    # 检查是否接了配偶（到 22.97,113.72 停留 ≥ 15min）
    pickup_done_time = None
    pickup_wait = 0
    for r in rows:
        if r["step_end"] <= family_start:
            continue
        if r["action"] == "wait":
            if haversine_km(r["after_lat"], r["after_lng"], 22.97, 113.72) <= 1.0:
                pickup_wait += r["exec_cost"]
                if pickup_wait >= 15 and pickup_done_time is None:
                    pickup_done_time = r["step_end"]
            else:
                pickup_wait = 0
        else:
            pickup_wait = 0

    # 检查是否到家（22.95, 113.70）
    first_home_time = None
    for r in rows:
        if r["step_end"] < family_start:
            continue
        if haversine_km(r["after_lat"], r["after_lng"], 22.95, 113.70) <= 1.0:
            first_home_time = r["step_end"]
            break

    sequence_ok = (pickup_done_time is not None and first_home_time is not None
                   and pickup_done_time <= first_home_time)

    # 计算不在家分钟数（从 family_start 到 stay_end）
    minutes_not_home = 0
    for r in rows:
        seg_a = max(r["step_start"], home_deadline)
        seg_b = min(r["step_end"], stay_end)
        if seg_b <= seg_a:
            continue
        at_home = (r["action"] == "wait"
                   and haversine_km(r["before_lat"], r["before_lng"], 22.95, 113.70) <= 1.0
                   and haversine_km(r["after_lat"], r["after_lng"], 22.95, 113.70) <= 1.0)
        if not at_home:
            minutes_not_home += (seg_b - seg_a)

    pen_absence = minutes_not_home * 8
    pen_fixed = 12000 if not sequence_ok else 0

    # 检查是否提前离家
    left_early = False
    if first_home_time is not None:
        for r in rows:
            if r["step_start"] < first_home_time:
                continue
            if r["step_start"] >= stay_end:
                break
            if r["action"] in ("take_order", "reposition"):
                left_early = True
                break
    never_arrived = first_home_time is None
    if never_arrived or left_early:
        pen_fixed = 12000

    pen_family = pen_absence + pen_fixed  # penalty_cap = null
    total_penalty += pen_family
    print(f"  [1] 家事(3/15-3/18):")
    print(f"      接配偶: {'✅' if pickup_done_time else '❌'} (time={pickup_done_time})")
    print(f"      到家: {'✅' if first_home_time else '❌'} (time={first_home_time}, deadline={home_deadline})")
    if first_home_time and first_home_time > home_deadline:
        late_min = first_home_time - home_deadline
        print(f"      迟到: {late_min} 分钟")
    print(f"      序列正确: {sequence_ok}, 提前离家: {left_early}")
    print(f"      不在家分钟: {minutes_not_home}, 缺勤罚: {pen_absence}")
    print(f"      固定罚: {pen_fixed}")
    print(f"      家事总罚: {pen_family:.0f} (无封顶)")

    # 2) rest_constraint: 除家事期间外，每天连续停车休息 ≥ 5h
    family_days = {14, 15, 16, 17}  # 3/15-3/18 对应 day 14-17
    viol = 0
    for d in days:
        if d in family_days:
            continue
        if longest_wait_streak(rows, d) < 300:
            viol += 1
    pen = min(viol * 350, 10000)
    total_penalty += pen
    print(f"  [2] 每日连续休息≥5h(除家事): 违规 {viol} 天, 罚 {pen:.0f} (cap 10000)")

    # 3) forbidden_category: 不接废品回收/煤炭矿产
    forbidden = {"废品回收", "煤炭矿产"}
    viol = 0
    for r in orders:
        cid = str(r["params"].get("cargo_id", ""))
        cg = cargo_map.get(cid)
        if cg and cg["cargo_name"] in forbidden:
            viol += 1
    pen = min(viol * 450, 4500)
    total_penalty += pen
    print(f"  [3] 禁接废品回收/煤炭矿产: 违规 {viol} 单, 罚 {pen:.0f} (cap 4500)")

    dist = sum_distance(rows)
    cost = dist * COST_PER_KM
    print(f"  ── 总里程: {dist:.1f} km, 成本: {cost:.1f}")
    print(f"  ── 总罚分: {total_penalty:.0f}")
    return total_penalty, cost


# ── main ─────────────────────────────────────────────────
def main():
    cargo_map = load_cargo_map()
    # 从 summary 拿毛收入
    summary = json.loads((LOG_DIR / "summary.json").read_text())
    driver_income = summary["income"]["driver_details"]

    results = {}
    for did, check_fn in [("H001", check_h001), ("H002", check_h002), ("H006", check_h006)]:
        rows = load_actions(did)
        penalty, cost = check_fn(rows, cargo_map)
        gross = driver_income[did]["gross_income"]
        net = gross - cost - penalty
        results[did] = {"gross": gross, "cost": cost, "penalty": penalty, "net": net}
        print(f"\n  ★ {did} 最终: 毛收入 {gross:.0f} - 成本 {cost:.0f} - 罚分 {penalty:.0f} = 净收益 {net:+,.0f}")
        print()

    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"  {'司机':<6} {'毛收入':>10} {'成本':>10} {'罚分':>10} {'净收益':>12}")
    print(f"  {'─' * 52}")
    total_net = 0
    for did in ["H001", "H002", "H006"]:
        r = results[did]
        print(f"  {did:<6} {r['gross']:>10,.0f} {r['cost']:>10,.0f} {r['penalty']:>10,.0f} {r['net']:>+12,.0f}")
        total_net += r["net"]
    print(f"  {'─' * 52}")
    print(f"  {'合计':<6} {sum(r['gross'] for r in results.values()):>10,.0f} "
          f"{sum(r['cost'] for r in results.values()):>10,.0f} "
          f"{sum(r['penalty'] for r in results.values()):>10,.0f} "
          f"{total_net:>+12,.0f}")


if __name__ == "__main__":
    main()
