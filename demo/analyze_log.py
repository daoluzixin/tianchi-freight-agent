"""分析仿真日志，提取 D001 的关键统计数据。"""
import re
import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else "/var/folders/j0/kfq02d6173b6h2s4mx19btvh0000gn/T/catpaw-cli-shell/terminals/ee47d2ac-9b0e-4d99-9c6c-3192c02961e0-shell-32.log"

with open(log_path) as f:
    lines = f.readlines()

steps = []
orders = []
actions = {}
total_tokens = 0
rule_pass = []
daily_reviews = []
llm_decisions = []

for line in lines:
    # STEP lines
    m = re.search(r'\[STEP\].*driver=(\w+)\s+step=(\d+).*sim_clock=(\S+)->(\S+).*decision=(\w+).*params=(\{[^}]*\}).*total=(\d+)', line)
    if m:
        driver = m.group(1)
        step = int(m.group(2))
        sim_start = m.group(3)
        sim_end = m.group(4)
        action = m.group(5)
        tokens = int(m.group(7))
        actions[action] = actions.get(action, 0) + 1
        total_tokens += tokens
        steps.append({"step": step, "driver": driver, "action": action, "tokens": tokens, "sim_end": sim_end})

    # Accepted orders
    if '"accepted":true' in line or '"accepted": true' in line:
        pickup_m = re.search(r'pickup_deadhead_km":\s*([\d.]+)', line)
        haul_m = re.search(r'haul_distance_km":\s*([\d.]+)', line)
        sim_m = re.search(r'simulation_wall_time":"([^"]+)"', line)
        cargo_m = re.search(r'cargo_id":"([^"]+)"', line)
        if pickup_m and haul_m:
            orders.append({
                "cargo_id": cargo_m.group(1) if cargo_m else "?",
                "pickup_km": float(pickup_m.group(1)),
                "haul_km": float(haul_m.group(1)),
                "time": sim_m.group(1) if sim_m else "?"
            })

    # Rule engine
    rm = re.search(r'rule engine: (\d+)/(\d+) passed', line)
    if rm:
        rule_pass.append((int(rm.group(1)), int(rm.group(2))))

    # Daily review
    dr = re.search(r'daily review day=(\d+): aggression=([\d.]+), wait_threshold=([\d.-]+)', line)
    if dr:
        daily_reviews.append({
            "day": int(dr.group(1)),
            "aggression": float(dr.group(2)),
            "wait_threshold": float(dr.group(3)),
        })

    # LLM decisions
    ld = re.search(r'LLM enhanced: (\w+) (\w+) \(conf=([\d.]+)', line)
    if ld:
        llm_decisions.append({
            "action": ld.group(1),
            "cargo": ld.group(2),
            "conf": float(ld.group(3)),
        })

print("=" * 60)
print("D001 仿真中期分析报告（截至 step {}）".format(len(steps)))
print("=" * 60)

print(f"\n--- 总步数: {len(steps)}")
print(f"--- 动作分布:")
for a, c in sorted(actions.items(), key=lambda x: -x[1]):
    pct = c / len(steps) * 100 if steps else 0
    print(f"    {a}: {c} ({pct:.0f}%)")

print(f"\n--- 接单统计:")
print(f"    成功接单: {len(orders)} 单")
if orders:
    total_pickup = sum(o["pickup_km"] for o in orders)
    total_haul = sum(o["haul_km"] for o in orders)
    total_dist = total_pickup + total_haul
    print(f"    总空驶距离: {total_pickup:.1f} km")
    print(f"    平均空驶距离: {total_pickup/len(orders):.1f} km")
    print(f"    总运距: {total_haul:.1f} km")
    print(f"    平均运距: {total_haul/len(orders):.1f} km")
    print(f"    空驶比: {total_pickup/total_dist*100:.1f}%")
    print(f"    总行驶距离: {total_dist:.1f} km")
    print(f"    距离成本 (x1.5): {total_dist * 1.5:.1f} 元")
    # 每天接单频率
    if orders:
        first_time = orders[0]["time"]
        last_time = orders[-1]["time"]
        # 粗略估算天数
        first_day = int(first_time.split(" ")[0].split("-")[2])
        last_day = int(last_time.split(" ")[0].split("-")[2])
        days = last_day - first_day + 1
        print(f"    跨越天数: {days} 天")
        print(f"    日均接单: {len(orders)/days:.1f} 单")

print(f"\n--- 规则引擎过滤:")
if rule_pass:
    avg_pass = sum(p for p, t in rule_pass) / len(rule_pass)
    avg_total = sum(t for p, t in rule_pass) / len(rule_pass)
    print(f"    查询次数: {len(rule_pass)}")
    print(f"    平均通过: {avg_pass:.1f}/{avg_total:.0f} ({avg_pass/avg_total*100:.1f}%)")
    min_pass = min(p for p, t in rule_pass)
    max_pass = max(p for p, t in rule_pass)
    print(f"    通过范围: {min_pass} ~ {max_pass}")

print(f"\n--- Token 使用:")
print(f"    总 token: {total_tokens:,}")
llm_steps = sum(1 for s in steps if s["tokens"] > 0)
print(f"    LLM 调用步数: {llm_steps}/{len(steps)} ({llm_steps/len(steps)*100:.0f}%)" if steps else "    无步数")
if llm_steps > 0:
    avg_tokens = total_tokens / llm_steps
    print(f"    平均每次 LLM: {avg_tokens:.0f} token")
    # 估算月度 token 预算
    # D001 约 300 步/月，10 个司机约 3000 步
    est_monthly = total_tokens / len(steps) * 300 * 10
    print(f"    估算月度总 token: {est_monthly:,.0f} (预算 5M)")

print(f"\n--- 每日策略回顾:")
for dr in daily_reviews:
    print(f"    Day {dr['day']}: aggression={dr['aggression']}, wait_threshold={dr['wait_threshold']}")

print(f"\n--- LLM 增强决策:")
print(f"    总次数: {len(llm_decisions)}")
if llm_decisions:
    avg_conf = sum(d["conf"] for d in llm_decisions) / len(llm_decisions)
    print(f"    平均置信度: {avg_conf:.2f}")
    take_count = sum(1 for d in llm_decisions if d["action"] == "take")
    print(f"    接单决策: {take_count}/{len(llm_decisions)}")

print(f"\n--- 接单时间线:")
for i, o in enumerate(orders):
    print(f"    #{i+1:2d}: {o['time']} | cargo={o['cargo_id']:>6s} | 空驶 {o['pickup_km']:5.1f}km | 运距 {o['haul_km']:5.1f}km")

# 按天统计
print(f"\n--- 按天接单统计:")
day_orders = {}
for o in orders:
    day = o["time"].split(" ")[0]
    if day not in day_orders:
        day_orders[day] = []
    day_orders[day].append(o)
for day in sorted(day_orders.keys()):
    ods = day_orders[day]
    day_pickup = sum(o["pickup_km"] for o in ods)
    day_haul = sum(o["haul_km"] for o in ods)
    print(f"    {day}: {len(ods)} 单 | 空驶 {day_pickup:.1f}km | 运距 {day_haul:.1f}km")
