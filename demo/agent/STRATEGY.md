# 司机找货 Agent 策略设计方案

## 一、总体思路

这不是一个简单的"选最贵货"问题，而是一个**受约束的序贯决策优化**。核心挑战在于：偏好罚分可能远超单笔订单利润（一个 D010 家事违规就扣 9000+），因此策略的第一优先级是**零违规**，第二优先级才是**最大化接单收益**。

设计哲学：**规则驱动的硬约束层 + LLM 驱动的软决策层**。把能确定性计算的东西（时间窗检查、距离阈值、禁忌品类过滤）从 LLM 决策中剥离出来做前置过滤，LLM 只负责在合规候选中做"接哪个 / 是否等待 / 去哪空驶"的权衡判断。

---

## 二、架构总览

```
┌─────────────────────────────────────────────────────┐
│                  decide(driver_id)                    │
├─────────────────────────────────────────────────────┤
│ 1. StateTracker     — 维护累计状态（收益/里程/休息）    │
│ 2. SchedulePlanner  — 时间规划（强制休息窗/特殊事件）   │
│ 3. RuleEngine       — 偏好规则前置过滤               │
│ 4. CargoScorer      — 候选评分排序                   │
│ 5. LLMDecider       — 最终决策（精简 prompt）         │
└─────────────────────────────────────────────────────┘
```

每步决策的执行流程：

```
get_driver_status → StateTracker.update
                  → SchedulePlanner.check_mandatory_action
                  → (若有强制动作) → 直接返回 wait/reposition
                  → (否则) → query_cargo
                  → RuleEngine.filter(candidates)
                  → CargoScorer.rank(filtered)
                  → LLMDecider.decide(context)
                  → 返回 action
```

---

## 三、各模块详细设计

### 3.1 StateTracker（状态追踪器）

在 Agent 进程内维护一个轻量字典，每步更新，不依赖 LLM 记忆：

```python
state = {
    "driver_id": "D001",
    "current_day": 5,              # 当前是几号（day_idx）
    "sim_minutes": 7200,           # 仿真累计分钟
    "total_gross_income": 12500.0, # 累计毛收入
    "total_distance_km": 320.5,    # 累计总里程
    "total_deadhead_km": 45.2,     # 累计空驶里程（D003 关键）
    "today_order_count": 2,        # 今日接单数（D004 关键）
    "today_first_order_minute": None,  # 今日首单时刻
    "last_rest_start": 6800,       # 上次连续休息开始时刻
    "longest_rest_today": 180,     # 今日最长连续休息分钟数
    "off_days": [3, 8],            # 完全不接单的天（D002/D006/D007/D008）
    "visit_target_days": set(),    # D010 到访目标点的天数
    "forbidden_cargo_names": {"化工塑料", "煤炭矿产"},  # 禁忌品类
    # ... 按 driver 动态配置
}
```

关键点：用 `query_decision_history(driver_id, -1)` 在首步拉取全量历史重建状态（应对重启场景），后续每步增量更新。

### 3.2 SchedulePlanner（时间规划器）

对每个司机，根据偏好预先计算出**强制行为时间表**。这是拉开分数的核心——在 LLM 做决策之前，先检查"此刻是否必须做某事"。

#### 通用规划逻辑：

```python
def check_mandatory_action(state, driver_config):
    sim_min = state["sim_minutes"]
    day = sim_min // 1440
    hour_in_day = (sim_min % 1440) // 60

    # 检查：是否处于"禁止活动"时段
    if is_in_quiet_window(driver_config, sim_min):
        # 计算距禁止窗口结束还有多久，直接 wait
        remaining = quiet_window_end(driver_config, day) - sim_min
        return {"action": "wait", "params": {"duration_minutes": remaining}}

    # 检查：今日是否需要强制休息来满足连续休息要求
    if needs_mandatory_rest(state, driver_config):
        needed = driver_config["min_continuous_rest"] - state["longest_rest_today"]
        return {"action": "wait", "params": {"duration_minutes": needed}}

    # 检查：D009 必须在 23 点前回家
    if driver_config["must_return_home_by_23"]:
        minutes_to_23 = day * 1440 + 23 * 60 - sim_min
        travel_time_home = estimate_travel_minutes(state["position"], driver_config["home"])
        if minutes_to_23 <= travel_time_home + 30:  # 留 30 分钟余量
            return {"action": "reposition", "params": home_coords}

    # 检查：D010 特殊事件窗口
    if driver_config.get("family_event") and is_family_event_active(sim_min):
        return handle_family_event(state, driver_config)

    return None  # 无强制动作，进入正常决策流程
```

#### 各司机的时间规划重点：

| 司机 | 强制休息需求 | 禁止活动窗口 | 特殊事件 |
|------|------------|------------|---------|
| D001 | 每天连续 8h | 无 | 无 |
| D002 | 每天连续 4h + 月内 4 天全休 | 无 | 无 |
| D003 | 无 | 每天 02:00-05:00 | 无 |
| D004 | 无 | 每天 12:00-13:00 | 无 |
| D005 | 无 | 每天 23:00-06:00 | 无 |
| D006 | 每天连续 5h + 月内 2 天全休 | 无 | 无 |
| D007 | 月内 1 天全休 | 每天 23:00-04:00 | 无 |
| D008 | 平日连续 4h + 月内 2 天全休 | 无 | 无 |
| D009 | 无 | 每天 23:00-08:00 | 必接熟货 240646 |
| D010 | 每天连续 3h | 无 | 3/10 家事（最高优先级） |

**休息策略建议**：把强制休息安排在**货源低谷时段**（通常凌晨 0-6 点货少），这样既满足约束又不损失高价值接单窗口。例如 D001 需要 8 小时连续休息，最佳安排是 22:00-06:00 休息，白天全力接单。

### 3.3 RuleEngine（规则引擎）

在 query_cargo 返回结果后、送入 LLM 前，做确定性过滤：

```python
def filter_cargo(candidates, state, driver_config):
    valid = []
    for item in candidates:
        cargo = item["cargo"]

        # 1. 禁忌品类过滤
        if cargo.get("cargo_name") in driver_config["forbidden_categories"]:
            continue

        # 2. 距离上限过滤（D005: haul≤100km, pickup≤90km; D006: haul≤150km 等）
        if driver_config.get("max_haul_km"):
            haul_km = haversine(cargo["start"], cargo["end"])
            if haul_km > driver_config["max_haul_km"]:
                continue
        if driver_config.get("max_pickup_km"):
            if item["distance_km"] > driver_config["max_pickup_km"]:
                continue

        # 3. 装货时间窗检查（到达时已过窗则排除）
        if cargo.get("load_time"):
            arrival_min = estimate_arrival(state, cargo["start"], item["distance_km"])
            load_end = parse_to_minutes(cargo["load_time"][1])
            if arrival_min > load_end:
                continue

        # 4. 地理围栏检查
        if driver_config.get("geo_fence"):
            if not in_fence(cargo["start"], driver_config["geo_fence"]):
                continue
            if not in_fence(cargo["end"], driver_config["geo_fence"]):
                continue

        # 5. 禁入区域检查（D003: 卸货点不在禁区）
        if driver_config.get("forbidden_zone"):
            zone = driver_config["forbidden_zone"]
            if haversine(cargo["end"], zone["center"]) <= zone["radius_km"]:
                continue

        # 6. 接单后完成时刻不超月末
        finish_estimate = estimate_finish_time(state, cargo, item["distance_km"])
        if finish_estimate > 31 * 1440:  # 超月末
            continue

        # 7. D004: 今日已 3 单则不再接
        if driver_config.get("max_daily_orders"):
            if state["today_order_count"] >= driver_config["max_daily_orders"]:
                continue

        # 8. 时间窗约束：接单/空驶动作不能落在禁止时段内
        if would_violate_quiet_window(state, cargo, driver_config):
            continue

        valid.append(item)

    return valid
```

这一层过滤掉 60-80% 的无效候选，大幅减轻 LLM 负担。

### 3.4 CargoScorer（货源评分）

对过滤后的候选做多维度打分排序，取 Top-5 送入 LLM：

```python
def score_cargo(item, state, driver_config):
    cargo = item["cargo"]
    price_yuan = cargo["price"]  # 已转换为元
    pickup_km = item["distance_km"]
    haul_km = haversine(cargo["start"], cargo["end"])
    total_km = pickup_km + haul_km
    cost = total_km * driver_config["cost_per_km"]

    # 净利润
    profit = price_yuan - cost

    # 时间效率：利润 / 总耗时（含空驶 + 运输）
    total_minutes = estimate_total_minutes(pickup_km, cargo["cost_time_minutes"])
    efficiency = profit / max(total_minutes, 1)

    # 位置奖励：卸货点靠近下一步货源热区加分
    position_bonus = 0
    if is_near_cargo_hotspot(cargo["end"]):
        position_bonus = 0.1 * profit

    # D010 目标点加分：卸货点如果靠近 (23.13, 113.26) 则加分
    visit_bonus = 0
    if driver_config.get("visit_target"):
        if haversine(cargo["end"], driver_config["visit_target"]) <= 1.0:
            if state["current_day"] not in state["visit_target_days"]:
                visit_bonus = 500  # 一次到访等价 500 元收益

    score = efficiency + position_bonus + visit_bonus
    return score
```

### 3.5 LLMDecider（模型决策器）

经过前面层层过滤后，LLM 收到的 prompt 极度精简，只需做"最后一公里"判断：

```python
SYSTEM_PROMPT = """你是货运调度决策器。基于以下信息输出一个 JSON 决策。
规则：
1. 如有高评分候选且当前非休息需求时段，优先接单
2. 如无合格货源且非禁止时段，wait 30-60 分钟后重新查看
3. 如当前位置货源持续匮乏（连续 2 步无货），考虑 reposition 到最近热区
4. 始终保守——宁可少接一单也不违规

输出格式：{"action":"take_order|reposition|wait","params":{...},"reasoning":"一句话理由"}
"""

def build_user_prompt(state, top_candidates, driver_config):
    return json.dumps({
        "sim_time": format_time(state["sim_minutes"]),
        "position": state["position"],
        "today_stats": {
            "orders": state["today_order_count"],
            "rest_minutes": state["longest_rest_today"],
            "remaining_hours": (24 - (state["sim_minutes"] % 1440) / 60)
        },
        "monthly_progress": {
            "day": state["current_day"] + 1,
            "net_income_so_far": state["total_gross_income"] - state["total_distance_km"] * 1.5,
            "off_days_achieved": len(state["off_days"]),
        },
        "top_candidates": [
            {
                "cargo_id": c["cargo"]["cargo_id"],
                "price_yuan": c["cargo"]["price"],
                "pickup_km": round(c["distance_km"], 1),
                "haul_km": round(haversine(c["cargo"]["start"], c["cargo"]["end"]), 1),
                "category": c["cargo"].get("cargo_name"),
                "score": round(c["_score"], 2),
                "estimated_profit": round(c["_profit"], 0),
            }
            for c in top_candidates[:5]
        ],
        "nearby_hotspots": get_nearby_hotspots(state["position"]),
    }, ensure_ascii=False)
```

关键优化点：只给 LLM 5 个已评分的候选（而非默认的 20 个原始候选），Prompt 总 token 控制在 500-800 以内，复赛 500 万 Token 预算下可支撑约 3000-5000 步决策。

---

## 四、特殊事件处理（拉开差距的关键）

### 4.1 D009 — 熟货必接

货源编号 240646，上架时间 2026-03-03 14:43:36，装货地韶关市浈江区 (24.81, 113.58)。

策略：
- 从仿真开始就追踪 sim_minutes，当接近 3/3 14:43（即 sim_min ≈ 3523）时开始布局
- 提前空驶至韶关附近等待（从广州 23.12,113.28 到韶关 24.81,113.58 约 190km，耗时约 190 分钟）
- 在 3/2 晚间或 3/3 凌晨出发，确保 14:43 前到达装货点附近
- query_cargo 时如果看到 cargo_id=240646，**立即无条件接单**

```python
def check_d009_special(state, candidates):
    for item in candidates:
        if item["cargo"].get("cargo_id") == "240646":
            return {"action": "take_order", "params": {"cargo_id": "240646"}}

    # 如果还没到上架时间但临近，预先空驶到韶关
    if state["sim_minutes"] < 3523 and state["sim_minutes"] > 3523 - 300:
        if haversine(state["position"], (24.81, 113.58)) > 5:
            return {"action": "reposition", "params": {"latitude": 24.81, "longitude": 113.58}}
    return None
```

### 4.2 D010 — 家事事件

这是全场最复杂的偏好，罚分无上限（每分钟 5 元 × 最长数千分钟 + 固定 9000）。必须精确执行：

**时间线**：
1. 3/10 10:00（sim_min=13200）前正常运营
2. 3/10 10:00 起：空驶到配偶位置 (23.21, 113.37)
3. 到达后原地 wait ≥ 10 分钟（"接上配偶"）
4. 空驶到老家 (23.19, 113.36)
5. 3/10 22:00 前必须到家（sim_min=13920）
6. 到家后持续 wait，直到 3/13 22:00（sim_min=18360）
7. 3/13 22:00 后恢复正常运营

```python
def handle_d010_family(state):
    sim_min = state["sim_minutes"]
    PICKUP_SPOUSE = (23.21, 113.37)
    HOME = (23.19, 113.36)

    # 阶段1：还没到事件时间，但需要提前往家的方向靠拢
    if sim_min < 13200 and sim_min > 13200 - 120:
        if haversine(state["position"], PICKUP_SPOUSE) > 2:
            return {"action": "reposition", "params": {"latitude": 23.21, "longitude": 113.37}}

    # 阶段2：事件激活，去接配偶
    if 13200 <= sim_min and not state.get("spouse_picked"):
        if haversine(state["position"], PICKUP_SPOUSE) > 1.0:
            return {"action": "reposition", "params": {"latitude": 23.21, "longitude": 113.37}}
        # 已到配偶位置，等待 10 分钟
        if state.get("spouse_wait_minutes", 0) < 10:
            return {"action": "wait", "params": {"duration_minutes": 10}}
        state["spouse_picked"] = True

    # 阶段3：去老家
    if state.get("spouse_picked") and not state.get("arrived_home"):
        if haversine(state["position"], HOME) > 1.0:
            return {"action": "reposition", "params": {"latitude": 23.19, "longitude": 113.36}}
        state["arrived_home"] = True

    # 阶段4：在家等待直到 3/13 22:00
    if state.get("arrived_home") and sim_min < 18360:
        remaining = min(18360 - sim_min, 480)  # 每次最多等 8 小时
        return {"action": "wait", "params": {"duration_minutes": remaining}}

    return None  # 事件结束，恢复正常
```

### 4.3 D010 — 月度到访目标点

(23.13, 113.26) 半径 1km 内，需要 5 个不同自然日到访。策略：
- D010 起始位置 (23.19, 113.36) 距目标约 11km，每次路过不远
- 在正常运营中优先选择卸货点靠近目标的货源
- 如果到第 25 天还不够 5 次，强制空驶过去打卡

---

## 五、各司机专属策略配置

```python
DRIVER_CONFIGS = {
    "D001": {
        "cost_per_km": 1.5,
        "min_continuous_rest_hours": 8,
        "rest_window": (22*60, 6*60),  # 建议 22:00-06:00 休息
        "forbidden_categories": {"化工塑料", "煤炭矿产"},
        "geo_fence": {"lat_min": 22.42, "lat_max": 22.89, "lng_min": 113.74, "lng_max": 114.66},
        "strategy": "深圳市内短途密集接单，每晚 22 点准时休息"
    },
    "D002": {
        "cost_per_km": 1.5,
        "min_continuous_rest_hours": 4,
        "monthly_off_days_required": 4,
        "forbidden_categories": {"蔬菜"},
        "strategy": "前 26 天接单，最后 4 天全休（或均匀分布周末休息）"
    },
    "D003": {
        "cost_per_km": 1.5,
        "quiet_window": (2*60, 5*60),  # 凌晨 2-5 点
        "max_monthly_deadhead_km": 100,
        "forbidden_zone": {"center": (23.30, 113.52), "radius_km": 20},
        "strategy": "严格控空驶，优先就近接单；时刻检查卸货点不落入禁区"
    },
    "D004": {
        "cost_per_km": 1.5,
        "max_daily_orders": 3,
        "quiet_window": (12*60, 13*60),  # 午休
        "first_order_deadline_hour": 12,
        "strategy": "早起接单（确保首单在12点前），每日严格≤3单，午休不动"
    },
    "D005": {
        "cost_per_km": 1.5,
        "quiet_window": (23*60, 30*60),  # 23:00-06:00（跨天）
        "max_haul_km": 100,
        "max_pickup_km": 90,
        "strategy": "短途专精，严格距离限制，晚间强制休息"
    },
    "D006": {
        "cost_per_km": 1.5,
        "min_continuous_rest_hours": 5,
        "monthly_off_days_required": 2,
        "forbidden_categories": {"鲜活水产品"},
        "max_haul_km": 150,
        "strategy": "中距离运输，每日 5h 休息，月末安排 2 天全休"
    },
    "D007": {
        "cost_per_km": 1.5,
        "quiet_window": (23*60, 28*60),  # 23:00-04:00
        "monthly_off_days_required": 1,
        "forbidden_categories": {"机械设备"},
        "max_haul_km": 180,
        "strategy": "覆盖范围最大，但夜间必须停工；安排 1 天全休"
    },
    "D008": {
        "cost_per_km": 1.5,
        "min_continuous_rest_hours": 4,  # 仅平日
        "rest_weekday_only": True,
        "monthly_off_days_required": 2,
        "forbidden_categories": {"食品饮料"},
        "max_pickup_km": 50,
        "strategy": "近距离接单（空驶≤50km），平日每天4h休息，月内2天全休"
    },
    "D009": {
        "cost_per_km": 1.5,
        "quiet_window": (23*60, 32*60),  # 23:00-08:00
        "must_return_home_by_23": True,
        "home": (23.12, 113.28),
        "forbidden_categories": {"快递快运搬家"},
        "special_cargo": {"cargo_id": "240646", "deadline_min": 3523 + 85},  # 上架到下架
        "strategy": "每天必须 23 点前回家，白天出发接单，运营半径受限；必接 240646"
    },
    "D010": {
        "cost_per_km": 1.5,
        "min_continuous_rest_hours": 3,
        "forbidden_categories": {"服饰纺织皮革"},  # 软约束，但有罚分
        "visit_target": (23.13, 113.26),
        "visit_days_required": 5,
        "family_event": {
            "start_min": 13200,       # 3/10 10:00
            "deadline_min": 13920,    # 3/10 22:00 前到家
            "end_min": 18360,         # 3/13 22:00
            "spouse_pos": (23.21, 113.37),
            "home_pos": (23.19, 113.36),
        },
        "strategy": "前 9 天正常运营并积累到访；3/10-3/13 处理家事；3/14 后恢复"
    },
}
```

---

## 六、Token 预算管理

复赛约束：每司机 500 万 Token，总时长 4 小时。

预估：
- 月度仿真 31 天 × 每天约 10-40 步决策 = 每司机约 300-1200 步
- 每步 Prompt ~800 token + Completion ~100 token = ~900 token/步
- 1200 步 × 900 = 108 万 Token（远低于 500 万限制）

**节省策略**：
1. 大部分步骤由 SchedulePlanner 和 RuleEngine 直接决定（不调 LLM）
2. 当 SchedulePlanner 返回强制动作时，**跳过 LLM 调用**，直接用固定 prompt 做一次廉价确认
3. 无货源时直接 wait 30-60 分钟，不调 LLM
4. 实际 LLM 调用可能只占 40% 的步骤

---

## 七、实现路径

### Phase 1：基础框架（第 1 周）
- 重写 `model_decision_service.py`，拆分为上述 5 个模块
- 实现 StateTracker + DriverConfig 加载
- 实现 SchedulePlanner 的强制休息/禁止时段逻辑

### Phase 2：规则引擎（第 2 周）
- 实现 RuleEngine 的品类/距离/地理围栏/时间窗过滤
- 实现 CargoScorer 的多维评分
- 压缩 LLM Prompt 至 Top-5 候选

### Phase 3：特殊事件（第 3 周）
- D009 熟货必接 + 预先空驶布局
- D010 家事事件完整状态机
- D010 月度到访打卡策略

### Phase 4：调优（第 4 周）
- 本地跑全量仿真对比收益
- 调整评分权重和休息时段选择
- Token 使用量分析和 prompt 精简
- 边界情况修复（月末接单超时、装货窗踩点等）

---

## 八、关键数据洞察

从 cargo_dataset 前几行可以看出：
- 价格单位是**分**（20301.15 分 = 203 元），经 API 查询时已转换为元
- 运输耗时较长（395-569 分钟 = 6.5-9.5 小时），单笔订单占据大量时间
- 货源覆盖广东省：深圳、广州、佛山、江门、惠州、肇庆、韶关、清远等
- 装货时间窗通常在白天（8:00-17:00）
- 品类多样：机械设备、水果、空包装、数码家电、农用物资等

**重要推论**：
- 每天有效接单次数可能只有 2-4 次（每笔 6-10 小时）
- D004 每日≤3单限制实际上不太容易触发（自然约束）
- 空驶成本 = 距离 × 1.5 元/km，空驶 50km = 75 元成本
- 一笔 200 元订单如果空驶 50km 去接，净利润只有 200 - (50+运距)×1.5

---

## 九、风险与应对

| 风险 | 应对 |
|------|------|
| LLM 返回非法 JSON | 解析失败时 fallback 为 wait 30 分钟 |
| 接单后发现超月末 | 环境会标记 income_eligible=False，不计收入但计成本，需预估避免 |
| 货源池临时清空 | wait 60 分钟后重试，避免空转 |
| D009 熟货被其他步骤错过窗口 | 从 3/2 晚就开始向韶关移动 |
| D010 家事期间误操作 | 状态机严格锁定，事件期间屏蔽所有非 wait 动作 |
| Token 超限 | 监控累计 token，接近阈值时退化为纯规则决策 |

---

## 十、R9 优化计划（规则修复 + 经验积累机制）

基于 R8.6 仿真结果（净收入 +200,368，罚分 -19,159），识别出两条优化路径：

### 10.1 规则层罚分修复（预期回收 ~12,100 元）

**A. D004 首单 12:00 前硬阻断**（~3,800 元）
- 问题：_work_mode 中缺少对首单 deadline 前的强制 WORK 保护，导致 D004 在午前被低优先级 rest/off-day 拦截
- 修复：在 _work_mode 开头加入首单 deadline 紧迫检测——已过 deadline 且无首单时阻止等待、降低接单阈值

**B. D010 到访目标提前触发**（~3,000 元）
- 问题：visit_target 触发条件过松（剩余天数 ≤ visits_needed + 2），月末才紧急空驶
- 修复：schedule_planner._handle_visit_target 中加入"均匀分布"检查——如果完成进度落后于预期天数进度，提前触发

**C. D003 凌晨 2-5 点硬阻断**（~1,400 元）
- 问题：D003 安静窗口 2:00-5:00 依赖偏好解析准确性，如解析遗漏则无保护
- 修复：_work_mode 中增加安静窗口二次验证——即使 SchedulePlanner 未拦截，若当前在安静窗口内仍返回 wait

**D. D001/D008 强制休息窗口**（~3,100 元）
- 问题：off-day 锁和 go-home 可能覆盖 daily_rest，导致休息时段被跳过
- 修复：已有逻辑较完善，优化 _handle_daily_rest 的 go_home 冲突保护精度

### 10.2 经验积累机制（来自论文启发）

**E. enhance_decision 注入 few-shot 历史决策**
- 来源：NeurIPS 2025 "Self-Generated In-Context Examples"
- 问题：当前 _build_decision_context 只传当前状态，无历史决策参考
- 修复：从 ExperienceTracker 提取"同时段同区域"的 top-3 历史决策（含结果），注入 prompt

**F. OPRO 缓冲区保留失败实验 + 违规反思**
- 来源：Reflexion (Shinn et al. 2023) + SLEA-RL
- 问题：ExperienceBuffer 只保留 top-10 高净效益实验，丢弃所有失败数据
- 修复：新增 _worst_buffer 保留 worst-3，daily_review prompt 同时展示好坏两极；违规反思注入

**G. wait 决策经验追踪**
- 来源：SLEA-RL step-level experience
- 问题：ExperienceTracker.record_decision 只记录 take_order，wait 决策无回溯
- 修复：新增 record_wait_decision 方法，记录等待时的时段/区域/后续是否获得更好货源
