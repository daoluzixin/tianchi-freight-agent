# 05 — 约束规划与启发式优化方案

> 日期：2026-05-14
> 目标：解决 D009 回家违规（16,200 元/月）和熟货错过（10,000 元）问题，同时提升全局接单收益
> 涉及模块：`schedule_planner.py`、`cargo_scorer.py`、`model_decision_service.py`、`strategy_advisor.py`

## 一、问题诊断

D009 在最近一次仿真中净收入 -10,351 元，两个核心瓶颈：

1. **回家违规 18 天 × 900 元 = 16,200 元**：`_handle_go_home` 的提前出发判断仅做"当前位置到家的行驶时间 vs 剩余时间"的单步校验，但未考虑"如果接了眼前这单，卸货后到家还来不来得及"。结果：白天接了一单远距离货，运完已过 23 点，回家 deadline 已破。

2. **熟货 240646 未接到，罚 10,000 元**：`_handle_special_cargo_approach` 正确触发了提前空驶，但 D009 因前一天接单跑远了（距韶关 >100km），等 reposition 到装货点附近时货源已下架或未出现在 query_cargo 返回结果中。

两个问题的共同根因是**缺乏前瞻性约束推演**：当前系统只在"约束即将被违反的当下"做出反应，而不是在"做决策时就排除会导致未来违约的选项"。

---

## 二、方向三：约束前瞻规划

### 2.1 设计思路（CATS 启发）

引用 CATS（Cost-Augmented Tree Search）的核心思想：在搜索/评分阶段就将约束成本前置，对紧约束做 early pruning，对松约束做 cost-aware scoring。

具体到本系统，不需要完整的 MCTS（单步动作空间有限且仿真不支持回溯），但可以做**一步前瞻（1-step lookahead）**：

```
评估一个 take_order 动作时：
  预估执行后的状态（位置、时间）
  → 检查该状态是否能满足接下来的硬约束
  → 不能满足 → 在 RuleEngine 阶段直接过滤掉
```

### 2.2 go_home 前瞻剪枝

**改动位置**：`rule_engine.py` 的 `filter_cargos()` 方法，增加一条过滤规则。

**算法**：

```python
def _violates_go_home_lookahead(self, cargo, state, config) -> bool:
    """接这单后，卸货完成时是否还能在 deadline 前到家？"""
    if not config.must_return_home or not config.home_pos:
        return False

    # 1. 预估接单后的完成时刻
    pickup_km = haversine_km(state.lat, state.lng, cargo.pickup_lat, cargo.pickup_lng)
    haul_km = haversine_km(cargo.pickup_lat, cargo.pickup_lng,
                           cargo.delivery_lat, cargo.delivery_lng)
    total_travel_min = (pickup_km + haul_km) / config.reposition_speed_kmpm

    # 加上装货等待和运输耗时（用 cargo.cost_time_minutes 如果有）
    cost_time = float(cargo.get("cost_time_minutes", total_travel_min))
    finish_sim_min = state.sim_minutes + cost_time

    # 2. 从卸货点到家的时间
    dist_home = haversine_km(cargo.delivery_lat, cargo.delivery_lng,
                             config.home_pos[0], config.home_pos[1])
    travel_home_min = dist_home / config.reposition_speed_kmpm

    # 3. 完成时刻 + 回家时间 > 当天 23:00？
    finish_day = int(finish_sim_min // 1440)
    deadline_min = finish_day * 1440 + config.home_deadline_hour * 60
    buffer = 30  # 安全缓冲

    return (finish_sim_min + travel_home_min + buffer) > deadline_min
```

**预期效果**：D009 每天白天接单时，远距离货（卸货后无法在 23 点前到家的）在过滤阶段就被剔除。18 天违规降为 0-2 天，挽回 14,400+ 元罚分。

---

### 2.3 特殊货源的路径约束（熟货保护）

**问题**：D009 在 3/2-3/3 需要在韶关附近接货 240646，但前一天接了远距离单导致来不及。

**方案**：在 `_work_mode` 中增加"特殊货源保护窗口"——当距特殊货源上架时间不到 N 小时时，只接卸货点靠近装货点方向的货源，拒绝反方向的货。

```python
def _is_cargo_compatible_with_special(self, cargo, state, config) -> bool:
    """在特殊货源保护窗口内，检查接单后是否还能及时赶到特殊货源装货点。"""
    sc = config.special_cargo
    if not sc or state.special_cargo_taken:
        return True

    time_to_special = sc.available_from_min - state.sim_minutes
    if time_to_special > 720:  # 距上架 >12h，不用保护
        return True

    # 接单后从卸货点到特殊货源装货点的距离
    dist_to_special = haversine_km(
        cargo.delivery_lat, cargo.delivery_lng,
        sc.pickup_lat, sc.pickup_lng)
    travel_to_special = dist_to_special / config.reposition_speed_kmpm

    # 接单执行时间
    cost_time = float(cargo.get("cost_time_minutes", 0))
    finish_min = state.sim_minutes + cost_time

    # 接完再去特殊装货点的到达时刻 vs 特殊货源上架时刻
    arrival_at_special = finish_min + travel_to_special
    # 要在上架后 60 分钟内到达（货源有效期有限）
    return arrival_at_special <= sc.available_from_min + 60
```

**改动位置**：`rule_engine.py` 新增过滤条件，或在 `_work_mode` 中 scored 之后做二次筛选。

---

### 2.4 接单后二次约束校验（PlanGEN 启发）

在 LLM 或规则决策给出 take_order 建议后，增加一步 constraint verification：

```python
# model_decision_service.py 中 _rule_based_decision 返回前
def _verify_action_constraints(self, action, state, config) -> bool:
    """验证拟执行动作是否满足所有前瞻约束。"""
    if action["action"] != "take_order":
        return True
    cargo = self._find_cargo_by_id(action["params"]["cargo_id"])
    if cargo is None:
        return True
    # 1. go_home 前瞻
    if self._rule_engine._violates_go_home_lookahead(cargo, state, config):
        return False
    # 2. 特殊货源保护
    if not self._is_cargo_compatible_with_special(cargo, state, config):
        return False
    # 3. off-day 保护（已有）
    return True
```

如果校验失败，降级到次优候选或 wait。这是"约束 Agent + 验证 Agent"双层架构的简化实现。

---

## 三、方向四：OPRO 风格 LLM 迭代优化评分参数

### 3.1 设计思路

当前 `StrategyParams` 的参数（`deadhead_penalty_factor`、`position_bonus_factor`、`time_cost_per_minute` 等）在 `daily_review` 时由 LLM 做单次调整。但 LLM 的单次判断受限于上下文中的信息量，往往给出平庸参数。

OPRO 的核心改进：**把历史参数组合和对应的收益结果作为 few-shot context 喂给 LLM，让它在已有经验基础上提出新方案**。

### 3.2 经验缓冲区设计

在 `StrategyAdvisor` 中新增一个 experience buffer：

```python
@dataclass
class ParameterExperiment:
    """一次参数实验的记录。"""
    day: int
    params_snapshot: dict[str, float]  # 当天使用的参数
    result: dict[str, float]           # 当天的收益指标
    # result = {income, deadhead_ratio, penalty, orders, efficiency}

class ExperienceBuffer:
    """OPRO 经验缓冲区：存储历史参数-收益对。"""
    def __init__(self, max_size: int = 10):
        self._buffer: list[ParameterExperiment] = []
        self._max_size = max_size

    def add(self, experiment: ParameterExperiment):
        self._buffer.append(experiment)
        # 按收益排序，保留 top-K（让 LLM 看到"好的参数长什么样"）
        self._buffer.sort(key=lambda x: x.result.get("income", 0), reverse=True)
        if len(self._buffer) > self._max_size:
            self._buffer = self._buffer[:self._max_size]

    def to_prompt_context(self) -> str:
        """生成 OPRO 风格的 few-shot context。"""
        lines = []
        for exp in self._buffer[:5]:  # 最多展示 top-5
            lines.append(
                f"Day {exp.day}: params={exp.params_snapshot} → "
                f"income={exp.result['income']:.0f}, "
                f"penalty={exp.result.get('penalty', 0):.0f}, "
                f"deadhead={exp.result.get('deadhead_ratio', 0):.2%}")
        return "\n".join(lines)
```

### 3.3 改进后的 daily_review prompt

```python
OPRO_REVIEW_PROMPT = """你是货运策略优化器。

## 历史实验记录（按收益排序）
{experience_context}

## 今日状态
{today_context}

## 任务
基于以上历史实验的经验，提出一组新的策略参数，目标是最大化净收益。
你可以看到哪些参数组合带来了高收益，尝试在那个方向上进一步优化。
也可以尝试与历史不同的探索方向。

输出 JSON：
{param_schema}
"""
```

### 3.4 Token 预算兼容

OPRO 风格的 prompt 比原来多约 500 token（经验 context），但调用频率不变（每天一次）。31 天增加约 15,500 token，在 500 万预算中占比 0.3%，完全可承受。

### 3.5 离线预训练（赛前优化）

更激进的做法：赛前本地跑 20 轮仿真，每轮用不同初始参数，记录结果。用 EoH（Evolution of Heuristics）思路让 LLM 在这些结果基础上迭代优化 CargoScorer 的评分公式本身（不仅是参数）。但这需要多次完整仿真，时间成本较高，作为 Phase 2 考虑。

---

## 四、实施优先级与完成状态

| 优先级 | 改动 | 预期收益 | 状态 | 实际改动位置 |
|--------|------|----------|------|-------------|
| P0 | go_home 前瞻剪枝 | 挽回 14,400+ 元/月 | ✅ 已实现 | `rule_engine.py` 规则 #9 |
| P0.5 | go_home 时间窗口保护 | 防止晚间接单导致违规 | ✅ 已实现 | `model_decision_service.py` go_home guard |
| P0.6 | 安静时段不空驶修复 | 避免双重违规 | ✅ 已实现 | `schedule_planner.py` _handle_go_home |
| P0.7 | 提前出发安全余量增加 | +10min 保护 | ✅ 已实现 | `schedule_planner.py` +40min buffer |
| P1 | 特殊货源路径保护 | 挽回 10,000 元 | ✅ 已实现 | `rule_engine.py` 规则 #13 |
| P1.5 | 特殊货源提前空驶增强 | 确保 190km 也能赶上 | ✅ 已实现 | `schedule_planner.py` 动态提前窗口 |
| P2 | 接单后约束校验 | 兜底安全网 | 待实现 | — |
| P3 | OPRO 经验缓冲区 | 提升 5-15% 净收益 | ✅ 已实现 | `strategy_advisor.py` ExperienceBuffer |

### 已完成改动清单

**1. `rule_engine.py`** — 规则 #9（go_home 前瞻剪枝）增强：
- 加入 `scan_cost = 10` 分钟仿真时间补偿
- 回家行驶时间用 `math.ceil()` 模拟仿真引擎取整
- 增加 30 分钟安全余量
- **关键修复**：以卸货完成所在那天计算 deadline（`delivery_day`），而非固定用 `current_day`，正确处理跨天订单

**2. `rule_engine.py`** — 新增规则 #13（特殊货源路径保护）：
- 在特殊货源上架前 4 小时（240 分钟）内激活
- 计算"接完此单 → 卸货后 → 空驶到特殊取货点"的到达时间
- 如果到达时间超过上架时间 + 60 分钟容忍窗口，则过滤掉

**3. `model_decision_service.py`** — go_home 时间窗口保护：
- 在 `_work_mode()` 入口处新增保护逻辑
- 计算"当前位置直接回家所需时间 + 40 分钟余量"
- 如果剩余时间不够"做一单 + 回家"（min_order_time=60min），停止接单
- 短等待 5 分钟让 schedule_planner 的 GO_HOME 接管

**4. `schedule_planner.py`** — 安静时段行为修复：
- **修复 bug**：进入 quiet window（23:00 后）但不在家时，不再发 GO_HOME（reposition 在 quiet window 内执行会被评测判为违规）
- 改为就地休息，今日违规已不可避免，但避免无意义的空驶消耗

**5. `schedule_planner.py`** — 提前出发增强：
- 安全余量从 30min 增加到 40min（`travel_min + 40 >= time_to_deadline`）
- 覆盖 scan_cost 10min + ceil 取整 + 路径偏差

**6. `schedule_planner.py`** — 特殊货源提前空驶增强：
- 动态提前窗口：>100km 提前 4h, >50km 提前 2h（之前固定 2h 且只对 >50km）
- priority 从 75 提高到 78，减少被 daily_rest 抢占的风险

**7. `strategy_advisor.py`** — OPRO 经验缓冲区（方向四核心）：
- 新增 `ParameterExperiment` dataclass：存储单日参数快照 + 收益结果
- 新增 `ExperienceBuffer` 类：Top-K 经验管理，按净效益排序，max_size=10
- `record_daily_stats()` 增加自动记录：每天跨天时将当日参数+结果存入 buffer
- `daily_review()` 注入 OPRO context：top-5 历史实验作为 few-shot 加入 LLM prompt
- System prompt 增加第 11 条指令：要求 LLM 参考历史最优组合做小幅探索性调整
- Token 增量：每次 daily_review 多约 500 token（经验 context），31 天增量 ≈ 15,500 token，占预算 0.3%

---

## 五、与面试叙事的桥接

这些改进对应的算法故事线：

- **方向三** → "我把约束满足问题建模为 lookahead search with cost augmentation，在评分阶段就剪枝不可行方案，而不是事后修补。" 可以类比 RLHF 中 reward model 对有害输出的前置过滤（Constitutional AI 思路）。

- **方向四** → "我用 LLM 作为黑盒优化器迭代搜索超参空间，类似 OPRO/EoH 的思路。Experience buffer 的设计类似 replay buffer 在 RL 中的角色——让优化器看到历史轨迹，做更有信息量的搜索。" 直接关联 RLHF pipeline 中的 reward shaping 和 hyperparameter search。

两者结合，讲的是"确定性约束保证 + 数据驱动参数优化"的混合架构，这是 Agent 系统设计中的通用范式。
