# 01 — 系统架构

## 设计哲学

**规则驱动的硬约束层 + LLM 驱动的软决策层。**

把能确定性计算的东西（时间窗检查、距离阈值、禁忌品类过滤）从 LLM 决策中剥离出来做前置过滤，LLM 只负责在合规候选中做"接哪个/是否等待/去哪空驶"的权衡判断。

## 五层架构

```
┌──────────────────────────────────────────────────┐
│            ModelDecisionService.decide()           │
├──────────────────────────────────────────────────┤
│ 1. StateTracker    — 维护累计状态（收益/里程/休息）│
│ 2. SchedulePlanner — 时间规划（强制休息/禁止时段）  │
│ 3. RuleEngine      — 偏好规则前置过滤             │
│ 4. CargoScorer     — 候选评分排序 Top-5           │
│ 5. StrategyAdvisor — LLM 增强决策 / 纯规则回退    │
└──────────────────────────────────────────────────┘
```

## 模块职责

| 模块 | 文件 | 职责 | LLM调用 |
|------|------|------|---------|
| PreferenceParser | `config/preference_parser.py` | 首步将偏好文本解析为结构化配置 | 1次/司机 |
| StateTracker | `core/state_tracker.py` | 进程内维护轻量状态字典 | 无 |
| SchedulePlanner | `core/schedule_planner.py` | 检查强制动作（休息/禁止时段/事件） | 无 |
| RuleEngine | `core/rule_engine.py` | 确定性过滤：品类/距离/地理围栏/时间窗 | 无 |
| CargoScorer | `scoring/cargo_scorer.py` | 多维度打分排序，取 Top-5 | 无 |
| SupplyPredictor | `scoring/supply_predictor.py` | 当前位置货源密度预测 | 无 |
| TokenBudgetManager | `strategy/token_budget.py` | 控制每步是否值得调 LLM | 无 |
| StrategyAdvisor | `strategy/strategy_advisor.py` | 每日策略回顾 + 高价值场景增强 | 选择性 |

## 数据流

```
SimulationApiPort (simkit 提供)
       │
       ├─ get_driver_status() → StateTracker 更新
       ├─ query_cargo()       → RuleEngine 过滤 → CargoScorer 打分
       └─ query_decision_history() → 首步重建状态
```

## 不变量（Invariants）

1. 每步决策耗时必须 < 30秒（避免超时）
2. 任何一步如果 LLM 失败，必须有规则回退路径
3. StateTracker 在进程内维护，不依赖 LLM 记忆
4. SchedulePlanner 的强制动作优先级高于一切——如果它说 wait，就 wait
