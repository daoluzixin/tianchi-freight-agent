# 02 — 决策流程详解

## 单步决策伪代码

```python
def decide(driver_id):
    # 0. 首步初始化
    if first_call:
        preferences_text = api.get_driver_status(driver_id)["preferences"]
        parsed = PreferenceParser.parse(preferences_text)  # LLM
        config = build_config_from_parsed(parsed)
        register_config(driver_id, config)
        state = StateTracker.init_from_history(api.query_decision_history())

    # 1. 更新状态
    status = api.get_driver_status(driver_id)
    state.update(status)

    # 2. 强制动作检查（最高优先级）
    mandatory = SchedulePlanner.check(state, config)
    if mandatory:
        return mandatory  # 直接返回，不调 LLM

    # 3. 查询货源
    candidates = api.query_cargo(state.position)

    # 4. 规则过滤
    filtered = RuleEngine.filter(candidates, state, config)

    # 5. 无货处理
    if not filtered:
        return wait_or_reposition(state, config)

    # 6. 评分排序
    scored = CargoScorer.rank(filtered, state, config)[:5]

    # 7. LLM 增强（可选）
    if TokenBudget.should_use_llm(state, scored):
        return StrategyAdvisor.enhance(state, scored, config)
    else:
        return take_top_scored(scored[0])
```

## 决策优先级栈

1. **SchedulePlanner 强制动作** — 不可覆盖
2. **特殊事件** — D009 熟货必接、D010 家事状态机
3. **RuleEngine 硬过滤** — 不合规的永远不选
4. **CargoScorer 排名** — 收益/效率/位置的综合打分
5. **StrategyAdvisor LLM** — 在前4层之后做最后判断

## 回退策略

| 异常场景 | 回退行为 |
|---------|---------|
| LLM API 超时/报错 | 使用 CargoScorer Top-1 直接接单 |
| LLM 返回非法 JSON | wait 30 分钟后重试 |
| 连续 3 次 LLM 失败 | 本司机剩余步骤全部纯规则 |
| 无货源 | wait（低谷）或 reposition（热区） |
| 接单失败（货源失效）| 仿真推进 1 分钟，下步重新决策 |

## 跨天逻辑

当 StateTracker 检测到跨天（current_day 变化）：
1. 重置 today_order_count
2. 重置 longest_rest_today
3. 触发 StrategyAdvisor.daily_review() — LLM 回顾昨日策略表现
