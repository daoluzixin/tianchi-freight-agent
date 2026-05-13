# 04 — Token 预算管理

## 约束条件

- 每司机 500 万 token/月
- 总仿真时长不超过 4 小时
- 月度仿真 31 天，每天约 10-40 步 = 每司机约 300-1200 步

## 预算分配

| 场景 | Token/次 | 频率 | 月度总量 |
|------|----------|------|---------|
| PreferenceParser（偏好解析）| ~3000 | 1次/司机 | 3K |
| StrategyAdvisor.daily_review | ~3000 | 31次/月 | ~93K |
| StrategyAdvisor.enhance_decision | ~1500 | ~40%步骤 | ~720K |
| Custom偏好LLM评估 | ~1000 | 选择性 | ~100K |
| **合计** | | | **~916K** |

远低于 500 万上限，留有充足余量应对重试和异常。

## TokenBudgetManager 策略

```python
def should_use_llm(state, scored_candidates):
    # 1. 预算耗尽 → 不调
    if remaining_budget < safety_margin:
        return False

    # 2. 强制场景 → 已被 SchedulePlanner 处理，不到这里

    # 3. 高价值场景 → 调
    #    - 多个候选分差 < 10%
    #    - 有 custom 偏好约束需要判断
    #    - 当日首次决策
    if candidates_score_gap < 0.1 or has_custom_constraint:
        return True

    # 4. 简单场景 → 纯规则
    #    - Top-1 远超 Top-2
    #    - 无货等待
    return False
```

## 节省策略

1. SchedulePlanner 返回强制动作时 **跳过 LLM**
2. 无货源时直接 wait，不调 LLM
3. Top-1 分数远超 Top-2 时直接选 Top-1
4. 连续无效步（wait/reposition）期间暂停 LLM 调用
5. 实际 LLM 调用只占约 40% 步骤

## 降级机制

当 LLM API 连续失败 N 次（默认 N=3）：
- 标记该司机为"纯规则模式"
- 剩余步骤全部用 CargoScorer Top-1 + SchedulePlanner
- 日志记录降级事件
- 不会崩溃，确保仿真完整运行
