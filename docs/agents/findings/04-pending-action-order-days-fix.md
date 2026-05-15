# 收获 04 — pending_action 补偿导致 order_days 丢失，off-day 误判

> 日期：2026-05-14
> 触发原因：D006 的 off-day 偏好（月内至少 2 整天不接单不空跑）始终只达成 1/2，罚 3,000 元

## 问题现象

D006 在多轮仿真中稳定出现 off-day 只有 1 天（Day 0），而 Day 6 的 off-day lock 始终不触发。诊断日志显示 `off_days={0, 5}`（done=2, needed=0），意味着系统认为已经有 2 天 off-day 了，但 Day 5 实际上有接单——它被误标为 off-day。

## 根因分析

问题涉及 `pending_action` 补偿机制和 `update_after_action` 的交互。共发现两个 bug，第二个是核心问题。

### Bug 1：pending_action 补偿时序（已在 v5 修复）

`init_from_status` 内部会调用 `_check_day_rollover`，但此时 pending_action 尚未处理，`order_days` 还没有包含上一步的接单信息。修复方式是增加 `skip_rollover` 参数，在有 pending_action 时跳过 rollover，补偿完成后再手动调用 `check_day_rollover`。

### Bug 2：pseudo_result 缺少 accepted 字段（本次修复核心）

pending_action 补偿时构造了一个最小化的 `pseudo_result`：

```python
pseudo_result = {"simulation_progress_minutes": state.sim_minutes}
```

但 `update_after_action` 中 take_order 的处理逻辑依赖 `accepted` 字段：

```python
elif action_name == "take_order":
    accepted = bool(result.get("accepted", False))  # pseudo_result 没有此字段 → False
    if accepted:
        ...
        state.order_days.add(state.current_day())   # 永远不会执行！
```

因此，通过 pending_action 补偿的 take_order **永远不会被记录到 order_days 中**。

### 影响链路

```
pending_action take_order → pseudo_result 无 accepted
  → order_days 不记录该天
    → 跨天 rollover 时该天被误判为 off-day
      → off_days 累积虚假计数
        → off_days_needed ≤ 0
          → 后续 off-day lock 永远不触发
            → 实际只有 Day 0 一天有效 off-day
```

## 修复方案

在 `model_decision_service.py` 的 pending_action 补偿逻辑中，为 take_order 添加 `accepted=True`：

```python
pseudo_result: dict[str, Any] = {"simulation_progress_minutes": state.sim_minutes}
if action_name == "take_order":
    pseudo_result["accepted"] = True
```

修改量：3 行代码。

### 设计考虑

pseudo_result 中缺少 `income`、`current_lat`、`pickup_deadhead_km` 等字段，这些会使用默认值（0 或当前状态值）。对于 pending_action 补偿来说，这些字段的精确值不重要——server 端已经完成了实际的状态更新，agent 内部的影子状态只需要保证 `order_days` 正确即可，因为它直接决定了 off-day 判定这一高罚分逻辑。

## 验证结果

### D006 单司机

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| off-day 达成 | 1/2（Day 0 唯一有效） | **2/2**（Day 0 + Day 6） |
| off-day 罚分 | 3,000 元 | **0 元** |
| 总偏好罚分 | 4,000 元 | 600 元（仅每日休息≥5h 违规） |
| 净收益 | 18,343 元 | 15,880 元 |
| 违规条数 | 2/4 | 1/4 |

净收益下降 2,463 元是因为 Day 6 现在真正执行了 off-day（全天不接单），减少了毛收入；但罚分节省了 3,400 元，综合看偏好满足度显著提升。

### 全量 10 司机

| 司机 | 净收益 | 罚分 | 违规 | off-day 相关 |
|------|--------|------|------|--------------|
| D001 | +12,576 | 300 | 1/3 | — |
| D002 | +21,864 | 1,000 | 1/3 | 4 天不接单需求：11 天达成 ✓ |
| D003 | +22,457 | 2,000 | 1/3 | — |
| D004 | +27,228 | 3,400 | 1/3 | — |
| D005 | +25,051 | 200 | 1/3 | — |
| D006 | +15,880 | 600 | 1/4 | **2 天 off-day 达成 ✓** |
| D007 | +23,093 | 500 | 1/4 | 1 天放空需求：1 天达成 ✓ |
| D008 | +18,175 | 4,900 | 3/4 | 2 天完全歇着需求：1 天达成 ✗（待查） |
| D009 | -9,293 | 25,300 | 2/3 | — |
| D010 | +12,809 | 7,510 | 4/4 | — |
| **合计** | **+169,840** | **45,710** | — | — |

D006 的 off-day 问题彻底解决。D008 的 off-day 仍只有 1/2 天，可能存在类似问题或调度策略不足，需单独排查。

## 文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `demo/agent/core/model_decision_service.py` | 修改 | pending_action 补偿的 pseudo_result 增加 `accepted=True`（3 行） |
| `demo/agent/core/state_tracker.py` | 修改 | 前序修复：`init_from_status` 增加 `skip_rollover` 参数 + 公开 `check_day_rollover` 方法 + 诊断日志 |
| `demo/agent/core/schedule_planner.py` | 修改 | 前序修复：off-day check 诊断日志 |

## 经验教训

1. **影子状态与真实状态的一致性**：agent 内部维护的 `order_days` 是 server 真实状态的影子，两者不同步时会产生级联错误。构造 pseudo_result 时必须确保所有影响决策逻辑的字段都被正确设置。

2. **诊断日志的价值**：这个 bug 纯靠代码审查很难发现（逻辑分散在 3 个文件中），最终是通过在 `_check_day_rollover` 和 `_handle_off_day_lock` 中添加诊断日志，打印 `order_days` 集合的实际内容，才精确定位到 Day 5 被误标的原因。

3. **跨步操作的补偿设计**：pending_action 机制本质上是在补偿"上一步的决策在本步才能观察到结果"的时序问题。这类补偿逻辑需要格外注意与原始逻辑的字段契约是否一致。
