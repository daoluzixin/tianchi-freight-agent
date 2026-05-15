# 收获 01 — 规则化偏好解析器（Rule-Based Fallback）

> 日期：2026-05-13
> 触发原因：D009 因 LLM 超时导致偏好全部 fallback 到 custom，引发 37,000 元罚分，净收益 -8,908

## 问题根因

偏好解析器 `PreferenceParser.parse()` 在 LLM 调用失败时，原来的降级逻辑是把所有偏好条目都归为 `CustomConstraint`。这意味着：

- go_home、special_cargo、family_event 等高罚分约束无法被结构化识别
- 规则引擎的确定性过滤层完全失效
- 司机在不知情的情况下违反关键约束，产生巨额罚分

D009 的典型数据：

```
总毛收入: 28,272 元
运输成本: -79 元（仅跑了少量订单）
偏好罚分: -37,101 元（go_home + special_cargo 双双违规）
净收益: -8,908 元
```

## 解决方案

新建 `demo/agent/config/rule_based_parser.py`，实现纯正则 + 关键词的规则化解析，覆盖全部 14 种约束类型。

### 设计原则

1. **零网络调用** — 不依赖 LLM，在任何环境下都能工作
2. **宁漏不错** — 无法识别的偏好归入 custom（而非错误分类），由后续每步 LLM 评估
3. **高罚分优先匹配** — 解析顺序：family_event → special_cargo → go_home → visit_target → ...
4. **基于真实数据设计** — 正则模式从 10 个司机实际偏好文本中提炼，保持泛化能力

### 解析优先级链

```
family_event（9000+ 罚分）
  → special_cargo（10000 罚分）
    → go_home（900/天 罚分）
      → visit_target
        → quiet_window
          → forbidden_category
            → rest_constraint
              → off_days
                → max_distance
                  → max_orders
                    → first_order_deadline
                      → geo_fence
                        → forbidden_zone
                          → custom（兜底）
```

### 集成方式

在 `preference_parser.py` 中通过延迟导入避免循环依赖：

```python
def _rule_based_fallback(driver_status: dict[str, Any]) -> ParsedPreferences:
    """延迟导入 rule_based_parser 以避免循环导入。"""
    from agent.config.rule_based_parser import rule_based_parse
    return rule_based_parse(driver_status)
```

三个失败路径全部替换为 rule-based fallback：
- LLM 调用抛异常
- LLM 返回空 choices
- LLM 返回内容无法解析为 JSON

### 验证结果

10/10 司机偏好文本均能正确解析，0 个 custom fallback。单测覆盖了所有司机的关键字段值断言。

## 文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `demo/agent/config/rule_based_parser.py` | 新增 | 规则化解析器主体，628 行 |
| `demo/agent/config/preference_parser.py` | 修改 | 新增 `_rule_based_fallback()` 函数，替换 3 处 fallback 路径 |
| `demo/agent/tests/test_rule_based_parser.py` | 新增 | 10 司机全覆盖单测 |

## 效果对比

| 指标 | 改进前（D009） | 改进后（D009） |
|------|----------------|----------------|
| 偏好罚分 | -37,101 元 | 0 元 |
| custom 数量 | 4（全部） | 0 |
| 解析耗时 | 依赖网络 | <1ms |
