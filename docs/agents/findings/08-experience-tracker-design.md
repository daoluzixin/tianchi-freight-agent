# 08 - ExperienceTracker 决策经验积累系统设计

> 日期：2026-05-15
> 基线：R8.6-clean (203,418元, commit de0b191)
> 目标：通过纯规则经验积累（零 Token）突破纯规则评分的 Pareto 前沿
> 理论来源：Self-Generated In-Context Examples (NeurIPS 2025) + SLEA-RL (2025)

---

## 一、问题诊断

R8.6-clean 的 203,418 元已是纯规则体系的天花板。瓶颈分析表明：

1. **CargoScorer 的 position_bonus 基于热点距离，缺乏时段感知**——同一个卸货点在早高峰和深夜的"下一单等待时间"差异巨大，但评分一视同仁。
2. **wait_value 估算依赖全局历史均分**——不区分时段和区域，在供给密度差异大的场景下误判严重。
3. **StrategyAdvisor 的 daily_review 只看聚合指标**——无法回答"哪类决策带来了高收益、哪类导致了损失"。

核心矛盾：系统**有观察数据但不积累经验**——每步决策后的结果（实际收入、下一单等待时长、卸货位置好坏）白白浪费了。

---

## 二、设计方案

### 2.1 核心思想

在仿真过程中自动积累 `(决策上下文, 实际结果)` 对，按 `(time_slot, region)` 聚类存储。决策时检索同类历史经验，用于校准 position_bonus 和 wait_value，让系统"越跑越聪明"。

**零 Token 实现**：全部用 dict 查表 + 统计聚合，不调用 LLM。

### 2.2 数据结构

```python
@dataclass
class DecisionExperience:
    """一条决策经验记录。"""
    # 上下文（决策时记录）
    time_slot: int          # 时段桶：0=深夜(0-6), 1=早高峰(6-11), 2=午间(11-14), 3=晚高峰(14-20), 4=晚间(20-24)
    region_key: tuple[int, int]  # 区域桶：(lat // 0.5, lng // 0.5)
    cargo_price: float
    pickup_km: float
    score_at_decision: float

    # 结果（延迟回填）
    actual_income: float = 0.0
    next_order_wait_minutes: float = 0.0  # 卸货后等了多久才接到下一单
    delivery_region_key: tuple[int, int] = (0, 0)

    # 元数据
    day: int = 0
    weight: float = 1.0       # 当前权重（受衰减影响）
    confirm_count: int = 1    # 被相似经验验证的次数
    settled: bool = False      # 结果是否已回填
```

### 2.3 经验聚类与检索

索引结构：`dict[(time_slot, region_key)] → list[DecisionExperience]`

检索逻辑：
- 精确匹配 `(time_slot, region_key)` 获取候选列表
- 按 weight 加权计算统计量：`avg_income`, `avg_next_wait`, `avg_score`
- **置信度门槛**：同一个桶内至少 3 条已落定经验才输出建议，否则返回"无经验"

### 2.4 自适应衰减策略

**核心改进**：不用统一的 0.9 日衰减，根据 `confirm_count`（经验被重复验证的次数）决定衰减速率：

| confirm_count | 衰减率 | 含义 | 10天后权重 |
|---------------|--------|------|-----------|
| >= 3 | 0.98 | 稳定模式（如"韶关早高峰总有好货"） | 0.82 |
| == 2 | 0.95 | 初步有规律 | 0.60 |
| <= 1 | 0.90 | 偶发观察 | 0.35 |

**confirm_count 更新规则**：新经验入库时，检查同 `(time_slot, region)` 是否已有结果方向一致（收入差 < 30%）的旧经验。如果有，旧经验的 `confirm_count += 1` 且权重重置为 1.0（被重新验证 = 刷新了有效期）。

### 2.5 冷启动保护

前 3 天（day < 3）不输出经验建议，只积累数据。避免在样本量极小时产生误导性的校准信号。

---

## 三、集成点

### 3.1 model_decision_service.py — 记录与回填

**记录**：在 `_rule_based_decision()` 或 `_try_llm_enhanced()` 做出 `take_order` 决策时，调用 `tracker.record_decision(state, scored_cargo)` 记录待确认经验。

**回填**：在 `decide()` 的 pending_action 补偿阶段，如果上一步是 `take_order` 且 accepted，调用 `tracker.settle_pending(driver_id, result, current_sim_minutes)` 回填实际结果。

**检索**：在 `_work_mode()` 入口处调用 `tracker.get_experience(time_slot, region)` 获取经验摘要，传递给 CargoScorer。

### 3.2 cargo_scorer.py — 经验驱动的 position_bonus 校准

在 `_enhanced_position_bonus()` 中，如果经验库对卸货位置的 `(time_slot, delivery_region)` 有足够数据：

```
experience_bonus = avg_income_at_delivery_region / normalization_factor
position_bonus = data_driven_bonus * 0.6 + experience_bonus * 0.4
```

经验 bonus 的权重在冷启动期（前 3 天）为 0，之后线性增长到 0.4。

### 3.3 strategy_advisor.py — daily_review 增强

在 `_build_review_context()` 中注入经验摘要：
- 昨天哪些 `(time_slot, region)` 组合带来了最高/最低收益
- 哪些区域的"下一单等待时间"异常长（可能需要避开）
- 哪些经验被多次验证（confirm_count >= 3）可作为稳定参考

这些信息作为 LLM daily_review 的额外上下文，帮助它做更有信息量的参数调整。

---

## 四、风险控制

1. **经验误导**：通过 confirm_count 门槛（>= 3）和冷启动保护（前 3 天不输出）双重防护。
2. **内存膨胀**：每个 `(time_slot, region)` 桶最多保留 20 条经验（按 weight 淘汰最低的）。10 个司机 × 5 时段 × ~20 区域 × 20 条 = ~20,000 条，内存可忽略。
3. **计算开销**：检索是 O(1) dict 查表 + O(20) 加权平均，每步 < 0.1ms。
4. **回归风险**：经验只用于**校准**（调整 position_bonus 和 wait_value 的幅度），不改变核心决策逻辑（RuleEngine 过滤 + CargoScorer 排序）。即使经验全部失效，系统退化为 R8.6-clean 的行为。

---

## 五、预期收益

| 改进点 | 机制 | 预期收益 |
|--------|------|----------|
| position_bonus 时段感知 | 卸货位置按时段区分好坏 | 减少"卸到荒区等 2 小时"的情况 |
| wait_value 区域感知 | 当前区域历史等待收益 | 高峰区少等多接，低谷区早走 |
| daily_review 经验注入 | LLM 看到细粒度收益分布 | 参数调整更精准 |
| 整体 | 系统从"无记忆"升级为"有经验" | 预计净收益 +3~8% |

---

## 六、实施清单

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | `agent/scoring/experience_tracker.py` | 新建核心模块 |
| 2 | `agent/core/model_decision_service.py` | 集成记录/回填/检索 |
| 3 | `agent/scoring/cargo_scorer.py` | 添加经验驱动的 position_bonus 校准 |
| 4 | `agent/strategy/strategy_advisor.py` | daily_review 注入经验摘要 |
| 5 | 仿真验证 | 运行 full simulation 对比 R8.6-clean |
