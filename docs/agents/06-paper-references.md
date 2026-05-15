# 06 — 论文参考索引：泛用性改进方向

> 日期：2026-05-14
> 来源：HuggingFace Papers + arXiv
> 目标：为决策智能体的泛化能力提供学术支撑，按"可落地改进方向"组织

## 概述

本文档按六个改进方向整理相关论文，每篇标注与本赛题的关联点和落地启发。选择标准：方法需具备泛用性（不依赖特定数据集硬编码），且在 Token 预算和仿真时长约束内可实现。

---

## 方向一：序贯决策中的前瞻与价值估计

当前 CargoScorer 本质是贪心评分——只看"这一单值不值"，不考虑"接了之后下一步机会如何"。以下论文直接解决这一瓶颈。

### 1.1 FLARE: Why Reasoning Fails to Plan

- **链接**：https://huggingface.co/papers/2601.22311
- **发表**：2026-02
- **核心发现**：LLM 的逐步推理等价于贪心策略，在长时域规划中系统性失败。提出 Future-aware Lookahead with Reward Estimation，通过显式前瞻 + 价值传播 + 有限承诺机制让下游结果影响早期决策。LLaMA-8B + FLARE 超越 GPT-4o 的逐步推理。
- **赛题关联**：当前 `_work_mode` 的 CargoScorer Top-1 选择就是贪心。可以引入 SupplyPredictor 的"未来位置价值"作为前瞻信号——对 Top-5 候选做 1-step lookahead（模拟接单后的位置，查该位置下一时段的供给质量），重排序后再选。纯规则实现，零 Token 消耗。
- **落地难度**：中（需要 SupplyPredictor 的区域价值矩阵足够准确）

### 1.2 Step-Level Q-Value Models for LLM Agent Decision

- **链接**：https://huggingface.co/papers/2409.09345
- **发表**：2024-09（AAAI 收录）
- **核心方法**：用 MCTS 收集带步级 Q 值标注的决策轨迹，训练 Q-value 模型来指导动作选择。Q 模型可泛化到不同 LLM Agent。
- **赛题关联**：可以离线跑多轮仿真，记录每步决策后的月度净收益差值作为 Q 值标签，训练一个轻量统计回归模型来估算"在 (位置, 时间, 货源特征) 下接单的长期价值"。这是 SupplyPredictor 从"供给密度预测"升级为"决策价值预测"的理论基础。
- **落地难度**：高（需要离线仿真数据积累 + 回归模型训练）

### 1.3 Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents

- **链接**：https://huggingface.co/papers/2602.16699
- **发表**：2026-02
- **核心方法**：将信息检索等任务形式化为不确定性下的序贯决策问题，引入 Calibrate-Then-Act 框架，让 LLM 显式推理"何时停止探索（继续等待）、何时提交动作（立即接单）"的成本-不确定性权衡。
- **赛题关联**：当前 `wait_or_reposition` 逻辑非常粗糙（固定等 30 或 60 分钟）。可以用"货源到达率估计 × 等待期间成本"来计算等待的期望价值——如果当前时段的"平均每分钟可获净收益"高于"等待成本"，则继续等；否则立即接最优货或空驶。
- **落地难度**：中（需要 SupplyPredictor 提供时段级到达率）

---

## 方向二：自我改进与经验积累

500 万 Token 预算和 4 小时时限内，让智能体在仿真过程中"越跑越聪明"。

### 2.1 Self-Generated In-Context Examples for Sequential Decision

- **链接**：https://huggingface.co/papers/2505.00234
- **发表**：2025-05（NeurIPS 2025）
- **核心方法**：让 Agent 自动积累自身成功轨迹作为 few-shot 上下文示例，无需人工筛选。配合轨迹筛选和去冗余机制效果更优。
- **赛题关联**：StrategyAdvisor 的 `daily_review` 可以记录"今天哪些决策带来高收益 / 哪些导致了罚分"，将成功 pattern 作为下一天 LLM enhance 的 few-shot context。当前 daily_review 只做参数调整，可扩展为"经验库驱动的提示增强"。
- **落地难度**：低（在现有 ExperienceBuffer 基础上扩展为 decision-level 记录）

### 2.2 SLEA-RL: Step-Level Experience Augmented RL

- **链接**：https://huggingface.co/papers/2603.18079
- **发表**：2025
- **核心方法**：在每个决策步根据当前观察动态检索相关经验（而非任务开始时一次性加载），通过观察聚类 + 自演化经验库实现步级信用分配。
- **赛题关联**：在 StateTracker 中维护一个"决策经验索引"——按 `(时段桶, 区域桶, 货源密度桶)` 聚类，每次决策前检索同类历史决策的收益结果，辅助当前判断。纯规则实现（dict 查表），不消耗 Token。
- **落地难度**：中（需要设计合理的聚类 key 和索引结构）

### 2.3 Reflexion: Language Agents with Verbal Reinforcement Learning

- **链接**：https://huggingface.co/papers/2303.11366
- **发表**：2023（NeurIPS 2023）
- **核心方法**：通过语言反思（verbal reflection）实现无参数更新的 RL，Agent 将失败经验转化为文本记忆指导后续决策。
- **赛题关联**：当 take_order 失败（货源已失效）或出现偏好违规时，在 StateTracker 中记录一条"反思笔记"（如"D009 第 5 天接了远距离单导致回不了家"），后续遇到类似场景时提供给 StrategyAdvisor 作为负面示例。低成本自我纠错。
- **落地难度**：低（文本记录 + prompt 注入）

---

## 方向三：约束规划与成本感知搜索

偏好罚分远超单笔利润（D010 一次违规扣 3000+），必须从规划层保证约束满足。

### 3.1 CATS: Cost-Augmented Monte Carlo Tree Search for Planning

- **链接**：https://huggingface.co/papers/2505.14656
- **发表**：2025-05
- **核心方法**：将显式成本感知引入 MCTS 规划——紧约束下快速排除不可行方案，松约束下优化最小成本路径。
- **赛题关联**：已在 `05-optimization-plan.md` 中落地为 go_home 前瞻剪枝（规则 #9）和特殊货源路径保护（规则 #13）。核心思想是"在评分/过滤阶段就将约束成本前置"。
- **落地状态**：✅ 已实现

### 3.2 PlanGEN: Constraint Agent + Verification Agent Framework

- **链接**：https://huggingface.co/papers/2502.16111
- **发表**：2025-02（EMNLP 2025）
- **核心方法**：多智能体框架中引入"约束 Agent"做约束引导、"验证 Agent"做迭代校验、"选择 Agent"做自适应算法选择。通过约束引导的迭代验证改善规划。
- **赛题关联**：在 StrategyAdvisor.enhance 输出决策后，增加一步 constraint verification——用 RuleEngine 做二次校验，如果 LLM 建议的动作违反偏好则拒绝并选次优方案。当前架构 RuleEngine 在前置过滤，但 LLM 输出后缺乏后置校验。
- **落地难度**：低（复用现有 RuleEngine 的检查逻辑）

### 3.3 SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search

- **链接**：https://huggingface.co/papers/2512.23167
- **发表**：2024-12（AAAI 2026）
- **核心方法**：三智能体认知架构嵌入 MCTS 循环——Planner（提议动作）、Critic（反思策略合理性）、Simulator（推演结果）。将 MCTS 从暴力搜索转化为引导式自纠正推理。
- **赛题关联**：StrategyAdvisor 可以在"提议-验证"之间加入一步 Critic 推演——"如果选择 cargo_A，预计明天的位置是 X，该位置历史供给质量如何？"这一步可以纯规则实现（调用 SupplyPredictor），不需要额外 LLM 调用。
- **落地难度**：中

---

## 方向四：LLM 作为启发式优化器

让 LLM 不只做单步决策，还能迭代优化评分函数和策略参数，提升泛化能力。

### 4.1 OPRO: Large Language Models as Optimizers

- **链接**：https://huggingface.co/papers/2309.03409
- **发表**：2023-09（ICLR 2024）
- **核心方法**：用自然语言描述优化目标，让 LLM 迭代生成更优解。每轮把历史解和对应得分作为 context，LLM 在已有经验基础上提出新方案。
- **赛题关联**：已在 `05-optimization-plan.md` 中落地为 ExperienceBuffer + OPRO-style daily_review。LLM 看到历史参数-收益对后做小幅探索性调整。
- **落地状态**：✅ 已实现

### 4.2 EoH: Evolution of Heuristics

- **链接**：arXiv 2401.02051（ICML 2024）
- **发表**：2024-01
- **核心方法**：LLM + 进化算法协同演化启发式的"思想"和"代码"。在 bin packing、TSP 等组合优化问题上超越手工设计。核心是进化"想法（自然语言）→ 代码（可执行）"的双层结构。
- **赛题关联**：如果允许离线预处理，可用 EoH 思路让 LLM 自动演化 `cargo_scorer.py` 的评分公式——给定仿真的月度净收益作为 fitness，迭代几代找到更优的评分策略。需要在赛前离线完成。
- **落地难度**：高（需要多轮完整仿真 + 代码生成验证循环）

### 4.3 HeurAgenix: LLM-Driven Hyper-Heuristic

- **链接**：arXiv 2506.15196
- **发表**：2025
- **核心方法**：两阶段超级启发式——先用 LLM 演化一组候选启发式，再根据问题实例特征动态选择最优启发式。
- **赛题关联**：不同司机的最优策略差异大（D006 偏好休息、D009 需要回家、D010 需要打卡）。可以为不同"司机画像"维护不同的策略函数集，决策时根据当前状态动态切换。从"配置参数"升级为"配置策略函数"。
- **落地难度**：高（需要策略框架的重构）

---

## 方向五：多维偏好权衡与帕累托优化

赛题得分 = 净收益 - 偏好罚分，本质是多目标优化。

### 5.1 Panacea: Pareto Alignment via Preference Adaptation

- **链接**：https://huggingface.co/papers/2402.02030
- **发表**：2024-02（NeurIPS 2024）
- **核心方法**：单一模型学习整个 Pareto 前沿，用户在推理时通过低维偏好向量自由调节多个维度之间的权衡。无需针对不同偏好组合重新训练。
- **赛题关联**：CargoScorer 的多维打分权重目前是静态的。可以根据月度进程动态调整——月初侧重收益积累（price_weight 高），月末侧重偏好满足（penalty_risk_weight 高，宁漏不违规）。用一个"阶段偏好向量"控制各维度权重。
- **落地难度**：低（在 CargoScorer 中加一个 `phase_factor` 参数，按 sim_day / 31 线性调度）

### 5.2 TPO: Test-Time Preference Optimization

- **链接**：https://huggingface.co/papers/2501.12895
- **发表**：2025-01（ICML 2025）
- **核心方法**：无需训练，推理时通过文本批评反馈迭代优化 LLM 输出。将奖励信号转化为文本 critique，2 轮迭代即可赶超已对齐模型。
- **赛题关联**：StrategyAdvisor 可以在给出决策建议后，追加一轮 self-critique："这个决策是否可能导致今天偏好违规？如果是，给出替代方案。" 成本仅多约 500 token/次，但只在高风险场景（临近 deadline、临近禁区）触发。
- **落地难度**：低（在 enhance_decision 后加一步 critique prompt）

---

## 方向六：在线学习与动态路由

仿真月度内环境分布变化（工作日 vs 周末、白天 vs 夜晚），需要自适应策略。

### 6.1 LLM Bandit: Preference-Conditioned Dynamic Routing

- **链接**：https://huggingface.co/papers/2502.02743
- **发表**：2025-02
- **核心方法**：将模型/策略选择建模为多臂老虎机问题，通过偏好条件动态路由实现成本-精度权衡。方法可泛化到未见过的新模型。
- **赛题关联**：TokenBudgetManager 当前用简单规则决定"是否调 LLM"。可以引入 bandit 逻辑——跟踪历史 LLM 调用的增益（比纯规则多赚多少），某类场景增益低则自动降频，高增益场景自动升频。实现一个 UCB1 计数器即可。
- **落地难度**：中（需要定义"场景类型"和"增益度量"）

### 6.2 AgentGym-RL: Training LLM Agents for Long-Horizon Decision

- **链接**：https://huggingface.co/papers/2509.08755
- **发表**：2025-09
- **核心方法**：模块化、解耦架构的统一 RL 训练框架，支持在多样真实环境中训练 LLM Agent 进行多轮交互决策。提出 ScalingInter-RL 方法平衡探索-利用。
- **赛题关联**：提供了"如何在仿真环境中做在线 RL 而不爆 Token"的设计参考。核心启发是将"高层策略选择"与"低层动作执行"解耦——高层用 LLM 选策略（每日一次），低层用规则执行（每步）。这正是我们 5 层架构的理论基础。
- **落地难度**：已体现在架构设计中

---

## 方向七：车辆调度与 Fleet Management（领域参考）

以下非 HuggingFace Papers 但与赛题领域直接相关，提供问题建模参考。

### 7.1 End-to-End RL for Micro-View Order-Dispatching

- **链接**：ACM CIKM 2024 (DOI: 10.1145/3627673.3680013)
- **核心方法**：将行为预测和组合优化统一在序贯决策框架中，一阶段端到端 RL 方法做订单调度。
- **赛题关联**：证明了"在动态环境中把调度问题建模为 MDP 并用 RL 求解"的可行性。我们的 5 层架构本质是手工设计的策略网络（RuleEngine=mask, CargoScorer=value function, SchedulePlanner=constraint layer）。

### 7.2 Federated Multi-Agent Deep RL for Order Dispatching

- **链接**：Expert Systems with Applications, 2024
- **核心方法**：联邦多智能体 RL 优化订单调度，解决数据隐私和跨区域协调问题。
- **赛题关联**：赛题中 10 位司机独立决策但共享货源池，这与多智能体调度场景类似。目前各司机独立运行，如果有货源竞争（多人抢同一单），可能需要考虑协调机制。

---

## 落地优先级总结

| 优先级 | 方向 | 核心论文 | 改进效果 | Token成本 | 实现量 |
|--------|------|---------|---------|-----------|--------|
| P0 | 约束前瞻 | CATS (3.1) | 罚分 ↓50%+ | 零 | ✅ 已完成 |
| P0 | 参数自优化 | OPRO (4.1) | 泛化 ↑ | +0.3% | ✅ 已完成 |
| P1 | 阶段权重调度 | Panacea (5.1) | 月末违规 ↓ | 零 | 1h |
| P1 | 未来位置前瞻 | FLARE (1.1) | 净收益 ↑10% | 零 | 2h |
| P1 | 后置约束校验 | PlanGEN (3.2) | 安全网兜底 | 零 | 1h |
| P2 | 经验库提示增强 | Self-Gen ICL (2.1) | 后半月质量 ↑ | +1% | 2h |
| P2 | 等待价值估计 | CTA (1.3) | 低谷期收益 ↑ | 零 | 3h |
| P2 | LLM调用bandit | LLM Bandit (6.1) | Token效率 ↑ | 零 | 2h |
| P3 | 步级经验索引 | SLEA-RL (2.2) | 全程质量 ↑ | 零 | 4h |
| P3 | 自我批评机制 | TPO (5.2) | 高风险场景 ↓ | +0.5% | 1h |
| P4 | 离线评分演化 | EoH (4.2) | 评分公式泛化 | 离线 | 8h+ |
| P4 | 策略函数集 | HeurAgenix (4.3) | 多司机适配 | 离线 | 8h+ |

---

## 泛化性设计原则（从论文中提炼）

基于以上论文的共同设计模式，总结"泛化性"的工程实现原则：

1. **不硬编码决策规则，用数据驱动参数**（OPRO、EoH）——评分权重从"手工调参"变为"LLM 在经验上下文中搜索"。即使换了新司机、新数据分布，只要有几天运行数据就能自动适配。

2. **约束表达分离于策略逻辑**（CATS、PlanGEN）——偏好约束通过 PreferenceParser 从文本动态解析，RuleEngine 按结构化配置执行。新增一种偏好类型只需加一条解析规则和一条过滤逻辑，不影响主链路。

3. **贪心 → 前瞻的平滑过渡**（FLARE、CTA）——不需要完整的 MCTS，只做 1-step lookahead 就能显著改善长时域表现。关键是"预估动作后状态 → 查该状态的价值"这个两步结构。

4. **经验积累无需训练**（Reflexion、Self-Gen ICL）——仿真过程中用文本记录成功/失败 pattern，后续决策时作为 few-shot context 提供。类似 replay buffer 但用自然语言存储，LLM 可直接理解。

5. **多目标权重随进程动态调度**（Panacea）——月初探索、月中稳健、月末保守，用一个时间衰减因子控制"收益 vs 安全"的 tradeoff。不是单一策略打天下。

6. **失败快速反馈、不重蹈覆辙**（Reflexion、TPO）——每次违规或失败都留下一条文本记忆，后续同类场景自动规避。比依赖 LLM 的"通用推理"更可靠。
