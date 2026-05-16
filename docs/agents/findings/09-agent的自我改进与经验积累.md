# 09 — Agent 的自我改进与经验积累：论文与开源项目索引

> 日期：2026-05-16
> 来源：HuggingFace Papers + arXiv + GitHub
> 目标：在 500 万 Token 预算和 4 小时时限内，让智能体在仿真过程中"越跑越聪明"

## 背景

R10-A-v2 实验表明：将经验注入从 top-1 改为 top-3 + 置信度排序后，全量 10 司机跑出的总净收益仍然是 202,959，与 R10 完全一致。这说明"给 LLM 多塞几条经验"这种朴素做法的边际收益接近于零——真正的瓶颈不在经验的数量，而在经验的质量、注入时机和表达形式。

本文档系统梳理"自我改进与经验积累"方向的最新论文和开源项目，按照与赛题的关联度和落地难度组织，为下一步改进提供学术支撑。

---

## 一、已有基础（06-paper-references.md 中已收录）

以下三篇已在主论文索引中记录，这里仅做简要回顾：

### 1.1 Self-Generated In-Context Examples for Sequential Decision

- **链接**：https://huggingface.co/papers/2505.00234（NeurIPS 2025）
- **核心**：Agent 自动积累成功轨迹作为 few-shot 示例，配合轨迹筛选和去冗余。
- **现状**：ExperienceTracker 已实现 decision-level 记录，但尚未做轨迹筛选和去冗余。

### 1.2 SLEA-RL: Step-Level Experience Augmented RL

- **链接**：https://huggingface.co/papers/2603.18079（2025）
- **核心**：每个决策步根据当前观察动态检索相关经验，通过观察聚类实现步级信用分配。
- **现状**：ExperienceTracker 按 `(time_slot, region)` 聚类查表，已部分实现。

### 1.3 Reflexion: Language Agents with Verbal Reinforcement Learning

- **链接**：https://huggingface.co/papers/2303.11366（NeurIPS 2023）
- **核心**：失败经验转化为文本记忆，指导后续决策。无参数更新的 verbal RL。
- **现状**：daily_review 有参数调整，但尚未做"失败反思笔记"的持久化和注入。
- **开源**：[noahshinn/reflexion](https://github.com/noahshinn/reflexion)

---

## 二、新发现论文（按落地难度排序）

### 2.1 ExpWeaver: Rethinking Experience Utilization in Self-Evolving Agents

- **链接**：https://arxiv.org/abs/2605.07164
- **发表**：2025/2026
- **核心方法**：指出现有 self-evolving agent 研究关注"经验怎么构建和存储"，但忽略了"经验该什么时候用、怎么用"。提出动态经验注入策略——不是每次都注入经验，而是根据当前任务的不确定性和经验的相关性动态决定是否注入。
- **赛题关联**：这正好解释了 R10-A-v2 top-3 注入没效果的原因——很多场景下规则层已经够用，强行注入经验反而是噪声。可以加一个"经验注入门控"：只在规则层打分模糊（top-1 和 top-2 分差很小）时才注入历史经验让 LLM 参考。
- **落地难度**：低（在 `_build_decision_context` 中加一个分差阈值判断）
- **落地时间**：0.5h

### 2.2 ReflAct: World-Grounded Decision Making via Goal-State Reflection

- **链接**：https://huggingface.co/papers/2505.15182
- **发表**：2025-05（EMNLP 2025）
- **核心方法**：改进 ReAct 框架，在每次 action 之前增加一个显式 reflection 步骤——Agent 先反思"我当前的状态是什么，离目标还差多远"，然后再决策。在 ALFWorld 上比 ReAct 提升 27.7%。
- **赛题关联**：StrategyAdvisor.enhance 目前直接给 LLM 状态 + 候选货源就让它决策。可以在 prompt 中增加一个显式的反思模板："当前 D003 已行驶 X 天，距离月底还有 Y 天，累计收益 Z，剩余目标 W，偏好约束是..."，让 LLM 先输出反思再给决策。只需改 prompt 模板。
- **落地难度**：极低（纯 prompt 工程，+200 token/次）
- **落地时间**：0.5h

### 2.3 Contextual Experience Replay (CER)

- **链接**：https://huggingface.co/papers/2506.06698
- **发表**：2025-06（ACL 2025）
- **核心方法**：Training-free 框架，将 Agent 过去的交互经验累积并合成到一个动态记忆缓冲区中，每次新决策前根据当前观察检索相关经验作为 context 注入。关键机制是"经验合成压缩"——不存原始轨迹，而是把多次同类经验压缩成一条摘要型经验。在 WebArena 上相对 GPT-4o baseline 提升 51%。
- **赛题关联**：当前 ExperienceTracker 存的是 best cargo detail（原始数据），可以参考 CER 的做法，将同一个 `(time_slot, region)` 桶下的多次经验合成为一段文本摘要（如"该时段该区域，净收益前3的货源平均净收益X元，共同特征是Y"），注入 prompt 时更精炼，节省 Token。
- **落地难度**：低（在 ExperienceTracker 中增加 `synthesize_summary()` 方法）
- **落地时间**：1h

### 2.4 MemR³: Memory Retrieval via Reflective Reasoning

- **链接**：https://huggingface.co/papers/2512.20237
- **发表**：2024-12
- **核心方法**：将记忆检索从被动的一次性操作变为主动的迭代过程。Router 在 Retrieve-Reflect-Answer 三个节点之间动态切换，维护一个"已知证据-缺失信息"的 gap tracker，直到信息充足才输出。
- **赛题关联**：经验查询可以增加"二次检索"——如果第一次检索结果的置信度都很低，自动扩大搜索范围（相邻时段/相邻区域）。纯规则实现，不需要 LLM 参与。
- **落地难度**：低（在 `query_top_experiences` 中加 fallback 逻辑）
- **落地时间**：1h
- **开源**：[Leagein/memr3](https://github.com/Leagein/memr3)

### 2.5 Prospector: Improving LLM Agents with Self-Asking and Trajectory Ranking

- **链接**：https://aclanthology.org/2024.findings-emnlp.879/
- **发表**：2024（EMNLP 2024 Findings）
- **核心方法**：双 LLM 架构——Actor 在 few-shot 示例中穿插"自问"步骤（目标检查、进度检查）以产生更合理的动作；Critic 对候选轨迹做排名，从环境反馈中挑选最优轨迹作为未来的 few-shot 示例。
- **赛题关联**：StrategyAdvisor 的 enhance prompt 可以加入"自问"机制——在给出决策前先问自己"这个选择是否符合偏好？是否影响明天回家？"。轨迹排名思路可以用于 daily_review：将当天的决策轨迹按收益排序，把最优轨迹存入经验库。
- **落地难度**：低（prompt 模板改造）
- **落地时间**：1h

### 2.6 EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle

- **链接**：https://huggingface.co/papers/2510.16079
- **发表**：2025-10
- **核心方法**：经验闭环生命周期——Agent 在线执行 → 离线将轨迹自蒸馏为抽象的"战略原则"（strategic principles）→ 语义去重 + 动态打分 → 在线检索原则指导新决策。存的不是具体数据，而是"原则"级别的抽象（如"长距离单中取货空驶超过100km的单子通常净收益为负"）。
- **赛题关联**：`daily_review` 目前只做参数微调，可以扩展为"原则提炼"——每天结束后让 LLM 从当天所有决策中提炼 2-3 条策略原则，存到 StateTracker，第二天注入 system prompt。比存具体货源数据更 Token 高效。
- **落地难度**：中低（扩展 daily_review 的输出格式）
- **落地时间**：2h
- **开源**：[KnowledgeXLab/EvolveR](https://github.com/KnowledgeXLab/EvolveR)

### 2.7 MUSE: Learning on the Job — Experience-Driven Self-Evolving Agent

- **链接**：https://huggingface.co/papers/2510.08002
- **发表**：2025-10
- **核心方法**：分层记忆模块 + Plan-Execute-Reflect-Memorize 闭环。记忆分三层：Episodic Memory（具体事件）、Semantic Memory（抽象规律）、Procedural Memory（操作流程）。Agent 执行后通过 Reflect Agent 反思，将经验分层存储。
- **赛题关联**：映射到我们的系统——ExperienceTracker 存 Episodic（具体接单记录），daily_review 提炼 Semantic（"D003 偏好远距离单但实际近距离单收益更稳"），SchedulePlanner 的规则相当于 Procedural。目前缺 Semantic 层——daily_review 应该产出文本级别的经验总结并持久化。
- **落地难度**：中（需要在 StateTracker 中增加 semantic memory 存储）
- **落地时间**：4h
- **开源**：[KnowledgeXLab/MUSE](https://github.com/KnowledgeXLab/MUSE)

### 2.8 GenericAgent: Token-Efficient Self-Evolving LLM Agent

- **链接**：https://huggingface.co/papers/2604.17091
- **发表**：2026-04
- **核心方法**：核心原则是"上下文信息密度最大化"——在有限 context window 中塞最有价值的信息。通过分层记忆 + 技能树自动生长 + context 压缩，用 1/6 的 Token 消耗达到同等性能。
- **赛题关联**：直接对应 Token 预算约束（500万/司机/月）。目前 enhance 调用时塞了大量原始数据进 prompt，可以参考 GA 的做法——对历史经验做信息密度排序，只注入"信息密度最高"的内容。与 2.3 CER 的经验压缩配合使用。
- **落地难度**：中（需要设计信息密度评估逻辑）
- **落地时间**：3h
- **开源**：[lsdefine/GenericAgent](https://github.com/lsdefine/GenericAgent)

### 2.9 Memento: Memory-Augmented LLM Agents via Episodic Case-Based Reasoning

- **链接**：https://arxiv.org/abs/2508.16153
- **发表**：2025-08
- **核心方法**：将持续学习形式化为 Memory-Augmented MDP（M-MDP），用 episodic memory 存储历史任务轨迹，通过 case-based reasoning（案例推理）检索相似案例指导新决策。不需要微调 LLM。
- **赛题关联**：ExperienceTracker 目前按 `(time_slot, region)` 做精确匹配，但 Memento 的启示是可以做模糊的 case-based matching——如果没有完全匹配的经验，检索"最相似"的案例（如相邻时段、相邻区域的经验），解决经验稀疏问题。
- **落地难度**：中（需要在 `query_top_experiences` 中增加 fallback 邻域搜索）
- **落地时间**：3h
- **开源**：[Memento-Teams/Memento](https://github.com/Memento-Teams/Memento)

### 2.10 RAGEN: Understanding Self-Evolution via Multi-Turn RL（警示性参考）

- **链接**：https://huggingface.co/papers/2504.20073
- **发表**：2025-04
- **核心方法**：提出 StarPO（State-Thinking-Actions-Reward Policy Optimization）通用框架做轨迹级 Agent RL。核心发现：多轮自我强化并不等同于真正的自我进化，纯结果导向奖励存在"推理消退"现象，需要过程监督（process supervision）。
- **赛题关联**：这是一个**警示性参考**——说明仅靠 daily_review 的结果级反馈（月度净收益）做参数调整可能不够，需要步级（单笔决策级）的反馈信号。这支持 1.2 SLEA-RL 的步级经验索引思路。
- **落地难度**：参考性（不直接落地，但影响设计决策）
- **开源**：[mll-lab-nu/RAGEN](https://github.com/mll-lab-nu/RAGEN)

---

## 三、开源项目索引

| 项目 | 说明 | 赛题相关度 |
|------|------|-----------|
| [noahshinn/reflexion](https://github.com/noahshinn/reflexion) | Reflexion 官方实现，verbal RL | 高——反思机制直接可参考 |
| [KnowledgeXLab/EvolveR](https://github.com/KnowledgeXLab/EvolveR) | 经验闭环 + 战略原则蒸馏 | 高——原则提炼可直接用 |
| [KnowledgeXLab/MUSE](https://github.com/KnowledgeXLab/MUSE) | 分层记忆 + Plan-Execute-Reflect | 高——分层记忆架构参考 |
| [lsdefine/GenericAgent](https://github.com/lsdefine/GenericAgent) | Token 高效的 self-evolving agent | 中——Token 优化思路参考 |
| [Memento-Teams/Memento](https://github.com/Memento-Teams/Memento) | episodic memory + case-based reasoning | 中——模糊匹配思路参考 |
| [Leagein/memr3](https://github.com/Leagein/memr3) | 反思式闭环记忆检索 | 中——二次检索逻辑参考 |
| [mll-lab-nu/RAGEN](https://github.com/mll-lab-nu/RAGEN) | 多轮 RL 训练评估框架 | 低——理论警示参考 |
| [XMUDeepLIT/Awesome-Self-Evolving-Agents](https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents) | 最全综述 + paper list | 工具——查阅索引 |
| [Tencent/SelfEvolvingAgent](https://github.com/Tencent/SelfEvolvingAgent) | 腾讯 AI Lab self-evolving agent 合集 | 中——工业界实践参考 |
| [modelscope/AgentEvolver](https://github.com/modelscope/AgentEvolver) | 阿里 AgentEvolver，高效自进化 | 中——自进化框架参考 |
| [MemTensor/MemRL](https://github.com/MemTensor/MemRL) | 基于 episodic memory 的运行时 RL | 中——运行时强化学习思路 |

---

## 四、落地优先级

| 优先级 | 改进点 | 论文来源 | 改进效果 | Token 成本 | 工时 |
|--------|--------|---------|---------|-----------|------|
| P1 | 经验注入门控（分差阈值） | ExpWeaver (2.1) | 消除噪声注入 | 零 | 0.5h |
| P1 | 决策前反思模板 | ReflAct (2.2) | LLM 决策对齐 ↑ | +200tok/次 | 0.5h |
| P2 | 经验合成压缩 | CER (2.3) | Token 效率 ↑ | 减少 | 1h |
| P2 | 经验邻域检索（二次 fallback） | MemR³ (2.4) | 经验覆盖率 ↑ | 零 | 1h |
| P2 | 决策前自问机制 | Prospector (2.5) | 偏好违规 ↓ | +100tok/次 | 1h |
| P2 | 策略原则提炼（daily_review 升级） | EvolveR (2.6) | 后半月质量 ↑ | +0.5% | 2h |
| P3 | 分层记忆架构（Semantic 层） | MUSE (2.7) | 全程质量 ↑ | +1% | 4h |
| P3 | 上下文信息密度优化 | GenericAgent (2.8) | Token 效率 ↑ | 减少 | 3h |
| P3 | 模糊案例匹配 | Memento (2.9) | 稀疏场景 ↑ | 零 | 3h |

---

## 五、设计原则提炼

基于以上论文的共同模式，总结经验积累机制的工程设计原则：

1. **经验存"原则"而非"数据"**（EvolveR、MUSE）——具体的货源数据（价格、距离、区域）有时效性且占 Token，抽象的策略原则（"夜间该区域长途单净收益显著优于短途"）更 Token 高效且泛化性更强。daily_review 的产出应从"参数调整"升级为"原则提炼"。

2. **经验注入需要门控，不是越多越好**（ExpWeaver）——R10-A-v2 top-3 注入无效果的实验已验证这一点。规则层已能处理的明确场景，注入经验是噪声；只在规则层"拿不准"时（候选分差小、场景罕见）才触发经验增强。

3. **决策前显式反思优于直接行动**（ReflAct、Prospector）——在 prompt 中加入"先反思当前状态与目标的差距，再给出决策"的模板，让 LLM 的推理锚定在真实状态上，避免 ReAct 式的"想到哪走到哪"。成本极低（+200 token），收益是决策一致性显著提升。

4. **记忆检索应支持模糊降级**（MemR³、Memento）——精确匹配 `(time_slot, region)` 经常 miss，应支持"先精确、再邻域、再全局"的三级降级，确保每次决策都有经验参考，哪怕是弱相关的。

5. **经验压缩优于经验堆积**（CER、GenericAgent）——在有限 context window 中，多条原始经验不如一条压缩摘要。ExperienceTracker 应提供 `synthesize_summary()` 而非逐条返回。

6. **步级反馈优于结果级反馈**（RAGEN 警示、SLEA-RL）——仅靠"月度净收益"做反馈太粗，容易导致"推理消退"。应在单笔决策层面记录"这笔接单的净收益是否达到同时段同区域的均值"，提供更精细的信用分配。

7. **经验积累无需训练**（Reflexion、CER、EvolveR）——所有方法都是 training-free，纯粹通过 prompt context 的组织和管理实现自我改进，完美适配赛题的"只有 API 调用，不能微调"约束。
