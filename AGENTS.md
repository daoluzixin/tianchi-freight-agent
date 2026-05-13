# AGENTS.md — 项目导航地图

> 本文件是 AI Agent（包括 CatDesk/CatPaw）理解本项目的唯一入口。
> 修改代码前必须先读此文件，了解项目边界和约定。

## 项目定位

天池 Agent 开发大赛 — **司机找货连续决策仿真赛**。
开发一个货运智能体，在模拟月度周期内对 10 位司机做接单/休息/空驶决策，目标是最大化净收益并满足个性化偏好。

## 目录结构

```
.
├── AGENTS.md              ← 你在这里
├── docs/                  ← 赛题文档（只读参考，不修改）
│   ├── 01-赛题详情.md
│   ├── 02-数据说明.md
│   ├── 03-评测规则.md
│   └── ...
├── docs/agents/           ← Harness 设计文档（决策沉淀）
│   ├── 01-architecture.md
│   ├── 02-decision-flow.md
│   ├── 03-preference-rules.md
│   └── 04-token-budget.md
├── demo/
│   ├── agent/             ← 【核心】选手决策实现（改这里）
│   │   ├── config/        ← 偏好解析 & 司机配置
│   │   ├── core/          ← 决策主链路（model_decision_service）
│   │   ├── scoring/       ← 货源评分 & 供给预测
│   │   ├── strategy/      ← LLM 策略增强 & Token 预算
│   │   └── tests/         ← 单测
│   ├── simkit/            ← 仿真规则引擎（只读，勿改）
│   ├── server/            ← 评测编排（只读，勿改）
│   │   ├── bench/         ← 仿真主循环
│   │   ├── config/        ← 运行配置
│   │   └── data/          ← 原始数据（Agent 禁止直读）
│   ├── results/           ← 仿真产出（git 不追踪）
│   ├── calc_monthly_income.py  ← 收益计算脚本
│   └── run_simulation.py       ← 快速运行入口
├── .catpaw/rules/         ← Agent 编码规则约束
├── scripts/               ← 自动化脚本（lint、评测）
└── .gitignore
```

## 关键约束（红线）

1. **禁止直读数据文件** — Agent 代码不得 `open()` 读取 `server/data/` 下任何文件，必须通过 `SimulationApiPort` 接口获取信息。
2. **不动 simkit / server** — 这两个包由赛方维护，修改会导致提交失败。
3. **Token 预算** — 每司机 500 万 token/月，总时长 4 小时。
4. **偏好零违规优先** — 偏好罚分远超单笔利润（D010 家事违规扣 9000+），策略设计必须"宁漏接不违规"。
5. **无硬编码** — 不得对特定 driver_id 写 if/else；配置从运行时偏好文本解析。

## 修改规范

### 允许修改的文件
- `demo/agent/**` — 所有决策逻辑
- `demo/agent/STRATEGY.md` — 策略设计文档
- `docs/agents/**` — Harness 设计文档
- `.catpaw/rules/**` — 编码规则
- `scripts/**` — 辅助脚本

### 禁止修改的文件
- `demo/simkit/**`
- `demo/server/**`
- `docs/0[1-5]-*.md`

## 决策链路速览

```
decide(driver_id)
  → PreferenceParser（首步 LLM 解析偏好文本 → 结构化配置）
  → StateTracker.update（维护累计状态）
  → SchedulePlanner.check（强制动作检查：休息/禁区/事件）
  → query_cargo（环境接口获取候选货源）
  → RuleEngine.filter（确定性过滤：品类/距离/地理围栏）
  → CargoScorer.rank（多维打分 Top-5）
  → TokenBudget → StrategyAdvisor.enhance（LLM 增强或纯规则）
  → 返回 action
```

## 运行命令

```bash
# 安装依赖
cd demo/server && pip install -r requirements.txt

# 运行仿真
cd demo/server && python main.py

# 收益计算
cd demo && python calc_monthly_income.py

# 运行测试
cd demo && python -m pytest agent/tests/ -v

# Lint 检查
bash scripts/lint-project.sh
```

## 设计决策索引

所有重大设计决策沉淀在 `docs/agents/` 目录下，包括：
- 为什么选 5 层架构而非纯 LLM
- Token 预算分配策略
- 偏好罚分避让优先级
- 特殊事件状态机设计

详见各文档。
