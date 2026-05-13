# 工作流规范

## 1. 修改代码前

1. 阅读 `AGENTS.md` 确认修改范围在允许区域内
2. 阅读 `docs/agents/` 中相关设计文档
3. 确认修改不违反 `decision-principles.md` 的不变量

## 2. 修改代码后

1. 运行测试：`cd demo && python -m pytest agent/tests/ -v`
2. 运行 lint：`bash scripts/lint-project.sh`
3. 如果修改了决策逻辑，跑一次快速仿真验证：
   ```bash
   cd demo/server && python main.py  # 或使用少步数配置
   ```
4. 检查仿真日志无异常（无 ERROR/崩溃/格式错误）

## 3. 设计决策沉淀

任何重大设计变更（新增模块、修改架构、调整策略方向）必须：
1. 在 `docs/agents/` 下新建或更新对应文档
2. 在 commit message 中说明决策理由
3. 如果是架构变更，同步更新 `AGENTS.md` 的结构描述

## 4. 仿真验证等级

| 修改类型 | 最低验证要求 |
|---------|-------------|
| 评分权重微调 | 10 步快速仿真 |
| 新增规则过滤 | 单测 + 单司机完整月 |
| 架构变更 | 全部司机完整月仿真 + 收益计算 |
| Prompt 修改 | 10 步仿真确认无格式错误 |

## 5. 提交规范

- commit message 格式：`<type>(<scope>): <description>`
- type: feat / fix / refactor / docs / test / perf
- scope: agent / scorer / planner / engine / strategy / harness
- 示例：`feat(strategy): add daily review for token budget allocation`
