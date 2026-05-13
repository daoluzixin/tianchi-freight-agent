# 编码规范

本规则适用于 `demo/agent/` 下所有 Python 代码。

## 1. 禁止事项（红线）

- **禁止直读数据文件**：不得使用 `open()`、`Path().read_text()`、`json.load()` 等方式读取 `server/data/` 下的任何文件。所有数据必须通过 `SimulationApiPort` 接口获取。
- **禁止硬编码 driver_id**：不得对特定 driver_id（如 D009、D010）写 if/else 分支。配置必须来自运行时 PreferenceParser 解析结果。
- **禁止硬编码 cargo_id**：不得将特定货物 ID 写入代码。特殊货物信息从偏好文本中动态解析。
- **禁止修改 simkit/ 或 server/**：这两个包由赛方维护。
- **禁止 `import *`**：所有 import 必须显式指定。

## 2. 代码组织

- 每个模块职责单一，参照 `docs/agents/01-architecture.md` 的模块划分。
- 新增功能必须有对应的单测（`demo/agent/tests/`）。
- 公开函数必须有 docstring，说明参数与返回值。
- 类型注解：所有公开函数签名必须有类型标注。

## 3. 错误处理

- LLM API 调用必须有 try/except，失败时使用规则回退。
- 不得 bare `except:`，必须指定具体异常类型。
- 所有异常路径必须有日志记录（使用 `logging` 模块）。

## 4. 命名规范

- 模块文件：`snake_case.py`
- 类：`PascalCase`
- 函数/变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 私有方法/属性：以 `_` 开头

## 5. 性能约束

- 单步决策必须在 30 秒内返回。
- 避免在决策热路径上做大量计算（如遍历全部货源）。
- 使用 `haversine_km()` 计算距离（已有实现在 `state_tracker.py`），不自己重写。

## 6. 日志规范

- 使用 `logging.getLogger(__name__)`
- 决策关键节点必须有 INFO 级别日志
- LLM 调用结果必须记录 token 使用量
- 异常用 WARNING 或 ERROR 级别
