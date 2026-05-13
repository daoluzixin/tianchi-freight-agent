"""参赛选手 Agent 包：决策逻辑见 `core.model_decision_service`。

子包结构：
  - core/      状态追踪、时间规划、规则引擎、决策服务
  - scoring/   货源评分、供给预测
  - strategy/  LLM 策略顾问、Token 预算管理
  - config/    司机配置、偏好解析
  - tests/     测试用例
"""

from agent.core.model_decision_service import ModelDecisionService
from agent.core.state_tracker import StateTracker, DriverState
from agent.config.driver_config import DriverConfig, get_config, register_config
