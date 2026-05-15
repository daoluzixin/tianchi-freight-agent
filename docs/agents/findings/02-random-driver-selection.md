# 收获 02 — 随机司机选取与仿真运行器改进

> 日期：2026-05-13
> 目标：快速迭代调试，每次只跑 3 个司机，减少仿真耗时

## 改动内容

修改 `demo/run_simulation.py`，支持灵活的司机选取策略。

### 功能设计

- **默认行为**：从 10 位司机中 `random.sample(ALL_DRIVER_IDS, 3)` 随机选 3 个
- **`--drivers D001,D009,D010`**：手动指定
- **`--all`**：运行全部 10 个司机

### 技术实现：Monkey-Patch

由于 `server/` 代码不可修改（赛方红线），用 monkey-patch 方式过滤司机：

```python
from simkit.driver_state_manager import DriverStateManager

_original_list_driver_ids = DriverStateManager.list_driver_ids

def _patched_list_driver_ids(self):
    all_ids = _original_list_driver_ids(self)
    return [d for d in all_ids if d in drivers]

DriverStateManager.list_driver_ids = _patched_list_driver_ids
```

仿真结束后恢复原始方法，确保无副作用。

### CLI 参数

```bash
# 随机 3 个（默认）
python run_simulation.py --max-steps 500

# 指定
python run_simulation.py --max-steps 500 --drivers D006,D008,D009

# 全部
python run_simulation.py --max-steps 500 --all
```

### 收益

- 单次仿真从 ~30min（10 司机）降至 ~9min（3 司机）
- 快速验证单个司机策略调整的效果
- 多次随机抽样可覆盖不同司机组合

## 文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `demo/run_simulation.py` | 修改 | 新增 random 选取、--drivers/--all 参数、monkey-patch 逻辑 |
