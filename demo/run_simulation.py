"""仿真运行器：执行仿真并将结果记录到 demo/log/ 文件夹。

每次运行会：
1. 执行仿真（支持 --max-steps 参数）
2. 将 results/ 中的结果复制到 log/<timestamp>/ 目录
3. 生成运行摘要 log/<timestamp>/summary.json
4. 打印关键指标

用法：
    python3 run_simulation.py                    # 使用 config 中的 max_steps
    python3 run_simulation.py --max-steps 1000   # 指定步数
    python3 run_simulation.py --max-steps 500 --note "优化后首次测试"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent
_SERVER_ROOT = _DEMO_ROOT / "server"
_LOG_DIR = _DEMO_ROOT / "log"

# 确保 import 路径正确
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))


def run(max_steps: int | None = None, note: str = "") -> dict:
    """执行仿真并记录结果。"""
    from bench.evaluation_runner import EvaluationRunner

    # 创建本次运行的 log 目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = _LOG_DIR / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  仿真开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  max_steps: {max_steps or '(config default)'}")
    print(f"  日志目录: {run_log_dir}")
    if note:
        print(f"  备注: {note}")
    print(f"{'='*60}")
    print()

    wall_start = time.perf_counter()

    try:
        runner = EvaluationRunner(config_path=None, max_steps=max_steps)
        result = runner.run()
    except Exception as e:
        # 记录失败信息
        error_info = {
            "timestamp": timestamp,
            "status": "FAILED",
            "error": str(e),
            "max_steps": max_steps,
            "note": note,
        }
        (run_log_dir / "summary.json").write_text(
            json.dumps(error_info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n仿真失败: {e}")
        raise

    wall_elapsed = time.perf_counter() - wall_start

    # 复制 results/ 内容到 log 目录
    results_dir = _DEMO_ROOT / "results"
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_file():
                shutil.copy2(str(item), str(run_log_dir / item.name))
            elif item.is_dir() and item.name == "logs":
                # 复制日志子目录
                log_dest = run_log_dir / "logs"
                if log_dest.exists():
                    shutil.rmtree(str(log_dest))
                shutil.copytree(str(item), str(log_dest))

    # 生成运行摘要
    summary = {
        "timestamp": timestamp,
        "status": "SUCCESS",
        "wall_time_seconds": round(wall_elapsed, 2),
        "max_steps_config": max_steps,
        "note": note,
        "result": {
            "completed_steps": result.get("completed_steps"),
            "simulate_time_seconds": result.get("simulate_time_seconds"),
            "driver_completed_steps": result.get("driver_completed_steps"),
            "remaining_cargo_count": result.get("remaining_cargo_count"),
        },
    }

    # 计算关键指标
    steps = result.get("completed_steps", 0)
    sim_time = result.get("simulate_time_seconds", 0)
    steps_per_sec = steps / sim_time if sim_time > 0 else 0
    summary["metrics"] = {
        "steps_per_second": round(steps_per_sec, 2),
        "wall_time_minutes": round(wall_elapsed / 60, 2),
    }

    summary_path = run_log_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印结果
    print()
    print(f"{'='*60}")
    print(f"  仿真完成!")
    print(f"{'='*60}")
    print(f"  总步数: {steps}")
    print(f"  仿真耗时: {sim_time:.1f}s ({sim_time/60:.1f}min)")
    print(f"  实际耗时: {wall_elapsed:.1f}s ({wall_elapsed/60:.1f}min)")
    print(f"  速度: {steps_per_sec:.2f} steps/s")
    print()

    driver_steps = result.get("driver_completed_steps", {})
    if driver_steps:
        print("  各司机步数:")
        for did, s in sorted(driver_steps.items()):
            print(f"    {did}: {s} steps")
    print()
    print(f"  结果已保存到: {run_log_dir}")
    print(f"{'='*60}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="运行仿真并记录结果到 demo/log/")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="全局最大步数（省略则使用 config.json 中的值）",
    )
    parser.add_argument(
        "--note", type=str, default="",
        help="本次运行的备注说明",
    )
    args = parser.parse_args()

    try:
        run(max_steps=args.max_steps, note=args.note)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
