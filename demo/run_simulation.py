"""仿真运行器：执行仿真并将结果记录到 demo/log/ 文件夹。

每次运行会：
1. 执行仿真（支持 --max-steps 参数）
2. 将 results/ 中的结果（含 actions_*.jsonl）完整复制到 log/<timestamp>/ 目录
3. 自动运行 calc_monthly_income.py 计算收益和偏好罚分
4. 生成完整运行摘要 log/<timestamp>/summary.json（含收益数据）
5. 打印关键指标（净收益、偏好罚分、各司机明细）

用法：
    python3 run_simulation.py                    # 使用 config 中的 max_steps
    python3 run_simulation.py --max-steps 1000   # 指定步数
    python3 run_simulation.py --max-steps 500 --note "优化后首次测试"
    python3 run_simulation.py --skip-income      # 跳过收益计算（加速调试）
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
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


def _copy_results_to_log(run_log_dir: Path) -> None:
    """将 results/ 完整复制到 log 目录，确保 actions_*.jsonl 被保留。"""
    results_dir = _DEMO_ROOT / "results"
    if not results_dir.exists():
        return
    for item in results_dir.iterdir():
        if item.is_file():
            shutil.copy2(str(item), str(run_log_dir / item.name))
        elif item.is_dir() and item.name == "logs":
            log_dest = run_log_dir / "logs"
            if log_dest.exists():
                shutil.rmtree(str(log_dest))
            shutil.copytree(str(item), str(log_dest))


def _run_income_calculation(run_log_dir: Path) -> dict | None:
    """运行 calc_monthly_income.py 计算收益，返回解析后的 JSON 或 None。

    使用 log 目录中的 actions 文件作为输入（而非 results/，避免被覆盖）。
    """
    calc_script = _DEMO_ROOT / "calc_monthly_income.py"
    if not calc_script.exists():
        return None

    # 检查 log 目录是否有 actions 文件
    action_files = list(run_log_dir.glob("actions_202603_*.jsonl"))
    if not action_files:
        return None

    # 需要 run_summary_202603.json 也在 log 目录
    run_summary = run_log_dir / "run_summary_202603.json"
    if not run_summary.exists():
        return None

    try:
        result = subprocess.run(
            [
                sys.executable, str(calc_script),
                "--project-root", str(_DEMO_ROOT),
                "--results-dir", str(run_log_dir),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            # calc_monthly_income.py 输出 JSON 到 stdout
            return json.loads(result.stdout)
        else:
            print(f"  [WARN] 收益计算失败: {result.stderr[:200]}")
            return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"  [WARN] 收益计算异常: {e}")
        return None


def _extract_income_summary(income_data: dict) -> dict:
    """从 calc_monthly_income.py 输出中提取关键收益指标。"""
    summary = income_data.get("summary", {})
    drivers = income_data.get("drivers", [])

    driver_details = {}
    for d in drivers:
        did = d.get("driver_id", "")
        inc = d.get("income", {})
        pref = d.get("preference_check", {})
        rules = pref.get("rules", [])

        # 统计违规条数
        violated_rules = [r for r in rules if r.get("penalty", 0) > 0
                          or r.get("satisfied") is False]

        driver_details[did] = {
            "gross_income": inc.get("gross_income", 0),
            "cost": inc.get("cost", 0),
            "preference_penalty": inc.get("preference_penalty", 0),
            "net_income": inc.get("net_income", 0),
            "violated_rules": len(violated_rules),
            "total_rules": len(rules),
        }

    return {
        "total_net_income": summary.get("total_net_income_all_drivers", 0),
        "total_preference_penalty": summary.get("total_preference_penalty", 0),
        "failed_driver_count": summary.get("failed_driver_count", 0),
        "driver_details": driver_details,
    }


def _print_income_report(income_summary: dict) -> None:
    """打印收益报告到控制台。"""
    print()
    print(f"{'─'*60}")
    print(f"  💰 收益报告")
    print(f"{'─'*60}")
    total = income_summary["total_net_income"]
    penalty = income_summary["total_preference_penalty"]
    color_total = f"\033[92m{total:+,.2f}\033[0m" if total >= 0 else f"\033[91m{total:+,.2f}\033[0m"
    print(f"  总净收益:     {color_total} 元")
    print(f"  总偏好罚分:   \033[93m-{penalty:,.2f}\033[0m 元")
    if income_summary["failed_driver_count"] > 0:
        print(f"  校验失败司机: {income_summary['failed_driver_count']} 人")
    print()

    details = income_summary["driver_details"]
    if details:
        print(f"  {'司机':<6} {'毛收入':>10} {'成本':>10} {'罚分':>10} {'净收益':>12} {'违规'}")
        print(f"  {'─'*62}")
        for did in sorted(details.keys()):
            d = details[did]
            net = d["net_income"]
            net_str = f"{net:+,.1f}"
            violated = f"{d['violated_rules']}/{d['total_rules']}"
            print(f"  {did:<6} {d['gross_income']:>10,.1f} {d['cost']:>10,.1f} "
                  f"{d['preference_penalty']:>10,.1f} {net_str:>12} {violated:>6}")
    print()


ALL_DRIVER_IDS = [f"D{i:03d}" for i in range(1, 11)]


def run(max_steps: int | None = None, note: str = "",
        skip_income: bool = False, drivers: list[str] | None = None,
        config_path: str | None = None) -> dict:
    """执行仿真并记录结果。

    Args:
        drivers: 要运行的司机列表。None 表示随机选 3 个。
        config_path: 自定义 config.json 路径。None 使用默认。
    """
    from bench.evaluation_runner import EvaluationRunner
    from simkit.driver_state_manager import DriverStateManager

    # 确定本次运行的司机
    if drivers is None:
        drivers = sorted(random.sample(ALL_DRIVER_IDS, 3))

    # Monkey-patch DriverStateManager.list_driver_ids 来只运行选定的司机
    _original_list_driver_ids = DriverStateManager.list_driver_ids

    def _patched_list_driver_ids(self):
        all_ids = _original_list_driver_ids(self)
        return [d for d in all_ids if d in drivers]

    DriverStateManager.list_driver_ids = _patched_list_driver_ids

    # 创建本次运行的 log 目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = _LOG_DIR / timestamp
    run_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  仿真开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  司机: {', '.join(drivers)}")
    print(f"  max_steps: {max_steps or '(config default)'}")
    print(f"  日志目录: {run_log_dir}")
    if note:
        print(f"  备注: {note}")
    print(f"{'='*60}")
    print()

    wall_start = time.perf_counter()

    try:
        _config = Path(config_path) if config_path else None
        runner = EvaluationRunner(config_path=_config, max_steps=max_steps)
        result = runner.run()
    except Exception as e:
        DriverStateManager.list_driver_ids = _original_list_driver_ids
        # 仿真失败：也保存已有的 results 文件
        _copy_results_to_log(run_log_dir)
        error_info = {
            "timestamp": timestamp,
            "status": "FAILED",
            "error": str(e),
            "error_type": type(e).__name__,
            "max_steps": max_steps,
            "note": note,
        }
        (run_log_dir / "summary.json").write_text(
            json.dumps(error_info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n仿真失败: {e}")
        raise

    wall_elapsed = time.perf_counter() - wall_start

    # 恢复原始方法
    DriverStateManager.list_driver_ids = _original_list_driver_ids

    # 复制 results/ 完整内容到 log 目录（永久保留）
    _copy_results_to_log(run_log_dir)

    # 计算关键指标
    steps = result.get("completed_steps", 0)
    sim_time = result.get("simulate_time_seconds", 0)
    steps_per_sec = steps / sim_time if sim_time > 0 else 0

    # 生成运行摘要
    summary = {
        "timestamp": timestamp,
        "status": "SUCCESS",
        "wall_time_seconds": round(wall_elapsed, 2),
        "max_steps_config": max_steps,
        "note": note,
        "result": {
            "completed_steps": steps,
            "simulate_time_seconds": result.get("simulate_time_seconds"),
            "driver_completed_steps": result.get("driver_completed_steps"),
            "remaining_cargo_count": result.get("remaining_cargo_count"),
        },
        "metrics": {
            "steps_per_second": round(steps_per_sec, 2),
            "wall_time_minutes": round(wall_elapsed / 60, 2),
        },
    }

    # 自动运行收益计算
    income_summary = None
    if not skip_income:
        print("  正在计算收益...")
        income_data = _run_income_calculation(run_log_dir)
        if income_data:
            # 将完整收益结果单独保存
            income_path = run_log_dir / "monthly_income_202603.json"
            income_path.write_text(
                json.dumps(income_data, ensure_ascii=False, indent=2), encoding="utf-8")

            # 提取摘要写入 summary
            income_summary = _extract_income_summary(income_data)
            summary["income"] = income_summary
        else:
            summary["income"] = None
            summary["income_note"] = "收益计算未执行或失败"

    # 写入最终 summary
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

    # 打印收益报告
    if income_summary:
        _print_income_report(income_summary)

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
    parser.add_argument(
        "--skip-income", action="store_true",
        help="跳过收益计算（加速调试）",
    )
    parser.add_argument(
        "--drivers", type=str, default=None,
        help="指定运行的司机（逗号分隔，如 D001,D009,D010）。不指定则随机选 3 个。",
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_drivers",
        help="运行全部 10 个司机",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="自定义 config.json 路径（默认使用 server/config/config.json）",
    )
    args = parser.parse_args()

    # 解析司机参数
    if args.all_drivers:
        selected_drivers = ALL_DRIVER_IDS[:]
    elif args.drivers:
        selected_drivers = [d.strip() for d in args.drivers.split(",") if d.strip()]
    else:
        selected_drivers = None  # 随机选 3 个

    try:
        run(max_steps=args.max_steps, note=args.note,
            skip_income=args.skip_income, drivers=selected_drivers,
            config_path=args.config)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
