"""并行仿真运行器：每个司机独立进程并行仿真，大幅加速实验。

原理：
  赛方仿真引擎按 driver_id 串行跑，但每个司机的仿真是独立的
  （独立加载数据、独立状态推进）。本脚本为每个司机启动一个独立子进程，
  各自运行完整仿真后汇总结果。

  关键隔离：每个子进程使用独立的 results 目录（results_<driver_id>/），
  避免多进程并发写同一目录的文件冲突。

  10 个司机从串行 ~10x 单司机时间 → 并行 ~1x 单司机时间（受 CPU/API 并发限制）。

用法：
    python3 run_parallel.py                        # 全部 10 个司机并行
    python3 run_parallel.py --drivers D001,D005     # 只跑指定司机
    python3 run_parallel.py --workers 5             # 最多 5 个并行进程
    python3 run_parallel.py --max-steps 500         # 每个司机最多 500 步
    python3 run_parallel.py --skip-income           # 跳过收益计算（加速调试）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent
_SERVER_ROOT = _DEMO_ROOT / "server"
_DEFAULT_CONFIG = _SERVER_ROOT / "config" / "config.json"
ALL_DRIVER_IDS = [f"D{i:03d}" for i in range(1, 11)]


def _make_isolated_config(driver_id: str, base_config: Path, run_dir: Path) -> Path:
    """为单个司机生成隔离的 config.json，results_dir 和 log_dir 指向独立路径。"""
    raw = json.loads(base_config.read_text(encoding="utf-8"))

    # 每个司机的 results 放在 run_dir/<driver_id>/results/
    driver_results = run_dir / driver_id / "results"
    driver_logs = run_dir / driver_id / "results" / "logs"
    driver_results.mkdir(parents=True, exist_ok=True)
    driver_logs.mkdir(parents=True, exist_ok=True)

    # results_dir 和 log_dir 使用绝对路径，覆盖相对路径
    raw["results_dir"] = str(driver_results)
    raw["log_dir"] = str(driver_logs)

    config_path = run_dir / driver_id / "config.json"
    config_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def _run_single_driver(
    driver_id: str,
    max_steps: int | None,
    skip_income: bool,
    note: str,
    config_path: str,
) -> dict:
    """在子进程中运行单个司机的仿真，返回结果摘要。"""
    cmd = [
        sys.executable,
        str(_DEMO_ROOT / "run_simulation.py"),
        "--drivers", driver_id,
        "--config", config_path,
    ]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if skip_income:
        cmd.append("--skip-income")
    if note:
        cmd += ["--note", f"{note} [{driver_id}]"]

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 单司机最长 2 小时
            cwd=str(_SERVER_ROOT),  # 从 server/ 目录运行，保证相对路径正确
        )
        elapsed = time.perf_counter() - start
        return {
            "driver_id": driver_id,
            "status": "SUCCESS" if result.returncode == 0 else "FAILED",
            "returncode": result.returncode,
            "elapsed_seconds": round(elapsed, 2),
            "stdout_tail": result.stdout[-3000:] if result.stdout else "",
            "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "driver_id": driver_id,
            "status": "TIMEOUT",
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 2),
            "stdout_tail": "",
            "stderr_tail": "Process timed out after 7200s",
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "driver_id": driver_id,
            "status": "ERROR",
            "returncode": -1,
            "elapsed_seconds": round(elapsed, 2),
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


def _collect_results(run_dir: Path, drivers: list[str], log_dir: Path) -> dict | None:
    """汇总各司机的 results 到统一的 log 目录，并运行收益计算。"""
    # 合并所有司机的 actions 文件到 log 目录
    merged_count = 0
    for did in drivers:
        driver_results = run_dir / did / "results"
        if not driver_results.exists():
            continue
        for f in driver_results.iterdir():
            if f.is_file() and f.name.startswith("actions_"):
                shutil.copy2(str(f), str(log_dir / f.name))
                merged_count += 1
            elif f.is_file() and f.name.startswith("run_summary"):
                # 也复制 run_summary（收益计算需要）
                shutil.copy2(str(f), str(log_dir / f.name))

    if merged_count == 0:
        return None

    # 运行收益计算
    calc_script = _DEMO_ROOT / "calc_monthly_income.py"
    if not calc_script.exists():
        return None

    try:
        result = subprocess.run(
            [
                sys.executable, str(calc_script),
                "--project-root", str(_DEMO_ROOT),
                "--results-dir", str(log_dir),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def _print_summary_table(results: list[dict], total_elapsed: float,
                         income_data: dict | None) -> None:
    """打印汇总表格。"""
    print()
    print(f"{'='*70}")
    print(f"  并行仿真完成  |  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*70}")
    print(f"  {'司机':<8} {'状态':<10} {'耗时(s)':>10} {'备注'}")
    print(f"  {'─'*60}")

    success_count = 0
    for r in sorted(results, key=lambda x: x["driver_id"]):
        status = r["status"]
        elapsed = r["elapsed_seconds"]
        note = ""
        if status == "SUCCESS":
            success_count += 1
        elif status == "FAILED":
            note = r.get("stderr_tail", "")[:60]
        elif status == "TIMEOUT":
            note = "超时"

        status_colored = {
            "SUCCESS": f"\033[92m{status}\033[0m",
            "FAILED": f"\033[91m{status}\033[0m",
            "TIMEOUT": f"\033[93m{status}\033[0m",
            "ERROR": f"\033[91m{status}\033[0m",
        }.get(status, status)

        print(f"  {r['driver_id']:<8} {status_colored:<20} {elapsed:>8.1f}   {note}")

    print(f"  {'─'*60}")
    print(f"  成功: {success_count}/{len(results)}")

    # 加速比
    serial_time = sum(r["elapsed_seconds"] for r in results)
    speedup = serial_time / total_elapsed if total_elapsed > 0 else 1
    print(f"  串行总耗时: {serial_time:.1f}s | 并行实际: {total_elapsed:.1f}s | 加速比: {speedup:.1f}x")

    # 收益报告
    if income_data:
        summary = income_data.get("summary", {})
        drivers_data = income_data.get("drivers", [])
        total_net = summary.get("total_net_income_all_drivers", 0)
        total_penalty = summary.get("total_preference_penalty", 0)
        print()
        print(f"  {'─'*60}")
        color = "\033[92m" if total_net >= 0 else "\033[91m"
        print(f"  总净收益:   {color}{total_net:+,.2f}\033[0m 元")
        print(f"  总偏好罚分: \033[93m-{total_penalty:,.2f}\033[0m 元")
        print()
        if drivers_data:
            print(f"  {'司机':<6} {'毛收入':>10} {'成本':>10} {'罚分':>10} {'净收益':>12}")
            print(f"  {'─'*52}")
            for d in sorted(drivers_data, key=lambda x: x.get("driver_id", "")):
                did = d.get("driver_id", "")
                inc = d.get("income", {})
                net = inc.get("net_income", 0)
                print(f"  {did:<6} {inc.get('gross_income', 0):>10,.1f} "
                      f"{inc.get('cost', 0):>10,.1f} "
                      f"{inc.get('preference_penalty', 0):>10,.1f} "
                      f"{net:>+12,.1f}")

    print(f"{'='*70}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="并行仿真：每个司机独立进程同时运行")
    parser.add_argument(
        "--drivers", type=str, default=None,
        help="指定司机（逗号分隔），默认全部 10 个")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="最大并行进程数，默认等于司机数量（即全并行）")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="每个司机的最大步数")
    parser.add_argument(
        "--skip-income", action="store_true",
        help="跳过收益计算")
    parser.add_argument(
        "--note", type=str, default="",
        help="运行备注")
    parser.add_argument(
        "--config", type=str, default=None,
        help="自定义 config.json 路径")
    args = parser.parse_args()

    # 解析司机列表
    if args.drivers:
        drivers = [d.strip() for d in args.drivers.split(",") if d.strip()]
    else:
        drivers = ALL_DRIVER_IDS[:]

    workers = args.workers or len(drivers)
    workers = min(workers, len(drivers))

    base_config = Path(args.config) if args.config else _DEFAULT_CONFIG

    # 为本次并行运行创建独立的工作目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _DEMO_ROOT / "log" / f"parallel_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  并行仿真启动")
    print(f"  司机: {', '.join(drivers)}")
    print(f"  并行度: {workers} workers")
    print(f"  max_steps: {args.max_steps or '(config default)'}")
    print(f"  工作目录: {run_dir}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print()

    # 为每个司机生成隔离的 config
    driver_configs: dict[str, str] = {}
    for did in drivers:
        cfg = _make_isolated_config(did, base_config, run_dir)
        driver_configs[did] = str(cfg)

    wall_start = time.perf_counter()
    results: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_single_driver,
                driver_id=did,
                max_steps=args.max_steps,
                skip_income=True,  # 子进程先跳过收益计算，最后统一算
                note=args.note,
                config_path=driver_configs[did],
            ): did
            for did in drivers
        }

        for future in as_completed(futures):
            did = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "driver_id": did,
                    "status": "ERROR",
                    "returncode": -1,
                    "elapsed_seconds": 0,
                    "stdout_tail": "",
                    "stderr_tail": str(e),
                }
            results.append(result)
            # 实时打印完成的司机
            status_icon = "✓" if result["status"] == "SUCCESS" else "✗"
            print(f"  {status_icon} {did} 完成 ({result['elapsed_seconds']:.1f}s) - {result['status']}")

    total_elapsed = time.perf_counter() - wall_start

    # 汇总收益计算
    income_data = None
    if not args.skip_income:
        print("\n  正在汇总收益计算...")
        income_data = _collect_results(run_dir, drivers, run_dir)
        if income_data:
            income_path = run_dir / "monthly_income_202603.json"
            income_path.write_text(
                json.dumps(income_data, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary_table(results, total_elapsed, income_data)

    # 保存并行运行的汇总
    summary_path = run_dir / "parallel_summary.json"
    summary_path.write_text(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "drivers": drivers,
            "workers": workers,
            "max_steps": args.max_steps,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  结果目录: {run_dir}")

    failed = [r for r in results if r["status"] != "SUCCESS"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
