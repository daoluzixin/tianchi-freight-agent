#!/usr/bin/env bash
# parallel-sim.sh — 5 进程并行仿真，每组 2 个司机
# 用法:
#   bash scripts/parallel-sim.sh                  # 默认运行
#   bash scripts/parallel-sim.sh "测试描述"        # 带备注
#   bash scripts/parallel-sim.sh "描述" 500        # 带备注 + max-steps

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")/../demo" && pwd)"
NOTE="${1:-parallel-run}"
MAX_STEPS="${2:-}"

STEP_ARG=""
[ -n "$MAX_STEPS" ] && STEP_ARG="--max-steps $MAX_STEPS"

GROUPS=(
  "D001,D002"
  "D003,D004"
  "D005,D006"
  "D007,D008"
  "D009,D010"
)

PIDS=()
LOGS=()

echo "============================================================"
echo "  并行仿真启动: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  备注: $NOTE"
echo "  分组: ${GROUPS[*]}"
[ -n "$MAX_STEPS" ] && echo "  max-steps: $MAX_STEPS"
echo "============================================================"
echo

for i in "${!GROUPS[@]}"; do
  g="${GROUPS[$i]}"
  log="/tmp/sim_p$((i+1)).log"
  LOGS+=("$log")
  cd "$DEMO_DIR"
  python run_simulation.py --drivers "$g" $STEP_ARG --note "${NOTE}-p$((i+1))" > "$log" 2>&1 &
  PIDS+=($!)
  echo "  P$((i+1)) started: drivers=$g  pid=$!  log=$log"
done

echo
echo "  等待全部完成..."
echo

FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  P$((i+1)) [${GROUPS[$i]}] ✓ 完成"
  else
    echo "  P$((i+1)) [${GROUPS[$i]}] ✗ 失败 (exit=$?)"
    FAILED=$((FAILED+1))
  fi
done

echo
echo "============================================================"
echo "  全部完成: $(date '+%Y-%m-%d %H:%M:%S')  失败: $FAILED/5"
echo "============================================================"
echo

# 汇总收益
for i in "${!LOGS[@]}"; do
  result=$(grep "总净收益" "${LOGS[$i]}" 2>/dev/null | tail -1 || true)
  penalty=$(grep "总偏好罚分" "${LOGS[$i]}" 2>/dev/null | tail -1 || true)
  if [ -n "$result" ]; then
    echo "  P$((i+1)) [${GROUPS[$i]}]: $result  $penalty"
  fi
done

# 取最完整的那组（P1 通常最后完成，能看到全部司机）
FULL_LOG=$(for l in "${LOGS[@]}"; do grep -c "D0" "$l" 2>/dev/null | paste - <(echo "$l"); done | sort -rn | head -1 | awk '{print $2}')
echo
echo "  完整报告见: $FULL_LOG"
echo
grep -A 30 "收益报告" "$FULL_LOG" 2>/dev/null | head -35 || true
