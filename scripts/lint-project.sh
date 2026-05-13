#!/usr/bin/env bash
# lint-project.sh — 项目自定义 Linter
# 将"品味"编码为可执行规则，确保 Agent 编码规范持续满足。
#
# 用法：bash scripts/lint-project.sh
# 返回：0=通过, 1=有错误

set -euo pipefail

ERRORS=0
WARNINGS=0
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$PROJECT_ROOT/demo/agent"

echo "======================================"
echo " 🔍 Project Lint — Harness Validator"
echo "======================================"
echo ""

# ─── 1. 禁止直读数据文件 ───────────────────────────────────────────
echo "▸ [Rule 1] 检查禁止直读数据文件..."
if grep -rn "server/data/" "$AGENT_DIR" --include="*.py" 2>/dev/null | grep -v "__pycache__" | grep -v ".pyc" | grep -v "test_" | grep -v "print("; then
    echo "  ❌ ERROR: Agent 代码中引用了 server/data/ 路径"
    ((ERRORS++))
else
    echo "  ✅ 通过"
fi

if grep -rn "cargo_dataset\|drivers\.json" "$AGENT_DIR" --include="*.py" 2>/dev/null | grep -v "__pycache__" | grep -v ".pyc" | grep -v "test_" | grep -v "#"; then
    echo "  ❌ ERROR: Agent 代码中直接引用了数据文件名"
    ((ERRORS++))
else
    echo "  ✅ 通过"
fi
echo ""

# ─── 2. 禁止硬编码 driver_id ──────────────────────────────────────
echo "▸ [Rule 2] 检查禁止硬编码 driver_id..."
# 排除测试文件、注释、字符串中的 D001 等
HARDCODED=$(grep -rn '"D0[0-9][0-9]"' "$AGENT_DIR" --include="*.py" 2>/dev/null | grep -v "__pycache__" | grep -v "test_" | grep -v "# " || true)
if [ -n "$HARDCODED" ]; then
    echo "  ⚠️  WARNING: 发现可能的硬编码 driver_id（请确认是否在测试/注释中）:"
    echo "$HARDCODED" | head -5
    ((WARNINGS++))
else
    echo "  ✅ 通过"
fi
echo ""

# ─── 3. 禁止 import * ─────────────────────────────────────────────
echo "▸ [Rule 3] 检查禁止 import *..."
STAR_IMPORTS=$(grep -rn "from .* import \*" "$AGENT_DIR" --include="*.py" 2>/dev/null | grep -v "__pycache__" || true)
if [ -n "$STAR_IMPORTS" ]; then
    echo "  ❌ ERROR: 发现 import * 用法:"
    echo "$STAR_IMPORTS"
    ((ERRORS++))
else
    echo "  ✅ 通过"
fi
echo ""

# ─── 4. 检查 simkit/server 是否被修改 ────────────────────────────
echo "▸ [Rule 4] 检查 simkit/server 未被修改..."
if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
    MODIFIED_FORBIDDEN=$(git diff --name-only HEAD 2>/dev/null | grep -E "^demo/(simkit|server)/" || true)
    if [ -n "$MODIFIED_FORBIDDEN" ]; then
        echo "  ❌ ERROR: 检测到禁止修改的文件被改动:"
        echo "$MODIFIED_FORBIDDEN"
        ((ERRORS++))
    else
        echo "  ✅ 通过"
    fi
else
    echo "  ⏭️  跳过（非 git 仓库或无 git 命令）"
fi
echo ""

# ─── 5. 检查 LLM 调用有 try/except ───────────────────────────────
echo "▸ [Rule 5] 检查 LLM API 调用有异常处理..."
# 查找 model_gateway_client 或 openai/dashscope 调用
LLM_CALLS=$(grep -rn "call_model\|chat\.completions\|Generation\.call\|_call_llm\|_invoke_model" "$AGENT_DIR" --include="*.py" 2>/dev/null | grep -v "__pycache__" | grep -v "test_" || true)
if [ -n "$LLM_CALLS" ]; then
    # 简单检查：调用行的前 5 行是否有 try
    UNPROTECTED=0
    while IFS= read -r line; do
        FILE=$(echo "$line" | cut -d: -f1)
        LINENO=$(echo "$line" | cut -d: -f2)
        # 检查前5行是否有 try
        START=$((LINENO - 5))
        if [ $START -lt 1 ]; then START=1; fi
        CONTEXT=$(sed -n "${START},${LINENO}p" "$FILE" 2>/dev/null || true)
        if ! echo "$CONTEXT" | grep -q "try"; then
            if [ $UNPROTECTED -eq 0 ]; then
                echo "  ⚠️  WARNING: 以下 LLM 调用可能缺少 try/except 保护:"
            fi
            echo "    $line"
            ((UNPROTECTED++))
        fi
    done <<< "$LLM_CALLS"
    if [ $UNPROTECTED -gt 0 ]; then
        ((WARNINGS++))
    else
        echo "  ✅ 通过"
    fi
else
    echo "  ✅ 通过（未检测到 LLM 调用）"
fi
echo ""

# ─── 6. 检查 bare except ──────────────────────────────────────────
echo "▸ [Rule 6] 检查禁止 bare except..."
BARE_EXCEPT=$(grep -rn "except:" "$AGENT_DIR" --include="*.py" 2>/dev/null | grep -v "__pycache__" | grep -v "except:\s*#" | grep -v "# noqa" || true)
if [ -n "$BARE_EXCEPT" ]; then
    echo "  ⚠️  WARNING: 发现 bare except（应指定具体异常类型）:"
    echo "$BARE_EXCEPT" | head -5
    ((WARNINGS++))
else
    echo "  ✅ 通过"
fi
echo ""

# ─── 7. 检查 str.format 与 JSON 花括号冲突 ────────────────────────
echo "▸ [Rule 7] 检查 Prompt 模板中的花括号安全..."
# 查找 .format() 调用中是否有 JSON 风格的裸花括号
FORMAT_ISSUES=$(grep -rn '\.format(' "$AGENT_DIR" --include="*.py" 2>/dev/null | grep -v "__pycache__" | grep -v "test_" || true)
if [ -n "$FORMAT_ISSUES" ]; then
    echo "  ⚠️  WARNING: 检测到 .format() 用法，请确认 JSON 示例中花括号已转义:"
    echo "$FORMAT_ISSUES" | head -3
    ((WARNINGS++))
else
    echo "  ✅ 通过"
fi
echo ""

# ─── 8. 检查测试文件存在 ─────────────────────────────────────────
echo "▸ [Rule 8] 检查测试覆盖..."
AGENT_MODULES=$(find "$AGENT_DIR" -name "*.py" -not -path "*/__pycache__/*" -not -path "*/tests/*" -not -name "__init__.py" | wc -l | tr -d ' ')
TEST_FILES=$(find "$AGENT_DIR/tests" -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
echo "  Agent 模块数: $AGENT_MODULES, 测试文件数: $TEST_FILES"
if [ "$TEST_FILES" -lt 2 ]; then
    echo "  ⚠️  WARNING: 测试文件过少，建议增加覆盖"
    ((WARNINGS++))
else
    echo "  ✅ 通过"
fi
echo ""

# ─── 汇总 ─────────────────────────────────────────────────────────
echo "======================================"
echo " 📊 结果汇总"
echo "======================================"
echo "  ERRORS:   $ERRORS"
echo "  WARNINGS: $WARNINGS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "❌ FAILED — 请修复以上 ERROR 后再提交"
    exit 1
else
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠️  PASSED with warnings — 建议检查以上 WARNING"
    else
        echo "✅ ALL PASSED"
    fi
    exit 0
fi
