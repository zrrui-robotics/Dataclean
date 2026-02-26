#!/usr/bin/env bash
# 一键执行：提取 JSON -> 平滑 -> 评估指标
# 在下面或通过环境变量指定 SOURCE_DIR（必选），可选 OUTPUT_DIR、PREFIX

set -e

# 源目录（必选，可改为你的路径或通过环境变量传入）
SOURCE_DIR="${SOURCE_DIR:-/Users/rui/Desktop/traj/task57}"
# 提取后的 JSON 输出目录（默认与 SOURCE_DIR 相同，即覆盖/写入同一目录）
OUTPUT_DIR="${OUTPUT_DIR:-$SOURCE_DIR}"
# 文件名前缀，如 57 得到 57-1.json, 57-2.json
PREFIX="${PREFIX:-57}"
# 后续平滑与指标都基于 OUTPUT_DIR（提取结果所在目录）
TASK_DIR="$OUTPUT_DIR"

# 脚本所在目录（用于在 data_clean 下执行 python）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== 1. 提取 JSON =========="
echo "  source: $SOURCE_DIR  output: $OUTPUT_DIR  prefix: $PREFIX"
python extra_json.py --source-dir "$SOURCE_DIR" --output-dir "$OUTPUT_DIR" --prefix "$PREFIX"

echo ""
echo "========== 2. 平滑轨迹 =========="
python smooth_resampled_traj.py -i "$TASK_DIR" -o "$TASK_DIR/cleaned_data"

echo ""
echo "========== 3. 评估指标 =========="
python -m util.metrics_util \
  -b "$TASK_DIR" \
  -a "$TASK_DIR/cleaned_data" \
  -o "$TASK_DIR/batch_metrics" \
  --output-format both

echo ""
echo "========== 完成 =========="
echo "指标已保存: $TASK_DIR/batch_metrics.json 与 $TASK_DIR/batch_metrics.csv"
