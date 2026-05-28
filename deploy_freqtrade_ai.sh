#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/freqtrade-ai}"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/freqtrade-ai-local-backups}"

cd "$REPO_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
backup_dir="$BACKUP_ROOT/$timestamp"
mkdir -p "$backup_dir"

# 备份本地运行数据（存在即备份，不存在则跳过）
runtime_paths=(
  "user_data/ai_memory"
  "user_data/config*.json"
  "user_data/strategies/MultiCoin_AI_Strategy_*.py"
  "user_data/strategies/MultiCoin_AI_Strategy_v*.py"
  "user_data/strategies/generated"
  ".env"
  ".env.bak.*"
)

for pattern in "${runtime_paths[@]}"; do
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if [[ -e "$path" ]]; then
      mkdir -p "$backup_dir/$(dirname "$path")"
      cp -a "$path" "$backup_dir/$path"
    fi
  done < <(compgen -G "$pattern" || true)
done

echo "已备份本地运行数据：$backup_dir"

# 仅检查代码文件是否有本地修改；忽略 runtime / ignored 文件
code_changes="$(git status --short --untracked-files=no)"
if [[ -n "$code_changes" ]]; then
  echo "检测到代码文件存在本地修改，已中止更新："
  echo "$code_changes"
  exit 1
fi

echo "已跳过 runtime 文件检查"
echo "开始拉取代码更新"

git fetch --all --prune
git pull --ff-only
