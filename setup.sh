#!/usr/bin/env bash
#
# setup.sh — one-shot placeholder resolver for the SynZIF-8 release.
#
# Usage:
#   bash setup.sh <DATA_ROOT>
#
# Example:
#   bash setup.sh /scratch/users/alice/synzif8_data
#
# What it does:
#   The source files in this release contain literal placeholders
#       <PROJECT_ROOT>   →  the directory of this repository
#       <DATA_ROOT>      →  where you extracted the SynZIF-8 dataset
#       <USER_HOME>      →  $HOME of the original developer (rewritten to $HOME here)
#   This script rewrites every occurrence in .py / .yaml / .yml / .sh / .md / .json
#   to your local absolute paths so the code can run as-is.
#
#   It is safe to re-run: re-running with a different DATA_ROOT just rewrites
#   the placeholder again (idempotent w.r.t. shape — but you must run on a
#   freshly-cloned tree, since the placeholders are gone after the first run).

set -euo pipefail

# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
  cat <<EOF >&2
Usage: bash setup.sh <DATA_ROOT>

  <DATA_ROOT>   Absolute path to the directory containing the SynZIF-8 dataset
                (the parent of dataset_v6/, dataset_v6_sem/, etc.).

Example:
  bash setup.sh /scratch/users/alice/synzif8_data
EOF
  exit 1
fi

DATA_ROOT="$1"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
USER_HOME="$HOME"

if [ ! -d "$DATA_ROOT" ]; then
  echo "WARNING: <DATA_ROOT>='$DATA_ROOT' does not exist yet." >&2
  echo "         Continuing — but you must download the dataset before running pipelines." >&2
fi

echo "============================================================"
echo "SynZIF-8 setup"
echo "  PROJECT_ROOT = $PROJECT_ROOT"
echo "  DATA_ROOT    = $DATA_ROOT"
echo "  USER_HOME    = $USER_HOME"
echo "============================================================"

# -----------------------------------------------------------------------------
# Rewrite placeholders. We escape '/' in paths for sed via '|' delimiter.
# -----------------------------------------------------------------------------
ESC_PROJECT=${PROJECT_ROOT//\\/\\\\}
ESC_DATA=${DATA_ROOT//\\/\\\\}
ESC_HOME=${USER_HOME//\\/\\\\}

# Files to touch
FILES=$(find "$PROJECT_ROOT" \
  \( -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.sh" \
     -o -name "*.md" -o -name "*.json" -o -name "*.ipynb" \) \
  -not -path "*/_staging_meta/*" \
  -not -path "*/.git/*" \
  -not -name "setup.sh" \
  -type f)

COUNT_PROJECT=0
COUNT_DATA=0
COUNT_HOME=0

for f in $FILES; do
  changed=0
  if grep -q "<PROJECT_ROOT>" "$f"; then
    sed -i "s|<PROJECT_ROOT>|$ESC_PROJECT|g" "$f"
    COUNT_PROJECT=$((COUNT_PROJECT + 1))
    changed=1
  fi
  if grep -q "<DATA_ROOT>" "$f"; then
    sed -i "s|<DATA_ROOT>|$ESC_DATA|g" "$f"
    COUNT_DATA=$((COUNT_DATA + 1))
    changed=1
  fi
  if grep -q "<USER_HOME>" "$f"; then
    sed -i "s|<USER_HOME>|$ESC_HOME|g" "$f"
    COUNT_HOME=$((COUNT_HOME + 1))
    changed=1
  fi
done

echo ""
echo "Replacements:"
printf "  <PROJECT_ROOT>  →  %s   (%d files)\n" "$PROJECT_ROOT" "$COUNT_PROJECT"
printf "  <DATA_ROOT>     →  %s   (%d files)\n" "$DATA_ROOT"    "$COUNT_DATA"
printf "  <USER_HOME>     →  %s   (%d files)\n" "$USER_HOME"    "$COUNT_HOME"

echo ""
echo "Done. You can now:"
echo "  pip install -r requirements.txt"
echo "  python baselines/common/overfit_runner_ddp.py --model A1_gdrnet ..."
