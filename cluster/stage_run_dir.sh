#!/bin/bash
# Stage Dynamo onto /home for nodes that cannot see /rg (athena-post), and
# collect results back. Run from the Dynamo repo root on a node that sees both.
#   bash cluster/stage_run_dir.sh [DATASET_DIR...]          # stage code + tables
#   bash cluster/stage_run_dir.sh --collect DATASET_DIR...  # copy outputs back
set -euo pipefail
RUN_DIR="${RUN_DIR:-/home/ilay.kamai/dynamo_run}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ "${1:-}" = "--collect" ]; then
  shift
  for d in "$@"; do
    mkdir -p "${REPO}/${d}"
    cp -r "${RUN_DIR}/${d}/." "${REPO}/${d}/"
    echo "collected ${RUN_DIR}/${d} -> ${REPO}/${d}"
  done
  exit 0
fi
mkdir -p "${RUN_DIR}"
git -C "${REPO}" archive HEAD | tar -x -C "${RUN_DIR}"
for d in "$@"; do
  mkdir -p "${RUN_DIR}/${d}"
  cp "${REPO}/${d}/simulation_properties.csv" "${RUN_DIR}/${d}/"
done
echo "staged $(git -C "${REPO}" rev-parse --short HEAD) -> ${RUN_DIR} ($*)"
