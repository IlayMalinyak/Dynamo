#!/bin/bash
# ============================================================================
# Dynamo dataset generation (CPU only) — runs Dynamo.create_data on one dataset
# directory as a job ARRAY: task k simulates table rows [k*ROWS, (k+1)*ROWS).
# Athena only allows CPU-only jobs in qos 4h_0g (<=4h, 8 CPUs, 16G), so the
# table is split into small slices. DATASET_DIR/simulation_properties.csv
# must already exist (e.g. from scripts/make_paired_tables.py); every task
# reads the same table, and chunk files are named by slice start (no clashes).
#
#   sbatch --array=0-$((NUM/ROWS-1)) cluster/dynamo_create_data.sbatch DATASET_DIR [NDAYS] [ROWS] [EXTRA_ARGS...]
#
# Faculae smoke test (two arms, identical spot maps, 200 stars = 8 x 25):
#   python scripts/make_paired_tables.py --n 200 --out dataset_faculae_smoke
#   sbatch --array=0-7 cluster/dynamo_create_data.sbatch dataset_faculae_smoke 1000
#   sbatch --array=0-7 cluster/dynamo_create_data.sbatch dataset_faculae_smoke_noQ 1000
#
# Spot-coverage calibration (docs/spot_coverage.md; four paired arms of 200 stars;
# the f0290 arm has up to 10000 spots per star (~3.5 min per star), so it uses half-size slices):
#   python scripts/make_coverage_tables.py --n 200 --out dataset_cov
#   bash cluster/stage_run_dir.sh dataset_cov_legacy dataset_cov_f0030 dataset_cov_f0100 dataset_cov_f0290
#   sbatch --array=0-7  cluster/dynamo_create_data.sbatch dataset_cov_legacy 1000 25
#   sbatch --array=0-7  cluster/dynamo_create_data.sbatch dataset_cov_f0030 1000 25
#   sbatch --array=0-7  cluster/dynamo_create_data.sbatch dataset_cov_f0100 1000 25
#   sbatch --array=0-15 cluster/dynamo_create_data.sbatch dataset_cov_f0290 1000 13
#
# l40s-public = athena-post only, which does not see /rg (cd failed on all 16
# tasks of the first submission). So the job runs from a copy under /home:
#   bash cluster/stage_run_dir.sh      # code (git HEAD) + dataset tables -> RUN_DIR
#   sbatch --array=0-7 ...             # outputs land in RUN_DIR/DATASET_DIR
#   bash cluster/stage_run_dir.sh --collect DATASET_DIR   # copy results back
#
# Uses the PHOENIX models already in ~/.dynamo (no downloads) and the same
# container the code was developed/tested in (numba + kiauhoku in ~/.local).
#
#SBATCH --job-name=dynamo_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=l40s-public
#SBATCH --qos=4h_0g
#SBATCH -t 04:00:00
#SBATCH --output=/home/ilay.kamai/logs/%x-%A_%a.out
#SBATCH --error=/home/ilay.kamai/logs/%x-%A_%a.err

set -uo pipefail
unset SLURM_TRES_PER_TASK SLURM_CPUS_PER_GPU 2>/dev/null || true

DYNAMO_DIR="${RUN_DIR:-/home/ilay.kamai/dynamo_run}"   # /home copy; athena-post cannot see /rg
SIF="/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif"

DATASET_DIR="${1:?usage: sbatch --array=0-K dynamo_create_data.sbatch DATASET_DIR [NDAYS] [ROWS] [EXTRA...]}"
NDAYS="${2:-1000}"
ROWS="${3:-25}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA="$*"
NCPU="${SLURM_CPUS_PER_TASK:-8}"
PLOT_DIR="${DATASET_DIR}/images"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK * ROWS ))
STOP=$(( START + ROWS ))

echo "[dynamo] host=$(hostname)"
if [ ! -d "${DYNAMO_DIR}" ] || [ ! -d "${HOME}/.dynamo/models/Phoenix_mu" ]; then
  echo "[dynamo] ERROR: repo or ~/.dynamo not visible on $(hostname)"
  exit 2
fi
echo "[dynamo] dataset=${DATASET_DIR} ndays=${NDAYS} rows=[${START},${STOP}) ncpu=${NCPU} extra=${EXTRA}"
if [ ! -f "${DYNAMO_DIR}/${DATASET_DIR}/simulation_properties.csv" ]; then
  echo "[dynamo] ERROR: ${DATASET_DIR}/simulation_properties.csv missing; build the table first"
  exit 1
fi

# The l40s-public container (athena-post) does not get /home/ilay.kamai or /rg
# mounted automatically, so bind them explicitly. Prefer a direct apptainer
# exec (explicit --bind); fall back to the srun OCI wrapper + APPTAINER_BIND.
BINDS="/home/ilay.kamai,/rg/perets_prj/ilay.kamai"
# NO single quotes inside the payload (apptainer-oci wrapper mangles them)
PAYLOAD="cd ${DYNAMO_DIR} || { echo [dynamo] ERROR: ${DYNAMO_DIR} not visible inside container; echo HOME=\$HOME; cat /proc/self/mountinfo | cut -d\  -f5 | sort -u | head -40; exit 2; }; \
  mkdir -p ${PLOT_DIR} && export HOME=/home/ilay.kamai PYTHONUNBUFFERED=1 MPLBACKEND=Agg NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 && \
  python3 -m Dynamo.create_data --dataset_dir ${DATASET_DIR} --plot_dir ${PLOT_DIR} --plot_every 50 \
    --ndays ${NDAYS} --n_cpu ${NCPU} --batch_size ${ROWS} --start ${START} --stop ${STOP} --skip_download ${EXTRA}"

if command -v apptainer >/dev/null 2>&1; then
  echo "[dynamo] launcher: apptainer exec --bind ${BINDS}"
  apptainer exec --bind "${BINDS}" --env HOME=/home/ilay.kamai "${SIF}" bash -lc "${PAYLOAD}"
else
  echo "[dynamo] launcher: srun --container (APPTAINER_BIND=${BINDS})"
  export APPTAINER_BIND="${BINDS}" SINGULARITY_BIND="${BINDS}"
  srun --ntasks=1 --cpu-bind=none --container="${SIF}" bash -lc "${PAYLOAD}"
fi
rc=$?
echo "[dynamo] create_data exit code ${rc}"
exit ${rc}
