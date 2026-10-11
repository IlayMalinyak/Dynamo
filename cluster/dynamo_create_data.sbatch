#!/bin/bash
# ============================================================================
# Dynamo dataset generation (CPU only) — runs Dynamo.create_data on one dataset
# directory. If DATASET_DIR/simulation_properties.csv exists it is reused
# (e.g. a paired table from scripts/make_paired_tables.py); otherwise a new
# table of NUM stars is drawn.
#
#   sbatch cluster/dynamo_create_data.sbatch DATASET_DIR [NDAYS] [NUM] [EXTRA_ARGS...]
#
# Faculae smoke test (two arms, identical spot maps):
#   python scripts/make_paired_tables.py --n 200 --out dataset_faculae_smoke
#   sbatch cluster/dynamo_create_data.sbatch dataset_faculae_smoke 1000
#   sbatch cluster/dynamo_create_data.sbatch dataset_faculae_smoke_noQ 1000
#
# Uses the PHOENIX models already in ~/.dynamo (no downloads) and the same
# container the code was developed/tested in (numba + kiauhoku in ~/.local).
#
#SBATCH --job-name=dynamo_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --partition=l40s-shared
#SBATCH --qos=12h_4g
#SBATCH -t 12:00:00
#SBATCH --output=/home/ilay.kamai/logs/%x-%j.out
#SBATCH --error=/home/ilay.kamai/logs/%x-%j.err

set -uo pipefail
unset SLURM_TRES_PER_TASK SLURM_CPUS_PER_GPU 2>/dev/null || true

DYNAMO_DIR="/home/ilay.kamai/work/Dynamo"
SIF="/apps/apptainer/images/pytorch/latest/torch2.7-tf2.20-cu128.sif"

DATASET_DIR="${1:?usage: sbatch dynamo_create_data.sbatch DATASET_DIR [NDAYS] [NUM] [EXTRA...]}"
NDAYS="${2:-1000}"
NUM="${3:-200}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA="$*"
NCPU="${SLURM_CPUS_PER_TASK:-32}"
PLOT_DIR="${DATASET_DIR}/images"

echo "[dynamo] dataset=${DATASET_DIR} ndays=${NDAYS} num=${NUM} ncpu=${NCPU} extra=${EXTRA}"
[ -f "${DYNAMO_DIR}/${DATASET_DIR}/simulation_properties.csv" ] \
  && echo "[dynamo] reusing existing simulation_properties.csv" \
  || echo "[dynamo] no table found: drawing ${NUM} new stars"

# NO single quotes inside the srun payload (apptainer-oci wrapper mangles them)
srun --ntasks=1 --cpu-bind=none --container="${SIF}" \
  bash -lc "cd ${DYNAMO_DIR} && mkdir -p ${PLOT_DIR} && export PYTHONUNBUFFERED=1 MPLBACKEND=Agg NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 && \
            python3 -m Dynamo.create_data --dataset_dir ${DATASET_DIR} --plot_dir ${PLOT_DIR} --plot_every 50 \
              --num_simulations ${NUM} --ndays ${NDAYS} --n_cpu ${NCPU} --batch_size 20 --skip_download ${EXTRA}"
rc=$?
echo "[dynamo] create_data exit code ${rc}"
exit ${rc}
