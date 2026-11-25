#!/bin/bash
# SLURM užduočių masyvas: kiekviena užduotis vykdo dalį scenarijaus pakartojimų (nepriklausomi paleidimai).
# Naudojimas:  sbatch --export=SCEN=S3 slurm/run_array.sbatch      (S1–S4, S13, S14 ir kt. be DKM)
#              sbatch --export=SCEN=S5,MODEL=DKM-A slurm/run_array.sbatch   (DKM scenarijams reikia veikiančio vLLM)
#SBATCH --job-name=careabm
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs results
CHUNK=${CHUNK:-10}                         # pakartojimų skaičius vienai užduočiai
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
# source ~/venvs/careabm/bin/activate      # jūsų aplinka
MODEL_ARG=""; [ -n "${MODEL:-}" ] && MODEL_ARG="--model ${MODEL}"
python -m care_abm.run_experiment --scenario "${SCEN}" ${MODEL_ARG} \
    --start ${START} --repeats ${CHUNK} --workers ${SLURM_CPUS_PER_TASK:-4}
