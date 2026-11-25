#!/bin/bash
# DKM-A / DKM-B scenarijus viename GPU mazge: paleidžiamas vLLM serveris (localhost),
# palaukiama, kol jis pasiruoš, įvykdomas scenarijus ir serveris išjungiamas.
# Taip nereikia žinoti mazgo vardo: .env lieka DKM_*_BASE_URL=http://localhost:800x/v1.
#
# VU MIF HPC: gpu eilė = NVIDIA DGX-1, 8 x V100 32 GB (sm_70) -> vLLM < 0.20, --dtype half (V100 neturi bf16).
# Naudojimas (paleisti iš /scratch/lustre/home/$USER/care-abm):
#   sbatch --export=ALL,DKM=DKM-A,SCEN=S5 slurm/llm_scenario.sbatch                       # 7B, 1 GPU
#   sbatch --export=ALL,DKM=DKM-B,SCEN=S6,TP=4 --gres=gpu:4 slurm/llm_scenario.sbatch     # 32B fp16, 4 GPU
# Modelio id imamas iš .env (DKM_A_MODEL / DKM_B_MODEL).
#SBATCH --job-name=careabm-llm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

: "${DKM:?nurodykite DKM=DKM-A arba DKM-B}"
: "${SCEN:?nurodykite SCEN, pvz. S5}"
TP=${TP:-1}
DTYPE=${DTYPE:-half}                        # V100: fp16
WORKERS=${WORKERS:-4}

source "${VENV:-$HOME/venvs/careabm}/bin/activate"
export HF_HOME=${HF_HOME:-/scratch/lustre/home/$USER/hf_cache}   # svoriai bendroje failų sistemoje
export HF_HUB_OFFLINE=1                       # svoriai parsiųsti iš anksto prisijungimo mazge

# modelio id ir portas iš .env
set -a; source .env; set +a
case "${DKM}" in
  DKM-A) MODEL_ID="${DKM_A_MODEL}"; PORT=8000; export DKM_A_BASE_URL="http://localhost:${PORT}/v1" ;;
  DKM-B) MODEL_ID="${DKM_B_MODEL}"; PORT=8001; export DKM_B_BASE_URL="http://localhost:${PORT}/v1" ;;
  *) echo "Nežinomas DKM: ${DKM}"; exit 1 ;;
esac
EXTRA=()
[[ "${MODEL_ID}" == *AWQ* || "${MODEL_ID}" == *awq* ]] && EXTRA+=(--quantization awq)

echo "Mazgas: $(hostname)  Modelis: ${MODEL_ID}  TP=${TP}"
nvidia-smi --query-gpu=name,memory.total --format=csv
vllm serve "${MODEL_ID}" --port "${PORT}" --tensor-parallel-size "${TP}" \
     --max-model-len 8192 --dtype "${DTYPE}" --seed 0 "${EXTRA[@]}" > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

# laukiama, kol serveris atsakys (svorių atsisiuntimas pirmą kartą gali užtrukti)
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null; then echo "vLLM pasiruošęs po ${i}0 s"; break; fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "vLLM nukrito, žr. logs/vllm_${SLURM_JOB_ID}.log"; exit 1; fi
  sleep 10
done
curl -sf "http://localhost:${PORT}/v1/models" > /dev/null || { echo "vLLM neatsakė per 30 min."; exit 1; }

# versijos darbo lentelėms (tab:env, tab:llms)
python -c "import vllm, numpy, scipy, pandas, sys; print('python', sys.version.split()[0], 'vllm', vllm.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)"

python -m care_abm.run_experiment --scenario "${SCEN}" --model "${DKM}" --workers "${WORKERS}" ${RUN_ARGS:-}
