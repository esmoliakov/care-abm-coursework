#!/bin/bash
# Atviro modelio aptarnavimas OpenAI-suderinamu serveriu (vLLM) – DKM-A / DKM-B.
# Naudojimas (GPU mazge):  bash slurm/serve_vllm.sh <HF_MODELIO_ID> <PORTAS> [TENSOR_PARALLEL]
#   pvz.:  bash slurm/serve_vllm.sh <7B_instruct_modelis> 8000
#          bash slurm/serve_vllm.sh <30B_plius_instruct_modelis> 8001 2
# Tada configs/models.yaml: base_url: http://<mazgo_vardas>:<portas>/v1,  model: <tas pats HF id>
# Pastaba: modelio versiją (revision) užfiksuokite ir įrašykite į darbo lentelę.
set -euo pipefail
MODEL=${1:?HF modelio id}; PORT=${2:-8000}; TP=${3:-1}
pip install -q vllm
vllm serve "${MODEL}" --port "${PORT}" --tensor-parallel-size "${TP}" --max-model-len 8192 \
     --dtype auto --seed 0
# Kvantizuotam modeliui pridėkite: --quantization awq   (jei modelis tokio formato)
