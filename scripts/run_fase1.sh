#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

pipe=(.venv/bin/python -m scripts.eval_ocr run)
paddle=(.venv-legacy/bin/python -m scripts.eval_ocr run)
# CPU 0 dan 1 adalah hyperthread dari satu P-core, jadi 0,2 = dua core fisik
two_cores=(taskset -c 0,2)

"${pipe[@]}" --engine gt-text
"${pipe[@]}" --engine rapid-cls
"${pipe[@]}" --engine rapid-cls --steps gray
"${pipe[@]}" --engine rapid-cls --steps deskew,gray
"${pipe[@]}" --engine rapid-cls --steps crop
"${pipe[@]}" --engine rapid-cls --steps crop,deskew,gray
taskset -c 0 "${pipe[@]}" --engine rapid-cls --threads 1
"${two_cores[@]}" "${pipe[@]}" --engine rapid-cls --threads 2
taskset -c 0,2,4,6 "${pipe[@]}" --engine rapid-cls --threads 4
"${two_cores[@]}" "${pipe[@]}" --engine rapid-cls --threads 2 --steps crop,deskew,gray
"${paddle[@]}" --engine paddle-tiny
"${paddle[@]}" --engine paddle-small
"${two_cores[@]}" "${paddle[@]}" --engine paddle-tiny --threads 2
"${two_cores[@]}" "${paddle[@]}" --engine paddle-small --threads 2
"${two_cores[@]}" "${paddle[@]}" --engine paddle-tiny --threads 2 --steps resize
"${two_cores[@]}" "${paddle[@]}" --engine paddle-tiny --threads 2 --steps resize,deskew,gray
taskset -c 0 "${pipe[@]}" --engine rapid-cls --threads 1 --steps deskew,gray
"${pipe[@]}" --engine rapid
taskset -c 0 "${pipe[@]}" --engine rapid --threads 1
"${two_cores[@]}" "${pipe[@]}" --engine rapid --threads 2
taskset -c 0 "${pipe[@]}" --engine rapid --threads 1 --steps deskew,gray
"${two_cores[@]}" "${pipe[@]}" --engine rapid --threads 2 --steps deskew,gray
"${two_cores[@]}" "${pipe[@]}" --engine rapid --threads 2 --steps crop,deskew,gray
