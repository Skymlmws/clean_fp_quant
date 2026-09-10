#!/usr/bin/env bash
set -euo pipefail

DESTINATION="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/vbench-official"
REVISION="fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
BASE_URL="https://raw.githubusercontent.com/Vchitect/VBench/${REVISION}"
mkdir -p "${DESTINATION}"
curl -L --fail --silent --show-error \
    "${BASE_URL}/vbench/VBench_full_info.json" \
    -o "${DESTINATION}/VBench_full_info.json"
curl -L --fail --silent --show-error \
    "${BASE_URL}/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt" \
    -o "${DESTINATION}/all_dimension_aug_wanx_seed42.txt"
cd "${DESTINATION}"
(
    echo "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd  VBench_full_info.json"
    echo "54fb940aea8f1ede1e8728f62d3e67efacdd7b341fb8a6c91308bcba874d5d5d  all_dimension_aug_wanx_seed42.txt"
) | sha256sum --check
