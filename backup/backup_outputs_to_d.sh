#!/usr/bin/env bash
set -Eeuo pipefail

# Run this script in WSL on the local Windows computer, preferably inside tmux.
# It downloads outputs to drive D:, excluding the run that was backed up before,
# and verifies every transferred file. It never deletes files from the server.

readonly SERVER="maoliming@mars"
readonly SSH_PORT="22022"
readonly REMOTE_DIR="/home/maoliming/FP-Quant/outputs"
readonly EXCLUDED_DIR="autumn_station-seed0-full"
readonly LOCAL_DIR="/mnt/d/FP-Quant-backup/outputs"
readonly SSH_TRANSPORT="ssh -p ${SSH_PORT}"

for command_name in ssh rsync df awk mktemp find wc; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
    fi
done

if [[ ! -d /mnt/d ]]; then
    echo "D: is not mounted at /mnt/d. Run this script inside WSL." >&2
    exit 1
fi

echo "Checking the server source..."
remote_stats="$(ssh -p "${SSH_PORT}" "${SERVER}" \
    "test -d '${REMOTE_DIR}' || exit 1
     total=\$(du -sb '${REMOTE_DIR}' | cut -f1)
     excluded=0
     if test -d '${REMOTE_DIR}/${EXCLUDED_DIR}'; then
         excluded=\$(du -sb '${REMOTE_DIR}/${EXCLUDED_DIR}' | cut -f1)
     fi
     printf '%s %s\n' \"\$total\" \"\$excluded\"")"
read -r remote_total_bytes excluded_bytes <<<"${remote_stats}"

if [[ ! "${remote_total_bytes}" =~ ^[0-9]+$ || ! "${excluded_bytes}" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the server directory sizes." >&2
    exit 1
fi

required_bytes=$((remote_total_bytes - excluded_bytes))
available_bytes="$(df -PB1 /mnt/d | awk 'NR == 2 {print $4}')"

echo "Server source:  ${REMOTE_DIR}"
echo "Excluded:       ${REMOTE_DIR}/${EXCLUDED_DIR}"
echo "Download bytes: ${required_bytes}"
echo "D: free bytes:  ${available_bytes}"

if (( available_bytes < required_bytes )); then
    echo "Not enough free space on D:." >&2
    exit 1
fi

mkdir -p "${LOCAL_DIR}"

echo "Starting resumable transfer..."
rsync -rltvh \
    --no-perms \
    --no-owner \
    --no-group \
    --partial \
    --append-verify \
    --info=progress2 \
    --exclude="/${EXCLUDED_DIR}/" \
    -e "${SSH_TRANSPORT}" \
    "${SERVER}:${REMOTE_DIR}/" \
    "${LOCAL_DIR}/"

echo "Transfer finished. Running full checksum verification..."
verification_log="$(mktemp /tmp/fp-quant-outputs-verify.XXXXXX)"
trap 'rm -f -- "${verification_log}"' EXIT

rsync -rlthnc \
    --no-perms \
    --no-owner \
    --no-group \
    --delete \
    --itemize-changes \
    --exclude="/${EXCLUDED_DIR}/" \
    -e "${SSH_TRANSPORT}" \
    "${SERVER}:${REMOTE_DIR}/" \
    "${LOCAL_DIR}/" >"${verification_log}"

if [[ -s "${verification_log}" ]]; then
    echo "Verification found differences." >&2
    cat "${verification_log}" >&2
    echo "Run this script again to resume or repair the copy." >&2
    exit 1
fi

local_files="$(find "${LOCAL_DIR}" \
    -path "${LOCAL_DIR}/${EXCLUDED_DIR}" -prune -o \
    -type f -print | wc -l)"

echo "Checksum verification passed."
echo "Local files: ${local_files}"
echo "Local backup: ${LOCAL_DIR}"
echo "Server files were kept."
