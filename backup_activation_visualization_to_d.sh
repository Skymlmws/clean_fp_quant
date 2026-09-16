#!/usr/bin/env bash
set -Eeuo pipefail

# Run this script in WSL on the local Windows computer, preferably inside tmux.
# It downloads the activation visualizations to drive D:, verifies every file,
# and optionally removes the server copy only after an explicit confirmation.

readonly SERVER="maoliming@mars"
readonly SSH_PORT="22022"
readonly REMOTE_DIR="/home/maoliming/FP-Quant/outputs/activation-visualization"
readonly LOCAL_DIR="/mnt/d/FP-Quant-backup/activation-visualization"
readonly SSH_CMD="ssh -p ${SSH_PORT}"

for command_name in ssh rsync df awk mktemp; do
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
remote_bytes="$(${SSH_CMD} "${SERVER}" "test -d '${REMOTE_DIR}' && du -sb '${REMOTE_DIR}' | cut -f1")"
if [[ ! "${remote_bytes}" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the server directory size." >&2
    exit 1
fi

available_bytes="$(df -PB1 /mnt/d | awk 'NR == 2 {print $4}')"
echo "Server source: ${REMOTE_DIR}"
echo "Source bytes:  ${remote_bytes}"
echo "D: free bytes: ${available_bytes}"

mkdir -p "${LOCAL_DIR}"

echo "Starting resumable transfer..."
rsync -rltvh \
    --no-perms \
    --no-owner \
    --no-group \
    --partial \
    --append-verify \
    --info=progress2 \
    -e "${SSH_CMD}" \
    "${SERVER}:${REMOTE_DIR}/" \
    "${LOCAL_DIR}/"

echo "Transfer finished. Running full checksum verification..."
verification_log="$(mktemp /tmp/fp-quant-rsync-verify.XXXXXX)"
trap 'rm -f -- "${verification_log}"' EXIT

rsync -rlthnc \
    --no-perms \
    --no-owner \
    --no-group \
    --delete \
    --itemize-changes \
    -e "${SSH_CMD}" \
    "${SERVER}:${REMOTE_DIR}/" \
    "${LOCAL_DIR}/" >"${verification_log}"

if [[ -s "${verification_log}" ]]; then
    echo "Verification found differences. Nothing was deleted from the server." >&2
    cat "${verification_log}" >&2
    echo "Run this script again to resume or repair the copy." >&2
    exit 1
fi

local_bytes="$(du -sb "${LOCAL_DIR}" | cut -f1)"
local_files="$(find "${LOCAL_DIR}" -type f | wc -l)"
echo "Checksum verification passed."
echo "Local bytes: ${local_bytes}"
echo "Local files: ${local_files}"
echo
echo "The verified server directory may now be permanently deleted:"
echo "  ${REMOTE_DIR}"
read -r -p "Type DELETE to remove it, or press Enter to keep it: " confirmation

if [[ "${confirmation}" != "DELETE" ]]; then
    echo "Server files were kept."
    exit 0
fi

${SSH_CMD} "${SERVER}" \
    "test -d '${REMOTE_DIR}' && rm -rf -- '${REMOTE_DIR}' && sync && df -h '/home/maoliming/FP-Quant'"

echo "Server copy deleted after successful checksum verification."
echo "Local backup: ${LOCAL_DIR}"
