#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
SUPERVISOR="${PROJECT_ROOT}/video_quant_lab/baselines/fp_quant/scripts/monitor_wan_baseline_pipeline.py"
PYTHON="${PIPELINE_PYTHON:-python3}"
RUNTIME_DIR="${PROJECT_ROOT}/vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0/pipeline"
PID_FILE="${RUNTIME_DIR}/supervisor.pid"
LOG_FILE="${RUNTIME_DIR}/supervisor.log"
LOCK_FILE="${RUNTIME_DIR}/supervisor.lock"
mkdir -p "${RUNTIME_DIR}"

is_running() {
    [[ -s "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null
}

case "${1:-status}" in
    start|resume)
        if is_running; then
            echo "pipeline already running: pid $(<"${PID_FILE}")"
            exit 0
        fi
        nohup setsid flock -n "${LOCK_FILE}" "${PYTHON}" "${SUPERVISOR}" run \
            >>"${LOG_FILE}" 2>&1 </dev/null &
        pid=$!
        echo "${pid}" >"${PID_FILE}"
        sleep 1
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "pipeline failed to start; inspect ${LOG_FILE}" >&2
            exit 1
        fi
        echo "pipeline started: pid ${pid}"
        echo "log: ${LOG_FILE}"
        ;;
    status)
        if is_running; then echo "supervisor: running (pid $(<"${PID_FILE}"))"; else echo "supervisor: stopped"; fi
        "${PYTHON}" "${SUPERVISOR}" status
        ;;
    logs)
        tail -n "${LOG_LINES:-80}" "${LOG_FILE}"
        ;;
    stop)
        if ! is_running; then
            echo "pipeline is not running"
            exit 0
        fi
        pid="$(<"${PID_FILE}")"
        kill -TERM -- "-${pid}"
        echo "stop requested for pipeline pid ${pid}"
        ;;
    *)
        echo "usage: $0 {start|status|logs|stop|resume}" >&2
        exit 2
        ;;
esac
