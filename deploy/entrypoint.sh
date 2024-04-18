#!/usr/bin/env bash
set -euo pipefail

#
# Entrypoint script for Docker image
#

function get_default_ip {
  ifname="$(ip route show default | grep -Eom 1 'dev \w+' | awk '{print $2}')"
  ip addr show "$ifname" | grep -m 1 inet | awk '{print $2}' | sed 's:/.*::g'
}

RUN_MODE="${RUN_MODE:-"${1}"}"
HOST_IP="${HOST_IP:-"$(get_default_ip)"}"
PORT="${PORT:8080}"
TASK_NAME="${TASK_NAME:-"indexing"}"

if [ "${RUN_MODE}" == "server" ]; then
  uvicorn "image_search.app.api:app" --host "${HOST_IP}" --port "${PORT}"
elif [ "${RUN_MODE}" == "task-queue" ]; then
  celery -A "image_search.app.tasks:app" worker
else
  echo "Invalid run mode: '${RUN_MODE}'." >&2
  exit 1
fi
