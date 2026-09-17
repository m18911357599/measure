#!/usr/bin/env bash
# Push the cann/hccl + analysis bundle to github.com/m18911357599/hccl.
# Requires write access to that repo (this Cloud Agent token does not have it).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="${ROOT}/docs/hccl/m18911357599_hccl_sync.bundle"
BRANCH="cursor/sync-cann-hccl-analysis-8203"
DEST="${1:-https://github.com/m18911357599/hccl.git}"

if [[ ! -f "${BUNDLE}" ]]; then
  echo "missing bundle: ${BUNDLE}" >&2
  exit 1
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

git clone "${DEST}" "${WORKDIR}/hccl"
cd "${WORKDIR}/hccl"
git fetch "${BUNDLE}" "${BRANCH}:${BRANCH}"
git checkout "${BRANCH}"
git push -u origin "${BRANCH}"
echo "pushed ${BRANCH} to ${DEST}"
