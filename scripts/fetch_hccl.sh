#!/usr/bin/env bash
# Fetch HCCL source into this checkout (not committed; see .gitignore .hccl-src/).
#
# Default: gitcode.com/cann/hccl  (algorithm source of record)
# Optional: github.com/m18911357599/hccl (working copy)
#
# Usage:
#   scripts/fetch_hccl.sh
#   scripts/fetch_hccl.sh --dest /tmp/hccl-src
#   scripts/fetch_hccl.sh --github
#   scripts/fetch_hccl.sh --github-only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${HCCL_DEST:-$ROOT/.hccl-src}"
CANN_URL="${HCCL_CANN_URL:-https://gitcode.com/cann/hccl.git}"
GITHUB_URL="${HCCL_GITHUB_URL:-https://github.com/m18911357599/hccl.git}"
CANN_REF="${HCCL_REF:-master}"
GITHUB_REF="${HCCL_GITHUB_REF:-main}"
DEPTH="${HCCL_DEPTH:-1}"
FETCH_CANN=1
FETCH_GITHUB=0

usage() {
  cat <<'EOF'
Fetch HCCL source into .hccl-src/ (gitignored).

Options:
  --dest DIR         clone root (default: <repo>/.hccl-src)
  --ref REF          cann/hccl branch or tag (default: master)
  --github-ref REF   m18911357599/hccl branch (default: main)
  --depth N          clone/fetch depth (default: 1)
  --github           also clone github.com/m18911357599/hccl
  --github-only      only clone github.com/m18911357599/hccl
  --cann-only        only clone gitcode.com/cann/hccl (default)
  -h, --help         show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      DEST="$2"
      shift 2
      ;;
    --ref)
      CANN_REF="$2"
      shift 2
      ;;
    --github-ref)
      GITHUB_REF="$2"
      shift 2
      ;;
    --depth)
      DEPTH="$2"
      shift 2
      ;;
    --github)
      FETCH_GITHUB=1
      shift
      ;;
    --github-only)
      FETCH_CANN=0
      FETCH_GITHUB=1
      shift
      ;;
    --cann-only)
      FETCH_CANN=1
      FETCH_GITHUB=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

clone_or_update() {
  local url="$1"
  local dir="$2"
  local ref="$3"
  mkdir -p "$(dirname "$dir")"
  if [[ -d "$dir/.git" ]]; then
    echo "+ git -C $dir fetch --depth $DEPTH origin $ref"
    if git -C "$dir" fetch --depth "$DEPTH" origin "$ref"; then
      git -C "$dir" checkout --force FETCH_HEAD
    else
      echo "fetch $ref failed, trying origin HEAD" >&2
      git -C "$dir" fetch --depth "$DEPTH" origin
      git -C "$dir" checkout --force FETCH_HEAD
    fi
  else
    rm -rf "$dir"
    echo "+ git clone --depth $DEPTH --branch $ref $url $dir"
    if ! git clone --depth "$DEPTH" --branch "$ref" "$url" "$dir"; then
      echo "branch $ref missing, cloning default HEAD" >&2
      git clone --depth "$DEPTH" "$url" "$dir"
    fi
  fi
  echo "synced $dir @ $(git -C "$dir" rev-parse --short HEAD) $(git -C "$dir" log -1 --format=%s)"
}

mkdir -p "$DEST"
if [[ "$FETCH_CANN" -eq 1 ]]; then
  clone_or_update "$CANN_URL" "$DEST/cann-hccl" "$CANN_REF"
fi
if [[ "$FETCH_GITHUB" -eq 1 ]]; then
  clone_or_update "$GITHUB_URL" "$DEST/hccl" "$GITHUB_REF"
fi

{
  echo "fetched_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ -d "$DEST/cann-hccl/.git" ]]; then
    echo "cann_url=$CANN_URL"
    echo "cann_sha=$(git -C "$DEST/cann-hccl" rev-parse HEAD)"
    echo "cann_subject=$(git -C "$DEST/cann-hccl" log -1 --format=%s)"
  fi
  if [[ -d "$DEST/hccl/.git" ]]; then
    echo "github_url=$GITHUB_URL"
    echo "github_sha=$(git -C "$DEST/hccl" rev-parse HEAD)"
    echo "github_subject=$(git -C "$DEST/hccl" log -1 --format=%s)"
  fi
} >"$DEST/SOURCE.txt"
echo "wrote $DEST/SOURCE.txt"
cat "$DEST/SOURCE.txt"
