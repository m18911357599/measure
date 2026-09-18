$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$bash = Get-Command bash -ErrorAction SilentlyContinue
if ($bash) {
  & bash scripts/fetch_hccl.sh @args
  exit $LASTEXITCODE
}
New-Item -ItemType Directory -Force -Path .hccl-src | Out-Null
if (Test-Path .hccl-src/cann-hccl/.git) {
  git -C .hccl-src/cann-hccl fetch --depth 1 origin master
  git -C .hccl-src/cann-hccl checkout --force FETCH_HEAD
} else {
  if (Test-Path .hccl-src/cann-hccl) { Remove-Item -Recurse -Force .hccl-src/cann-hccl }
  git clone --depth 1 -b master https://gitcode.com/cann/hccl.git .hccl-src/cann-hccl
}
