Set-Location -LiteralPath $PSScriptRoot
$src = $env:SIMT_SRC
if (-not $src) { $src = "D:\simt" }
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
  & py -3 "$PSScriptRoot\scripts\download_cann_ops.py" --skip-cann --skip-super --simt-local $src --dest "$PSScriptRoot" @args
  exit $LASTEXITCODE
}
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
  & python "$PSScriptRoot\scripts\download_cann_ops.py" --skip-cann --skip-super --simt-local $src --dest "$PSScriptRoot" @args
  exit $LASTEXITCODE
}
Write-Error "Python not found. Install Python 3 and retry."
exit 1
