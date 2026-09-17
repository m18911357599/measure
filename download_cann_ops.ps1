Set-Location -LiteralPath $PSScriptRoot
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
  & py -3 "$PSScriptRoot\scripts\download_cann_ops.py" --dest "$PSScriptRoot" @args
  exit $LASTEXITCODE
}
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
  & python "$PSScriptRoot\scripts\download_cann_ops.py" --dest "$PSScriptRoot" @args
  exit $LASTEXITCODE
}
Write-Error "Python not found. Install Python 3 and retry."
exit 1
