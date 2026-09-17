@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 scripts\download_cann_ops.py --dest "%cd%"
  exit /b %ERRORLEVEL%
)
where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python scripts\download_cann_ops.py --dest "%cd%"
  exit /b %ERRORLEVEL%
)
echo Python not found.
exit /b 1
