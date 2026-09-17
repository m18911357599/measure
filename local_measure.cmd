@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 scripts\measure.py local %*
  exit /b %ERRORLEVEL%
)
where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python scripts\measure.py local %*
  exit /b %ERRORLEVEL%
)
echo Python not found. Install Python 3 and retry.
exit /b 1
