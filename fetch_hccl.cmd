@echo off
setlocal
cd /d "%~dp0"
bash scripts\fetch_hccl.sh %*
if %ERRORLEVEL%==0 exit /b 0
where git >nul 2>&1
if errorlevel 1 (
  echo git not found.
  exit /b 1
)
echo retrying with git bash unavailable; using git clone directly
if not exist .hccl-src mkdir .hccl-src
git clone --depth 1 -b master https://gitcode.com/cann/hccl.git .hccl-src\cann-hccl
exit /b %ERRORLEVEL%
