@echo off

pushd %~dp0

set "DEPENDS_ON="
for /F "tokens=1* delims==" %%a in ('findstr /R "^DEPENDS_ON=" .env 2^>nul') do set "DEPENDS_ON=%%b"

echo Current directory: %CD% && echo Memory Backend: %DEPENDS_ON%

python scripts\check_requirements.py requirements.txt
if %errorlevel% == 1 (
    echo Installing missing packages...
    pip install -r requirements.txt || goto :pip_failed
)

set docker-compose-file=docker-compose.yml

if "%DEPENDS_ON%" == "local" set docker-compose-file=docker-compose-local.yml

if "%DEPENDS_ON%" == "redis" set docker-compose-file=docker-compose.redis.yml

if "%DEPENDS_ON%" == "weaviate" set docker-compose-file=docker-compose-weaviate.yml

echo Starting containers...
docker-compose -f %docker-compose-file% up -d --remove-orphans

echo Starting Autogpt..
python -m autogpt %*

echo Stopping all running containers...
for /f "tokens=*" %%i in ('docker ps -q') do docker stop -t 10 %%i

popd
exit /b

:pip_failed
echo Failed to install missing packages. Exiting...
exit /b 1

