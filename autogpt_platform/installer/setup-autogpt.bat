@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM AutoGPT Windows Setup
REM ----------------------------------------------------------------------------
REM Sets up AutoGPT on Windows.  Linux/macOS users: setup-autogpt.sh.
REM
REM Optional flags:
REM   /with-ollama           Install Ollama (via winget), pull a default chat
REM                          model, and wire backend\.env so AutoPilot runs
REM                          without any cloud API keys (CHAT_USE_LOCAL=true).
REM                          See docs/platform/copilot-local-llm.md.
REM   /ollama-model=NAME     Model to pull (default: hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M).
REM   /ollama-host=URL       Use an existing Ollama at this URL instead of
REM                          installing one locally. Skips the Ollama install
REM                          but still writes the CHAT_USE_LOCAL .env entries.
REM                          Example: /ollama-host=http://gpu-rig.lab:11434
REM ============================================================================

REM --- Variables ---
set SCRIPT_DIR=%~dp0
set REPO_DIR=%SCRIPT_DIR%..\..
set CLONE_NEEDED=0
set LOG_FILE=
set WITH_OLLAMA=0
set OLLAMA_MODEL=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
set OLLAMA_HOST_URL=

REM --- Parse args ---
REM NOTE: cmd.exe treats "=" as an argument delimiter (like space / comma /
REM semicolon), so "/ollama-model=foo" arrives as TWO tokens:
REM %1="/ollama-model", %2="foo". We therefore accept BOTH the "=" form
REM (present when the arg is quoted or otherwise not split) AND the bare
REM flag + next-token form, shifting an extra time for the latter so the
REM value isn't reparsed as a flag. (The .sh has no such issue — bash keeps
REM "--ollama-model=foo" as a single word.)
:parse_args
if "%~1"=="" goto args_done
set ARG=%~1
if /I "%ARG%"=="/with-ollama" (
    set WITH_OLLAMA=1
) else if /I "%ARG:~0,14%"=="/ollama-model=" (
    set OLLAMA_MODEL=%ARG:~14%
    set WITH_OLLAMA=1
) else if /I "%ARG%"=="/ollama-model" (
    set "OLLAMA_MODEL=%~2"
    set WITH_OLLAMA=1
    shift
) else if /I "%ARG:~0,13%"=="/ollama-host=" (
    set OLLAMA_HOST_URL=%ARG:~13%
    set WITH_OLLAMA=1
) else if /I "%ARG%"=="/ollama-host" (
    set "OLLAMA_HOST_URL=%~2"
    set WITH_OLLAMA=1
    shift
) else if /I "%ARG%"=="/h" (
    goto print_help
) else if /I "%ARG%"=="/help" (
    goto print_help
) else if /I "%ARG%"=="-h" (
    goto print_help
) else if /I "%ARG%"=="--help" (
    goto print_help
) else (
    echo Unknown flag: %ARG%
    echo Run with /help for usage.
    exit /b 2
)
shift
goto parse_args

:print_help
echo AutoGPT Windows Setup
echo.
echo Optional flags:
echo   /with-ollama           Install Ollama + pull default chat model + wire .env
echo   /ollama-model=NAME     Model tag to pull (default: %OLLAMA_MODEL%)
echo   /ollama-host=URL       Use existing Ollama at URL instead of installing
echo.
echo See docs/platform/copilot-local-llm.md for details.
exit /b 0

:args_done

echo =============================
echo   AutoGPT Windows Setup
echo =============================
echo.

REM --- Check prerequisites ---
echo Checking prerequisites...
where git >nul 2>nul
if errorlevel 1 (
    echo Git is not installed. Please install it and try again.
    pause
    exit /b 1
)
echo   Git is installed.

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker is not installed. Please install it and try again.
    pause
    exit /b 1
)
echo   Docker is installed.

if "%WITH_OLLAMA%"=="1" (
    REM curl ships with Windows 10/11 1803+ as curl.exe; we use it for
    REM the Ollama API probes and remote model-pull check, same as the
    REM Linux/macOS script. PowerShell's Invoke-WebRequest would also
    REM work but mixing tools makes the .bat harder to follow.
    where curl >nul 2>nul
    if errorlevel 1 (
        echo curl is not installed but /with-ollama needs it. Install curl ^(or update to Windows 10 1803+ which ships it^) and re-run.
        pause
        exit /b 1
    )
)
echo.

REM --- Detect repo ---
if exist "%REPO_DIR%\.git" (
    echo Using existing AutoGPT repository.
    set CLONE_NEEDED=0
) else (
    set REPO_DIR=%SCRIPT_DIR%AutoGPT
    set CLONE_NEEDED=1
)

REM --- Clone repo if needed ---
if "%CLONE_NEEDED%"=="1" (
    echo Cloning AutoGPT repository...
    git clone https://github.com/Significant-Gravitas/AutoGPT.git "%REPO_DIR%"
    if errorlevel 1 (
        echo Failed to clone repository.
        pause
        exit /b 1
    )
    echo Repository cloned successfully.
)
echo.

REM --- Bootstrap Ollama (optional) ---
if "%WITH_OLLAMA%"=="1" (
    call :bootstrap_ollama || exit /b 1
    call :write_local_env || exit /b 1
)

REM --- Navigate to autogpt_platform ---
cd /d "%REPO_DIR%\autogpt_platform"
if errorlevel 1 (
    echo Failed to navigate to autogpt_platform directory.
    pause
    exit /b 1
)

if not exist logs mkdir logs

REM --- Run docker compose ---
echo Starting AutoGPT services with Docker Compose...
echo This may take a few minutes on first run...
echo.
set LOG_FILE=%REPO_DIR%\autogpt_platform\logs\docker_setup.log
docker compose up -d > "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo Docker compose failed. Check log file for details: %LOG_FILE%
    echo.
    echo Common issues:
    echo - Docker is not running
    echo - Insufficient disk space
    echo - Port conflicts ^(check if ports 3000, 8000, etc. are in use^)
    pause
    exit /b 1
)

echo =============================
echo      Setup Complete!
echo =============================
echo.
echo Access AutoGPT at: http://localhost:3000
echo API available at: http://localhost:8000
if "%WITH_OLLAMA%"=="1" (
    echo.
    echo AutoPilot wired to Ollama ^(model: %OLLAMA_MODEL%^)
    echo Extended-thinking mode auto-downgrades to fast — Ollama doesn't speak
    echo Anthropic's wire protocol. See docs/platform/copilot-local-llm.md.
)
echo.
echo To stop services: docker compose down
echo To view logs: docker compose logs -f
echo.
echo Press any key to exit ^(services will keep running^)...
pause >nul
exit /b 0

REM ============================================================================
REM Subroutines
REM ============================================================================

:bootstrap_ollama
if not "%OLLAMA_HOST_URL%"=="" (
    REM Normalize remote URL: strip a trailing slash and an optional /v1
    REM so the operator can pass either the Ollama root
    REM ^(http://host:11434^) or a copy-pasted CHAT_BASE_URL ^(.../v1^).
    set OLLAMA_ROOT=%OLLAMA_HOST_URL%
    if "!OLLAMA_ROOT:~-1!"=="/" set OLLAMA_ROOT=!OLLAMA_ROOT:~0,-1!
    if /I "!OLLAMA_ROOT:~-3!"=="/v1" set OLLAMA_ROOT=!OLLAMA_ROOT:~0,-3!
    echo Using existing Ollama at !OLLAMA_ROOT!
    curl -sf "!OLLAMA_ROOT!/api/version" >nul
    if errorlevel 1 (
        echo Cannot reach Ollama at !OLLAMA_ROOT! — is it running and listening on 0.0.0.0?
        pause
        exit /b 1
    )
    REM Check whether the model is present on the remote; pull if not.
    curl -sf "!OLLAMA_ROOT!/api/tags" | findstr /C:"\"name\":\"%OLLAMA_MODEL%\"" >nul
    if errorlevel 1 (
        echo Model '%OLLAMA_MODEL%' missing on remote — requesting pull...
        REM /api/pull streams NDJSON; a registry 404 or network failure
        REM lands as an "error" object in the body rather than a non-2xx
        REM status, so we capture the stream to a file and grep for an
        REM explicit success / error frame.
        curl -sf -N "!OLLAMA_ROOT!/api/pull" -H "Content-Type: application/json" -d "{\"name\":\"%OLLAMA_MODEL%\"}" > "%TEMP%\ollama_pull.log"
        if errorlevel 1 (
            echo Pull request to !OLLAMA_ROOT!/api/pull failed
            pause
            exit /b 1
        )
        findstr /C:"\"error\"" "%TEMP%\ollama_pull.log" >nul
        if not errorlevel 1 (
            echo Pull of %OLLAMA_MODEL% failed. See %TEMP%\ollama_pull.log
            pause
            exit /b 1
        )
        findstr /C:"\"status\":\"success\"" "%TEMP%\ollama_pull.log" >nul
        if errorlevel 1 (
            echo Pull of %OLLAMA_MODEL% did not report success. See %TEMP%\ollama_pull.log
            pause
            exit /b 1
        )
        echo   Pulled %OLLAMA_MODEL% on remote.
    ) else (
        echo   Model %OLLAMA_MODEL% present on remote.
    )
    REM Stash the normalized root so write_local_env can reuse it.
    set OLLAMA_HOST_URL=!OLLAMA_ROOT!
    exit /b 0
)

REM Local Ollama install path. Prefer winget — it's preinstalled on
REM Windows 11 and most Windows 10 hosts that have App Installer.
REM If absent, point the operator at the official .exe rather than
REM piping a downloaded installer at them silently.
where ollama >nul 2>nul
if errorlevel 1 (
    where winget >nul 2>nul
    if errorlevel 1 (
        echo Ollama is not installed and winget is not available.
        echo Download and install Ollama from https://ollama.com/download/windows ^(default options^), then re-run this script.
        pause
        exit /b 1
    )
    echo Installing Ollama via winget...
    winget install --id Ollama.Ollama --silent --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo winget install Ollama.Ollama failed.
        echo Download Ollama manually from https://ollama.com/download/windows and re-run.
        pause
        exit /b 1
    )
) else (
    echo   Ollama already installed.
)

REM Set Ollama env vars for the current user so the desktop tray app
REM and any future `ollama serve` shell inherit them.
REM
REM - OLLAMA_HOST=0.0.0.0:11434 so containers can reach it via
REM   host.docker.internal ^(which Docker Desktop on Windows auto-injects
REM   in every container's /etc/hosts^).
REM - OLLAMA_CONTEXT_LENGTH=32768 because the OpenAI shim does NOT
REM   honor options.num_ctx in the request body ^(ollama/ollama#2714^);
REM   Ollama silently caps every request at the 4 k default otherwise,
REM   truncating AutoPilot's ~8 k system prompt.
REM
REM setx writes to HKCU but does NOT update the current cmd.exe session
REM — so we also export into this shell so the readiness probe below
REM and any post-script user shells see the new values immediately.
echo Setting OLLAMA_HOST and OLLAMA_CONTEXT_LENGTH ^(user env^)...
setx OLLAMA_HOST "0.0.0.0:11434" >nul
setx OLLAMA_CONTEXT_LENGTH "32768" >nul
set OLLAMA_HOST=0.0.0.0:11434
set OLLAMA_CONTEXT_LENGTH=32768

REM Restart Ollama so it picks up the new env. The Windows installer
REM registers Ollama as a background app that re-spawns on user login;
REM we kill it and start `ollama serve` so this run uses the new env
REM without waiting for a reboot.
taskkill /F /IM ollama.exe /T >nul 2>nul
taskkill /F /IM "ollama app.exe" /T >nul 2>nul
start "" /B ollama serve >nul 2>nul

REM Wait up to 30 s for /api/version. The serve process takes a
REM moment to bind on first launch, esp. on Windows Defender systems.
echo Waiting for Ollama to come up on http://localhost:11434 ...
set /a _try=0
:ollama_wait
set /a _try+=1
curl -sf http://localhost:11434/api/version >nul
if not errorlevel 1 goto ollama_ready
if %_try% GEQ 30 (
    echo Ollama did not become reachable on localhost:11434 within 30s.
    echo Open the Ollama tray app once to grant network permissions, then re-run.
    pause
    exit /b 1
)
timeout /t 1 /nobreak >nul
goto ollama_wait

:ollama_ready
echo Pulling model: %OLLAMA_MODEL% ^(this may take several minutes^)...
ollama pull "%OLLAMA_MODEL%"
if errorlevel 1 (
    echo Failed to pull %OLLAMA_MODEL%
    pause
    exit /b 1
)
echo   Ollama ready: http://localhost:11434
exit /b 0


:write_local_env
REM Write backend\.env wiring for the local transport. We use
REM host.docker.internal rather than 127.0.0.1 because Docker Desktop
REM on Windows auto-injects it in every container's /etc/hosts —
REM 127.0.0.1 inside a container points at the container, not the host.
REM
REM This block is idempotent: marker-bounded, so a re-run replaces our
REM lines and nothing else. We use a PowerShell one-liner instead of a
REM batch sed-loop because batch's line editing is genuinely painful
REM ^(no in-place edit, no regex address ranges^), and PowerShell ships
REM with every supported Windows.
cd /d "%REPO_DIR%\autogpt_platform\backend"
if errorlevel 1 (
    echo no backend dir
    exit /b 1
)
if not exist .env copy /Y .env.default .env >nul

set HOST_URL=
if not "%OLLAMA_HOST_URL%"=="" (
    set HOST_URL=%OLLAMA_HOST_URL%
) else (
    set HOST_URL=http://host.docker.internal:11434
)

set START_MARKER=# === Local-LLM AutoPilot wiring (added by setup-autogpt.bat /with-ollama) ===
set END_MARKER=# === End Local-LLM AutoPilot wiring ===

REM Strip any previous block we wrote so re-runs don't accumulate.
REM ``-replace`` with ``(?s)`` makes the regex span newlines; the
REM markers contain regex metacharacters ^(``.``, ``(``, ``)``^), so we
REM ``[Regex]::Escape`` them before splicing into the pattern.
powershell -NoProfile -Command "$start = [Regex]::Escape($env:START_MARKER); $end = [Regex]::Escape($env:END_MARKER); $p = (Resolve-Path .env).Path; $content = Get-Content -Raw $p; if ($null -eq $content) { $content = '' }; $content = $content -replace ('(?s)' + $start + '.*?' + $end + '\r?\n?'), ''; [IO.File]::WriteAllText($p, $content)"

REM Append a fresh block via per-line >> redirects — NOT a parenthesised
REM ( ... ) >> .env group. START_MARKER ends in ")" (...with-ollama) ===),
REM and an unescaped ")" inside a () block closes the group early, so the
REM marker and every line after it get echoed to the console instead of
REM written to .env (CHAT_USE_LOCAL never lands -> AutoPilot 401s). No
REM space before ">>" so model slugs don't gain a trailing space.
REM Ollama-side env knobs OLLAMA_HOST and OLLAMA_CONTEXT_LENGTH are set on
REM the host via setx above, NOT in backend\.env ^(.env is read by
REM containers, where those vars belong to the Ollama process itself^).
echo.>>.env
echo %START_MARKER%>>.env
echo # See docs/platform/copilot-local-llm.md for the full reference.>>.env
echo CHAT_USE_LOCAL=true>>.env
echo CHAT_BASE_URL=%HOST_URL%/v1>>.env
echo CHAT_API_KEY=ollama>>.env
echo CHAT_FAST_STANDARD_MODEL=%OLLAMA_MODEL%>>.env
echo CHAT_FAST_ADVANCED_MODEL=%OLLAMA_MODEL%>>.env
echo OLLAMA_HOST=%HOST_URL%>>.env
echo %END_MARKER%>>.env

echo   wrote backend\.env ^(CHAT_USE_LOCAL=true, Ollama at %HOST_URL%^)
cd /d "%REPO_DIR%"
exit /b 0
