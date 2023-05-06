@echo off
setlocal EnableDelayedExpansion EnableExtensions

rem Notes:
rem 
rem Windows batch does not support ANSI colors (without using external tools).  
rem If you run this script in Windows batch mode, you'll not get colored output.
rem
rem This batch file is provided for Windows Server compatibility and for those who do not have/want
rem to use PowerShell.
rem
rem We recommend that the standard modern Windows end-user uses the PowerShell variant, as long
rem as you feel comfortable doing so.

rem Entry-point
call :main
goto :eof

rem Usage instructions
:usage
echo Usage: %0 [-h] [-v] [-e VENV_PATH] [-c] [-s] [-C AI_SETTINGS] [-l CONTINUOUS_LIMIT] [-S] [-d] [-g] [-G] [-m MEMORY_TYPE] [-b BROWSER_NAME] [-A] [-N] [-w WORKSPACE_DIR] [-I] [args]
echo(
echo Command-line Options:
echo(
echo   -h, --help                                Show this help message and exit.
echo   -v, --virtualenv                          Use Python virtual environment. (default: false)
echo   -e, --venv-path VENV_PATH                 Set the path to the Python virtual environment.
echo   -c, --continuous                          Enable continuous mode for autogpt. (default: false)
echo   -s, --skip-reprompt                       Skip the re-prompting messages at the beginning of the script.
echo   -C, --ai-settings AI_SETTINGS             Specify which ai_settings.yaml file to use.
echo   -l, --continuous-limit CONTINUOUS_LIMIT   Define the number of times to run in continuous mode.
echo   -S, --speak                               Enable Speak Mode.
echo   -d, --debug                               Enable Debug Mode.
echo   -g, --gpt3only                            Enable GPT3.5 Only Mode.
echo   -G, --gpt4only                            Enable GPT4 Only Mode.
echo   -m, --use-memory MEMORY_TYPE              Define which Memory backend to use.
echo   -b, --browser-name BROWSER_NAME           Specify which web-browser to use when using selenium to scrape the web.
echo   -A, --allow-downloads                     Dangerous: Allows Auto-GPT to download files natively.
echo   -N, --skip-news                           Specifies whether to suppress the output of latest news on startup.
echo   -w, --workspace-directory WORKSPACE_DIR   Specifies which directory to use for the workspace.
echo   -I, --install-plugin-deps                 Installs external dependencies for 3rd party plugins.
echo(
echo Environment Variables:
echo   You can set these values instead of passing command line arguments,
echo   before executing this script.
echo(
echo   USE_VENV                                 Set to true to use Python virtual environment. (default: false)
echo   VENV_PATH                                Set the path to the Python virtual environment.
echo   CONTINUOUS                               Set to true to enable continuous mode for autogpt. (default: false)
echo   SKIP_REPROMPT                            Set to true to skip the re-prompting messages at the beginning of the script.
echo   AI_SETTINGS                              Specify which ai_settings.yaml file to use.
echo   CONTINUOUS_LIMIT                         Define the number of times to run in continuous mode.
echo   SPEAK                                    Set to true to enable Speak Mode.
echo   DEBUG                                    Set to true to enable Debug Mode.
echo   GPT3_ONLY                                Set to true to use GPT-3 API only for autogpt. (default: false)
echo   GPT4_ONLY                                Set to true to use GPT-4 API only for autogpt. (default: false)
echo   MEMORY_TYPE                              Define which Memory backend to use.
echo   BROWSER_NAME                             Specify which web-browser to use when using selenium to scrape the web.
echo   ALLOW_DOWNLOADS                          Set to true to allow Auto-GPT to download files natively. (default: false)
echo   SKIP_NEWS                                Set to true to suppress the output of latest news on startup. (default: false)
echo   WORKSPACE_DIRECTORY                      Specifies which directory to use for the workspace.
echo   INSTALL_PLUGIN_DEPS                      Set to true to install external dependencies for 3rd party plugins. (default: false)
exit /b

rem Argument parsing
:parse_args
if "%~1"=="" exit /b 1
if /I "%~1"=="-h" call :usage && exit /b 1
if /I "%~1"=="--help" call :usage && exit /b 1
if /I "%~1"=="-v" set "USE_VENV=true" && shift && goto parse_args
if /I "%~1"=="--virtualenv" set "USE_VENV=true" && shift && goto parse_args
if /I "%~1"=="-e" set "VENV_PATH=%~2" && shift && shift && goto parse_args
if /I "%~1"=="--venv-path" set "VENV_PATH=%~2" && shift && shift && goto parse_args
if /I "%~1"=="-c" set "CONTINUOUS=true" && shift && goto parse_args
if /I "%~1"=="--continuous" set "CONTINUOUS=true" && shift && goto parse_args
if /I "%~1"=="-s" set "SKIP_REPROMPT=true" && shift && goto parse_args
if /I "%~1"=="--skip-reprompt" set "SKIP_REPROMPT=true" && shift && goto parse_args
if /I "%~1"=="-C" set "AI_SETTINGS=%~2" && shift && shift && goto parse_args
if /I "%~1"=="--ai-settings" set "AI_SETTINGS=%~2" && shift && shift && goto parse_args
if /I "%~1"=="-l" set "CONTINUOUS_LIMIT=%~2" && shift && shift && goto parse_args
if /I "%~1"=="--continuous-limit" set "CONTINUOUS_LIMIT=%~2" && shift && shift && goto parse_args
if /I "%~1"=="-S" set "SPEAK=true" && shift && goto parse_args
if /I "%~1"=="--speak" set "SPEAK=true" && shift && goto parse_args
if /I "%~1"=="-d" set "DEBUG=true" && shift && goto parse_args
if /I "%~1"=="--debug" set "DEBUG=true" && shift && goto parse_args
if /I "%~1"=="-g" set "GPT3_ONLY=true" && shift && goto parse_args
if /I "%~1"=="--gpt3only" set "GPT3_ONLY=true" && shift && goto parse_args
if /I "%~1"=="-G" set "GPT4_ONLY=true" && shift && goto parse_args
if /I "%~1"=="--gpt4only" set "GPT4_ONLY=true" && shift && goto parse_args
if /I "%~1"=="-m" set "MEMORY_TYPE=%~2" && shift && shift && goto parse_args
if /I "%~1"=="--use-memory" set "MEMORY_TYPE=%~2" && shift && shift && goto parse_args
if /I "%~1"=="-b" set "BROWSER_NAME=%~2" && shift && shift && goto parse_args
if /I "%~1"=="--browser-name" set "BROWSER_NAME=%~2" && shift && shift && goto parse_args
if /I "%~1"=="-A" set "ALLOW_DOWNLOADS=true" && shift && goto parse_args
if /I "%~1"=="--allow-downloads" set "ALLOW_DOWNLOADS=true" && shift && goto parse_args
if /I "%~1"=="-N" set "SKIP_NEWS=true" && shift && goto parse_args
if /I "%~1"=="--skip-news" set "SKIP_NEWS=true" && shift && goto parse_args
if /I "%~1"=="-w" set "WORKSPACE_DIR=%~2" && shift && shift && goto parse_args
if /I "%~1"=="--workspace-directory" set "WORKSPACE_DIR=%~2" && shift && shift && goto parse_args
if /I "%~1"=="-I" set "INSTALL_PLUGIN_DEPS=true" && shift && goto parse_args
if /I "%~1"=="--install-plugin-deps" set "INSTALL_PLUGIN_DEPS=true" && shift && goto parse_args
shift && goto parse_args
exit /b

rem Command builder
:build_cmd
set "COMMAND=!PYTHON_CMD! -m autogpt"
if "!CONTINUOUS!"=="true" set "COMMAND=!COMMAND! --continuous"
if "!GPT3_ONLY!"=="true" set "COMMAND=!COMMAND! --gpt3only"
if "!GPT4_ONLY!"=="true" set "COMMAND=!COMMAND! --gpt4only"
if "!SKIP_REPROMPT!"=="true" set "COMMAND=!COMMAND! --skip-reprompt"
if not "!AI_SETTINGS!"=="" set "COMMAND=!COMMAND! --ai-settings !AI_SETTINGS!"
if not "!CONTINUOUS_LIMIT!"=="" set "COMMAND=!COMMAND! --continuous-limit !CONTINUOUS_LIMIT!"
if "!SPEAK!"=="true" set "COMMAND=!COMMAND! --speak"
if "!DEBUG!"=="true" set "COMMAND=!COMMAND! --debug"
if not "!MEMORY_TYPE!"=="" set "COMMAND=!COMMAND! --use-memory !MEMORY_TYPE!"
if not "!BROWSER_NAME!"=="" set "COMMAND=!COMMAND! --browser-name !BROWSER_NAME!"
if "!ALLOW_DOWNLOADS!"=="true" set "COMMAND=!COMMAND! --allow-downloads"
if "!SKIP_NEWS!"=="true" set "COMMAND=!COMMAND! --skip-news"
if not "!WORKSPACE_DIR!"=="" set "COMMAND=!COMMAND! --workspace-directory !WORKSPACE_DIR!"
if "!INSTALL_PLUGIN_DEPS!"=="true" set "COMMAND=!COMMAND! --install-plugin-deps"
set "COMMAND=!COMMAND! %*"
exit /b

rem Default setter
:set_defaults
if not defined USE_VENV set USE_VENV=false
if not defined VENV_PATH set VENV_PATH=./venv
if not defined CONTINUOUS set CONTINUOUS=false
if not defined SKIP_REPROMPT set SKIP_REPROMPT=false
if not defined AI_SETTINGS set AI_SETTINGS=./ai_settings.yaml
if not defined CONTINUOUS_LIMIT set CONTINUOUS_LIMIT=50
if not defined SPEAK set SPEAK=false
if not defined DEBUG set DEBUG=false
if not defined GPT3_ONLY set GPT3_ONLY=false
if not defined GPT4_ONLY set GPT4_ONLY=false
if not defined MEMORY_TYPE set MEMORY_TYPE=redis
if not defined BROWSER_NAME set BROWSER_NAME=chrome
if not defined ALLOW_DOWNLOADS set ALLOW_DOWNLOADS=false
if not defined SKIP_NEWS set SKIP_NEWS=false
if not defined WORKSPACE_DIR set WORKSPACE_DIR=./autogpt/auto_gpt_workspace
if not defined INSTALL_PLUGIN_DEPS set INSTALL_PLUGIN_DEPS=false
exit /b

:main
call :set_defaults
call :parse_args %*

rem Use virtual environment if requested
if %USE_VENV%==true (
    echo Using Python virtual environment: %VENV_PATH%
    call %VENV_PATH%\Scripts\activate.bat
)

rem Check and install missing requirements
where python3 >nul 2>nul || where python >nul 2>nul || (
    echo Python not found. Please install Python.
    exit /b 1
)
python -m scripts.check_requirements requirements.txt
if %errorlevel%==1 (
    echo Installing missing packages...
    python -m pip install -r requirements.txt
)

rem Run the command
set COMMAND=
call :build_cmd %* COMMAND
call %COMMAND%

rem Deactivate virtual environment if used
if %USE_VENV%==true (
    echo Deactivating Python virtual environment: %VENV_PATH%
    call %VENV_PATH%\Scripts\deactivate.bat
)

rem Press any key to continue...
pause
exit /b

rem End-of-file
:eof
exit /b 0