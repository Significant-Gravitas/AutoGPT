@echo off
setlocal
set "AUTOGPT_INSTALLER=%TEMP%\autogpt-install-%RANDOM%%RANDOM%%RANDOM%.ps1"
curl.exe --proto =https --proto-redir =https --tlsv1.2 -fsSL -o "%AUTOGPT_INSTALLER%" https://setup.agpt.co/install.ps1
if errorlevel 1 (
	del /f /q "%AUTOGPT_INSTALLER%" >nul 2>&1
	exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%AUTOGPT_INSTALLER%" %*
set "AUTOGPT_INSTALLER_EXIT=%ERRORLEVEL%"
del /f /q "%AUTOGPT_INSTALLER%" >nul 2>&1
exit /b %AUTOGPT_INSTALLER_EXIT%
