@echo off
rem AutoGPT Platform release installer for Windows, for the older command
rem that downloads https://setup.agpt.co/install.bat and runs it. Like the
rem PowerShell command in docs/platform/installer.md, it downloads install.ps1
rem to a unique temporary file and runs it only after a successful fetch.
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
