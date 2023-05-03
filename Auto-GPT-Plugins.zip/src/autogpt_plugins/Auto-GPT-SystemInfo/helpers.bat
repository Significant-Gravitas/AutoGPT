@echo off

if "%1" == "clean" (
  echo Removing build artifacts and temporary files...
  call :clean
) else if "%1" == "qa" (
  echo Running static analysis tools...
  call :qa
) else if "%1" == "style" (
  echo Running code formatters...
  call :style
) else (
  echo Usage: %0 [clean^|qa^|style]
  exit /b 1
)

exit /b 0

:clean
  rem Remove build artifacts and temporary files
  @del /s /q build 2>nul
  @del /s /q dist 2>nul
  @del /s /q __pycache__ 2>nul
  @del /s /q *.egg-info 2>nul
  @del /s /q **\*.egg-info 2>nul
  @del /s /q *.pyc 2>nul
  @del /s /q **\*.pyc 2>nul
  @del /s /q reports 2>nul
  echo Done!
  exit /b 0

:qa
  rem Run static analysis tools
  @flake8 .
  @python run_pylint.py
  echo Done!
  exit /b 0

:style
  rem Format code
  @isort .
  @black --exclude=".*\/*(dist|venv|.venv|test-results)\/*.*" .
  echo Done!
  exit /b 0
