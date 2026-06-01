# ============================================================================
# AutoGPT Platform - Ultimate Installer (Windows)
# ----------------------------------------------------------------------------
# Zero-prerequisite bootstrap: checks whether your machine CAN run AutoGPT,
# installs git + Docker Desktop if missing, fetches the repo at the version
# you pick, then hands off to setup-autogpt.bat to bring the stack up.
#
# One-liner:
#   powershell -ExecutionPolicy Bypass -Command "iwr https://setup.agpt.co/install.ps1 -OutFile install.ps1; ./install.ps1"
#
# Flags (parity with install.sh):
#   -Dev                 install the dev branch (default: latest release)
#   -Branch <name>       install a specific branch
#   -Release <tag>       install a specific release tag
#   -Dir <path>          install location (default: %USERPROFILE%\AutoGPT)
#   -WithOllama          also install Ollama + wire local-LLM AutoPilot (no cloud keys)
#   -OllamaModel <name>  model to pull (implies -WithOllama)
#   -OllamaHost <url>    use an existing Ollama at this URL (implies -WithOllama)
#   -Yes                 non-interactive; assume "yes" to prompts
#   -SkipPreflight       skip the capability checks (not recommended)
#   -Help
# ============================================================================
[CmdletBinding()]
param(
  [switch]$Dev,
  [string]$Branch,
  [string]$Release,
  [string]$Dir = "$env:USERPROFILE\AutoGPT",
  [switch]$WithOllama,
  [string]$OllamaModel,
  [string]$OllamaHost,
  [switch]$Yes,
  [switch]$SkipPreflight,
  [switch]$PreflightOnly,
  [switch]$Help
)

$ErrorActionPreference = 'Stop'
$REPO_URL = 'https://github.com/Significant-Gravitas/AutoGPT.git'
$REPO_API = 'https://api.github.com/repos/Significant-Gravitas/AutoGPT'
$MIN_RAM_GB  = 8
$MIN_DISK_GB = 25

# ---- tiny UI helpers (parity vocabulary with install.sh) ----
function Say   ($m) { Write-Host $m }
function Info  ($m) { Write-Host "  $m" -ForegroundColor Gray }
function Ok    ($m) { Write-Host "  [ OK ] $m" -ForegroundColor Green }
function Warn  ($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Fail  ($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red }
function Step  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Die   ($m) { Write-Host "`nError: $m" -ForegroundColor Red; exit 1 }

if ($Help) {
  Get-Content $PSCommandPath | Select-String -Pattern '^#' | ForEach-Object { $_.Line -replace '^# ?','' } | Select-Object -First 28
  exit 0
}

Say ""
Say "============================================="
Say "      AutoGPT Platform - Ultimate Installer"
Say "============================================="

# ---- resolve which version to install ----
function Resolve-Version {
  if ($Branch)  { return @{ kind='branch';  ref=$Branch } }
  if ($Release) { return @{ kind='tag';     ref=$Release } }
  if ($Dev)     { return @{ kind='branch';  ref='dev' } }
  # default: latest release tag
  try {
    $tag = (Invoke-RestMethod "$REPO_API/releases/latest" -UseBasicParsing).tag_name
    if ($tag) { return @{ kind='tag'; ref=$tag } }
  } catch { Warn "Couldn't resolve latest release ($($_.Exception.Message)); falling back to dev." }
  return @{ kind='branch'; ref='dev' }
}

# ---- privilege check ----
function Test-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  return ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ============================================================================
# PRE-FLIGHT: "will AutoGPT actually run on this machine?"  Reports each check,
# and HARD-STOPS (with a fix) on anything that means Docker can never work.
# ============================================================================
function Invoke-Preflight {
  Step "Pre-flight checks (can this machine run AutoGPT?)"
  $hardFail = $false

  # OS + edition + build
  $os = Get-CimInstance Win32_OperatingSystem
  $cs = Get-CimInstance Win32_ComputerSystem
  $build = [int]($os.BuildNumber)
  Info "$($os.Caption) (build $build), $([math]::Round($cs.TotalPhysicalMemory/1GB,1)) GB RAM"
  if ($build -lt 19041) {
    Fail "Windows 10 2004 (build 19041) or newer is required for WSL2/Docker. You have build $build."
    Info "Fix: update Windows (Settings > Windows Update) to 21H2+ or use Windows 11."
    $hardFail = $true
  } else { Ok "Windows version supports WSL2/Docker" }

  # CPU virtualization - the #1 'it will never work' blocker
  $hyper = $cs.HypervisorPresent
  $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
  $firmwareVirt = $cpu.VirtualizationFirmwareEnabled
  if ($hyper) {
    Ok "CPU virtualization active (a hypervisor is already running)"
  } elseif ($firmwareVirt) {
    Ok "CPU virtualization enabled in firmware (WSL2 can use it)"
  } else {
    Fail "CPU virtualization (VT-x / AMD-V / SVM) is DISABLED. Docker/WSL2 cannot run without it."
    Info "Fix: reboot into BIOS/UEFI and enable 'Intel VT-x' / 'AMD-V (SVM)' / 'Virtualization'."
    Info "If your CPU genuinely lacks it (very old hardware), AutoGPT can't run here."
    $hardFail = $true
  }

  # RAM
  $ramGB = [math]::Round($cs.TotalPhysicalMemory/1GB,1)
  if ($ramGB -lt $MIN_RAM_GB) { Warn "Only $ramGB GB RAM; the stack wants >= $MIN_RAM_GB GB. It may be slow / OOM." }
  else { Ok "$ramGB GB RAM (>= $MIN_RAM_GB GB)" }

  # Disk
  $sys = (Get-Item $env:SystemDrive).Root.Name.TrimEnd('\')
  $freeGB = [math]::Round((Get-PSDrive ($sys.TrimEnd(':'))).Free/1GB,1)
  if ($freeGB -lt $MIN_DISK_GB) {
    Fail "Only $freeGB GB free on $sys; AutoGPT images + stack need ~$MIN_DISK_GB GB."
    Info "Fix: free up space (the backend image + base images are ~15-20 GB) and re-run."
    $hardFail = $true
  } else { Ok "$freeGB GB free on $sys (>= $MIN_DISK_GB GB)" }

  # Admin (needed to install Docker Desktop / enable WSL2)
  if (Test-Admin) { Ok "Running with administrator rights" }
  else { Warn "Not elevated. Installing Docker Desktop / enabling WSL2 needs admin - you'll get UAC prompts (or re-run as administrator)." }

  if ($hardFail) {
    Die "This machine can't run AutoGPT yet - resolve the [FAIL] item(s) above and re-run. Nothing was installed."
  }
  Ok "Pre-flight passed - this machine can run AutoGPT."
}

# ============================================================================
# PREREQS: git + Docker Desktop via DIRECT installers (no winget dependency).
# ============================================================================
function Install-Git {
  if (Get-Command git -ErrorAction SilentlyContinue) { Ok "git already installed"; return }
  Step "Installing Git for Windows"
  $asset = (Invoke-RestMethod "https://api.github.com/repos/git-for-windows/git/releases/latest" -UseBasicParsing).assets |
    Where-Object { $_.name -match '-64-bit\.exe$' -and $_.name -notmatch 'Portable' } | Select-Object -First 1
  $exe = "$env:TEMP\git-install.exe"
  Info "Downloading $($asset.name)..."
  Invoke-WebRequest -UseBasicParsing $asset.browser_download_url -OutFile $exe
  Info "Installing silently..."
  Start-Process $exe -ArgumentList '/VERYSILENT','/NORESTART','/NOCANCEL','/SP-','/SUPPRESSMSGBOXES' -Wait
  $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
  if (Get-Command git -ErrorAction SilentlyContinue) { Ok "git installed" } else { Die "git install failed - install it from https://git-scm.com and re-run." }
}

function Test-DockerReady {
  $docker = Get-Command docker -ErrorAction SilentlyContinue
  if (-not $docker) { return $false }
  try { & docker info *>$null; return ($LASTEXITCODE -eq 0) } catch { return $false }
}

function Install-Docker {
  if (Test-DockerReady) { Ok "Docker is installed and running"; return }
  if (Get-Command docker -ErrorAction SilentlyContinue) {
    Step "Docker is installed but the engine isn't responding - starting Docker Desktop"
    $dd = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd }
    Wait-Docker
    return
  }
  Step "Installing Docker Desktop"
  Info "Ensuring WSL2 is available (needs admin)..."
  try { wsl.exe --install --no-distribution 2>&1 | Out-Null } catch {}
  try { wsl.exe --update --web-download 2>&1 | Out-Null } catch {}
  $exe = "$env:TEMP\DockerDesktopInstaller.exe"
  Info "Downloading Docker Desktop (~700 MB)..."
  Invoke-WebRequest -UseBasicParsing 'https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe' -OutFile $exe
  Info "Installing silently (WSL2 backend, license accepted)..."
  Start-Process $exe -ArgumentList 'install','--quiet','--accept-license','--backend=wsl-2' -Wait
  try { Add-LocalGroupMember -Group 'docker-users' -Member $env:USERNAME -ErrorAction SilentlyContinue } catch {}
  Warn "Docker Desktop installed. Windows usually needs a REBOOT before the engine starts."
  $dd = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
  if (Test-Path $dd) { Start-Process $dd }
  Wait-Docker
}

function Wait-Docker {
  Step "Waiting for the Docker engine (first start can take a few minutes)"
  for ($i=0; $i -lt 60; $i++) {
    if (Test-DockerReady) { Ok "Docker engine is up"; return }
    Start-Sleep 10
    if ($i -eq 11) { Warn "Still waiting - if Docker Desktop is asking you to reboot/sign in, do that, then re-run this installer (it'll resume)." }
  }
  Die "Docker engine didn't come up. Reboot, start Docker Desktop once, then re-run this installer."
}

# ============================================================================
# FETCH + HAND OFF
# ============================================================================
function Get-Repo($ver) {
  Step "Fetching AutoGPT ($($ver.kind): $($ver.ref)) into $Dir"
  if (Test-Path (Join-Path $Dir '.git')) {
    Info "Repo already present - updating..."
    git -C $Dir fetch --depth 1 origin $ver.ref 2>&1 | Out-Null
    git -C $Dir checkout FETCH_HEAD 2>&1 | Out-Null
  } else {
    New-Item -ItemType Directory -Force -Path (Split-Path $Dir) | Out-Null
    git clone --depth 1 --branch $ver.ref $REPO_URL $Dir 2>&1 | Out-Null
  }
  if (-not (Test-Path (Join-Path $Dir 'autogpt_platform\installer\setup-autogpt.bat'))) {
    Die "Clone/checkout failed - $Dir doesn't contain the installer."
  }
  Ok "Repo ready at $Dir"
}

function Invoke-Setup {
  Step "Handing off to setup-autogpt.bat (builds + starts the stack)"
  $bat = Join-Path $Dir 'autogpt_platform\installer\setup-autogpt.bat'
  $setupArgs = @()
  if ($WithOllama -or $OllamaModel -or $OllamaHost) { $setupArgs += '/with-ollama' }
  if ($OllamaModel) { $setupArgs += "/ollama-model=$OllamaModel" }
  if ($OllamaHost)  { $setupArgs += "/ollama-host=$OllamaHost" }
  Info "Running: setup-autogpt.bat $($setupArgs -join ' ')"
  & cmd /c "`"$bat`" $($setupArgs -join ' ')"
}

# ============================================================================
# MAIN
# ============================================================================
$ver = Resolve-Version
Info "Selected version -> $($ver.kind): $($ver.ref)"
if (-not $SkipPreflight) { Invoke-Preflight } else { Warn "Pre-flight skipped (-SkipPreflight)." }
if ($PreflightOnly) { Say "`n(-PreflightOnly: stopping before any install.)"; exit 0 }
Install-Git
Install-Docker
Get-Repo $ver
Invoke-Setup

Say ""
Say "============================================="
Say "  Done. AutoGPT should be at http://localhost:3000"
Say "============================================="
