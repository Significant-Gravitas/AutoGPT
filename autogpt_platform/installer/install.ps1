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
  # Print the contiguous leading comment header (stop at the first non-# line),
  # stripping the leading "# ".
  foreach ($line in Get-Content $PSCommandPath) {
    if ($line -notmatch '^#') { break }
    $line -replace '^# ?',''
  }
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
  } catch { }
  Die "Couldn't determine the latest AutoGPT release (GitHub API unreachable?). Re-run when you're back online, or pass -Dev for the development branch (or -Release <tag> for a specific version)."
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
    # VirtualizationFirmwareEnabled is unreliable: it reflects one specific
    # firmware flag and reports $false/$null on plenty of machines where
    # VT-x/AMD-V is actually on and WSL2/Docker run fine. Don't hard-block on
    # it — warn and let Docker's own engine start be the real test (Wait-Docker
    # surfaces a clear failure + BIOS fix if virtualization really is off).
    Warn "Couldn't confirm CPU virtualization from firmware flags (this probe is unreliable, so this isn't a hard stop)."
    Info "If Docker fails to start later, reboot into BIOS/UEFI and enable 'Intel VT-x' / 'AMD-V (SVM)' / 'Virtualization'."
  }

  # RAM
  $ramGB = [math]::Round($cs.TotalPhysicalMemory/1GB,1)
  if ($ramGB -lt $MIN_RAM_GB) { Warn "Only $ramGB GB RAM; the stack wants >= $MIN_RAM_GB GB. It may be slow / OOM." }
  else { Ok "$ramGB GB RAM (>= $MIN_RAM_GB GB)" }

  # Disk — check the drive the install actually lands on ($Dir, walking up to
  # its nearest existing parent), not just the system drive: with -Dir on
  # another volume the system drive could pass while the target is full.
  $diskTarget = $Dir
  while ($diskTarget -and -not (Test-Path $diskTarget)) { $diskTarget = Split-Path $diskTarget -Parent }
  if (-not $diskTarget) { $diskTarget = $env:SystemDrive }
  try { $drive = (Get-Item $diskTarget).PSDrive } catch { $drive = $null }
  if (-not $drive) { $drive = Get-PSDrive ($env:SystemDrive.TrimEnd(':')) }
  $freeGB = [math]::Round($drive.Free/1GB,1)
  if ($freeGB -lt $MIN_DISK_GB) {
    Fail "Only $freeGB GB free on $($drive.Name): (install target $Dir); AutoGPT images + stack need ~$MIN_DISK_GB GB."
    Info "Fix: free up space (the backend image + base images are ~15-20 GB) and re-run."
    $hardFail = $true
  } else { Ok "$freeGB GB free on $($drive.Name): (>= $MIN_DISK_GB GB)" }

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
  if (-not $asset) { Die "Couldn't find a Git for Windows installer in the latest release. Install it from https://git-scm.com and re-run." }
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
  # Docker Desktop ships separate x64 and ARM64 builds; the amd64 installer
  # won't run on ARM64 Windows (Snapdragon etc.). PROCESSOR_ARCHITEW6432 covers
  # the case where this is a 32-bit PowerShell on a 64-bit/ARM64 OS.
  $dockerArch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') { 'arm64' } else { 'amd64' }
  Info "Downloading Docker Desktop ($dockerArch, ~700 MB)..."
  Invoke-WebRequest -UseBasicParsing "https://desktop.docker.com/win/main/$dockerArch/Docker%20Desktop%20Installer.exe" -OutFile $exe
  Info "Installing silently (WSL2 backend, license accepted)..."
  $proc = Start-Process $exe -ArgumentList 'install','--quiet','--accept-license','--backend=wsl-2' -Wait -PassThru
  # 0 = success, 3010 = success but reboot required. Anything else failed — bail
  # now rather than polling Wait-Docker for ten minutes on a broken install.
  if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
    Die "Docker Desktop installer exited with code $($proc.ExitCode). Install it manually from https://www.docker.com/products/docker-desktop and re-run."
  }
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
    if ($LASTEXITCODE -ne 0) { Die "git fetch failed for '$($ver.ref)'. Check the branch/tag name and your network, then re-run." }
    git -C $Dir checkout FETCH_HEAD 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Die "git checkout failed in $Dir. Resolve the repo state (or remove $Dir) and re-run." }
  } elseif ((Test-Path $Dir) -and (Get-ChildItem -Force $Dir -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    # Non-empty but not a git checkout — usually a half-finished clone from an
    # interrupted run; a plain clone into it would fail every rerun.
    Die "$Dir exists but is not a git checkout (leftover from an interrupted run?). Remove it and re-run:  Remove-Item -Recurse -Force `"$Dir`""
  } else {
    # Split-Path is empty for a bare relative name like "AutoGPT" — only create
    # the parent when there is one.
    $parent = Split-Path $Dir -Parent
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    git clone --depth 1 --branch $ver.ref $REPO_URL $Dir 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Die "git clone failed (branch/tag '$($ver.ref)'). Check the name and your network, then re-run." }
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
  # Invoke the .bat directly with array args (NOT a composed `cmd /c "<string>"`):
  # PowerShell passes each element as one properly-quoted argument, so a value
  # like an Ollama URL containing '&' isn't split into separate cmd commands.
  & $bat @setupArgs
  if ($LASTEXITCODE -ne 0) {
    Die "setup-autogpt.bat failed (exit code $LASTEXITCODE). Scroll up for the error, fix it, then re-run."
  }
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
