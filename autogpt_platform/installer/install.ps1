# AutoGPT Platform release installer for Windows.
#
# Docker Desktop must already be installed, running, and using Linux containers.
# Download to a unique temporary file and execute only after a successful fetch:
#   & { $installer = Join-Path ([IO.Path]::GetTempPath()) "autogpt-install-$([guid]::NewGuid().ToString('N')).ps1"
#       try { curl.exe --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL -o $installer https://setup.agpt.co/install.ps1
#             if ($LASTEXITCODE -eq 0) { powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer } }
#       finally { Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue } }
#
# Runs on Windows PowerShell 5.1 and PowerShell 7. Keep this file ASCII:
# Windows PowerShell reads a script without a byte-order mark as ANSI.
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ScriptArgs = @($args)

$DeployImage = 'docker.io/significantgravitas/autogpt'
$ContainerName = 'autogpt'
$DataVolume = 'autogpt-platform-data'
$OwnerLabelKey = 'com.agpt.installer.owner'
$OwnerLabelValue = 'autogpt-platform-installer'
$ContractLabelKey = 'com.agpt.installer.contract'
$ContractLabelValue = '1'
$InstallIdLabelKey = 'com.agpt.installer.install-id'
$EnvHashLabelKey = 'com.agpt.installer.env-sha256'
$ExpectedTitle = 'AutoGPT Platform single-container'
$ExpectedSource = 'https://github.com/Significant-Gravitas/AutoGPT'
$MinRamGb = 8
$MinDiskGb = 25

$UserSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$SystemSid = 'S-1-5-18'
$AdministratorsSid = 'S-1-5-32-544'
$TrustedInstallerSid = 'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464'
$OwnerRightsSid = 'S-1-3-4'
# SYSTEM and Administrators play root's part: they can always reach the state.
$PrivateSids = @($UserSid, $SystemSid, $AdministratorsSid)
$TrustedSids = @($UserSid, $SystemSid, $AdministratorsSid, $TrustedInstallerSid, $OwnerRightsSid)
$OwnerSids = @($UserSid)
# What an elevated administrator creates can belong to the Administrators group
# instead (the default on Windows Server).
if (([System.Security.Principal.WindowsPrincipal][System.Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
	$OwnerSids += $AdministratorsSid
}
# Rights that let another principal delete, rename, re-permission, or take over
# an entry on the path. Rights that only create new entries are like a sticky
# directory's: they cannot replace what is already there.
$TamperRights = 0x40 -bor 0x10000 -bor 0x40000 -bor 0x80000 -bor 0x10000000

$CheckDirOnly = $false
$Release = ''
$SkipPreflight = $false
$PreflightOnly = $false
$ResolveOnly = $false
$VerRef = ''
$Image = ''
$ImageDigest = ''
$DaemonArch = ''
$DockerExe = ''
$InstallId = ''
$EnvFile = ''
$EnvHash = ''
$VolumeEstablished = $false

function Say([string]$Text) { Write-Host $Text }
function Info([string]$Text) { Write-Host "  $Text" }
function Ok([string]$Text) {
	Write-Host '  ' -NoNewline
	Write-Host '[ OK ]' -ForegroundColor Green -NoNewline
	Write-Host " $Text"
}
function Warn([string]$Text) {
	Write-Host '  ' -NoNewline
	Write-Host '[WARN]' -ForegroundColor Yellow -NoNewline
	Write-Host " $Text"
}
function Step([string]$Text) {
	Write-Host ''
	Write-Host "==> $Text" -ForegroundColor Cyan
}
function Die([string]$Text) {
	[Console]::Error.WriteLine('')
	$color = -not [Console]::IsErrorRedirected
	if ($color) {
		$previous = [Console]::ForegroundColor
		[Console]::ForegroundColor = [ConsoleColor]::Red
	}
	[Console]::Error.WriteLine("Error: $Text")
	if ($color) { [Console]::ForegroundColor = $previous }
	exit 1
}

trap {
	[Console]::Error.WriteLine('')
	[Console]::Error.WriteLine("Error: $($_.Exception.Message)")
	exit 1
}

function Show-Help {
	Write-Host @'
AutoGPT Platform Installer (Windows)

Installs the published single-container appliance. Docker Desktop must be
running locally and configured for Linux containers.

Safe download and execution:
  & { $installer = Join-Path ([IO.Path]::GetTempPath()) "autogpt-install-$([guid]::NewGuid().ToString('N')).ps1"
      try { curl.exe --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL -o $installer https://setup.agpt.co/install.ps1
            if ($LASTEXITCODE -eq 0) { powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer } }
      finally { Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue } }

Flags:
  -Release <tag>     published vX.Y.Z image (default: latest)
  -Dir <path>        private state directory (default: %LOCALAPPDATA%\AutoGPT)
  -SkipPreflight     skip RAM and disk checks; Docker checks remain mandatory
  -PreflightOnly     validate the machine and install nothing
  -ResolveOnly       print the selected image and install nothing
  -Help
'@
	exit 0
}

function Get-FlagValue([string]$Flag, [object[]]$Values, [int]$Index) {
	if ($Index -ge $Values.Count -or [string]::IsNullOrEmpty([string]$Values[$Index])) {
		Die "$Flag needs a value (try -Help)"
	}
	$value = [string]$Values[$Index]
	if ($value.StartsWith('-')) { Die "$Flag needs a value, not another option: $value" }
	$value
}

$Dir = $env:LOCALAPPDATA
if ([string]::IsNullOrEmpty($Dir)) { $Dir = [Environment]::GetFolderPath('LocalApplicationData') }
$Dir = $Dir.TrimEnd('\') + '\AutoGPT'

for ($index = 0; $index -lt $ScriptArgs.Count; $index++) {
	$argument = [string]$ScriptArgs[$index]
	$flag = $argument.ToLowerInvariant()
	# Run inside a PowerShell session, -Name:value arrives as '-Name:' and 'value'.
	if ($flag -match '^-[a-z]+:$') { $flag = $flag.TrimEnd(':') }
	$inline = $null
	if ($argument -match '^(--[A-Za-z-]+)=(.*)$') {
		$flag = $Matches[1].ToLowerInvariant()
		$inline = $Matches[2]
	}
	if ($flag -eq '-release' -or $flag -eq '--release') {
		if ($null -eq $inline) {
			$Release = Get-FlagValue $argument $ScriptArgs ($index + 1)
			$index++
		} elseif ($inline -eq '') {
			Die '--release needs a non-empty value (try -Help)'
		} else {
			$Release = $inline
		}
	} elseif ($flag -eq '-dir' -or $flag -eq '--dir') {
		if ($null -eq $inline) {
			$Dir = Get-FlagValue $argument $ScriptArgs ($index + 1)
			$index++
		} else {
			$Dir = $inline
		}
	} elseif ($flag -eq '-skippreflight' -or $flag -eq '--skip-preflight') {
		$SkipPreflight = $true
	} elseif ($flag -eq '-preflightonly' -or $flag -eq '--preflight-only') {
		$PreflightOnly = $true
	} elseif ($flag -eq '-resolveonly' -or $flag -eq '--resolve-only') {
		$ResolveOnly = $true
	} elseif ($flag -eq '-checkdironly' -or $flag -eq '--check-dir-only') {
		$CheckDirOnly = $true
	} elseif ($flag -eq '-help' -or $flag -eq '--help' -or $flag -eq '-h') {
		Show-Help
	} else {
		Die "Unknown flag: $argument (try -Help)"
	}
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
	Die "Unsupported OS: $([Environment]::OSVersion.Platform). This release installer is for Windows; on Linux and macOS use https://setup.agpt.co/install.sh."
}
if ($PSVersionTable.PSVersion.Major -lt 5) {
	Die "Windows PowerShell 5.1 or PowerShell 7 is required; this is PowerShell $($PSVersionTable.PSVersion)."
}
$Arch = $env:PROCESSOR_ARCHITECTURE
if ($env:PROCESSOR_ARCHITEW6432) { $Arch = $env:PROCESSOR_ARCHITEW6432 }

function Get-ReleaseImage([string]$Tag) {
	if ($Tag -cmatch '^autogpt-platform-beta-v([0-9]+\.[0-9]+\.[0-9]+)$' -or $Tag -cmatch '^v([0-9]+\.[0-9]+\.[0-9]+)$') {
		return "${DeployImage}:v$($Matches[1])"
	}
	Die "Unsupported release tag: ${Tag}. Expected autogpt-platform-beta-vX.Y.Z or vX.Y.Z."
}

function Resolve-Release {
	if ($Release) {
		$script:VerRef = $Release
		$script:Image = Get-ReleaseImage $Release
	} else {
		$script:VerRef = 'latest'
		$script:Image = "${DeployImage}:latest"
	}
}

function Test-PathEntry([string]$Path) {
	try {
		[void][System.IO.File]::GetAttributes($Path)
		return $true
	} catch [System.IO.FileNotFoundException], [System.IO.DirectoryNotFoundException] {
		return $false
	}
}

function Test-ReparsePoint([string]$Path) {
	([System.IO.File]::GetAttributes($Path) -band [System.IO.FileAttributes]::ReparsePoint) -ne 0
}

function Test-Directory([string]$Path) {
	([System.IO.File]::GetAttributes($Path) -band [System.IO.FileAttributes]::Directory) -ne 0
}

# Owner and allow rules that apply to the entry itself, by SID.
function Get-PathSecurity([string]$Path) {
	try {
		$acl = Get-Acl -LiteralPath $Path
	} catch {
		Die "Could not read the permissions of ${Path}: $($_.Exception.Message)"
	}
	$owner = $acl.GetOwner([System.Security.Principal.SecurityIdentifier])
	$rules = @()
	foreach ($rule in $acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])) {
		if ($rule.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) { continue }
		if (($rule.PropagationFlags -band [System.Security.AccessControl.PropagationFlags]::InheritOnly) -ne 0) { continue }
		$rules += [pscustomobject]@{ Sid = $rule.IdentityReference.Value; Rights = [int]$rule.FileSystemRights }
	}
	[pscustomobject]@{
		Owner = $(if ($owner) { $owner.Value } else { '' })
		NullDacl = ($acl.GetSecurityDescriptorSddlForm('Access') -match 'D:NO_ACCESS_CONTROL')
		Rules = $rules
	}
}

function Assert-ExistingPath([string[]]$Paths) {
	foreach ($current in $Paths) {
		if (-not (Test-PathEntry $current)) { return }
		if (Test-ReparsePoint $current) { Die "Install directory contains a symlink or junction component: $current" }
		if (-not (Test-Directory $current)) { Die "Install directory component is not a directory: $current" }
		$security = Get-PathSecurity $current
		if ($TrustedSids -notcontains $security.Owner) { Die "Install directory ancestor is owned by an untrusted user: $current" }
		if ($security.NullDacl) { Die "Install directory ancestor grants everyone full access: $current" }
		foreach ($rule in $security.Rules) {
			if ($TrustedSids -notcontains $rule.Sid -and ($rule.Rights -band $TamperRights) -ne 0) {
				Die "Install directory ancestor is writable by another user or group: $current ($($rule.Sid))"
			}
		}
	}
}

function Assert-InstallDirectory {
	if ([string]::IsNullOrEmpty($script:Dir)) { Die 'Install directory cannot be empty.' }
	$requested = $script:Dir.Replace('/', '\')
	if (-not [System.IO.Path]::IsPathRooted($requested)) {
		$requested = (Get-Location -PSProvider FileSystem).ProviderPath.TrimEnd('\') + '\' + $requested
	}
	if ($requested.StartsWith('\\')) { Die "Refusing a network or device path for the install directory: $($script:Dir)" }
	if ($requested -notmatch '^[A-Za-z]:\\') {
		Die "Install directory must be a full path such as C:\AutoGPT, or relative to the current folder: $($script:Dir)"
	}
	if ($requested.Substring(2).Contains('\\')) { Die "Install directory contains redundant path separators: $($script:Dir)" }
	$rest = $requested.Substring(3)
	if ($rest.EndsWith('\')) { $rest = $rest.Substring(0, $rest.Length - 1) }
	$drive = $requested.Substring(0, 1).ToUpperInvariant() + ':\'
	$driveType = (New-Object System.IO.DriveInfo ($drive.Substring(0, 1))).DriveType
	if ($driveType -eq [System.IO.DriveType]::Network) { Die "Refusing a network or device path for the install directory: $($script:Dir)" }
	if ($driveType -eq [System.IO.DriveType]::NoRootDirectory) { Die "Install directory drive does not exist: $drive" }
	$components = @()
	if ($rest) { $components = @($rest.Split([char]'\')) }
	foreach ($component in $components) {
		if ($component -eq '.') { Die "Install directory contains an unsafe '.' component: $($script:Dir)" }
		if ($component -eq '..') { Die "Install directory contains an unsafe '..' component: $($script:Dir)" }
		# Windows drops a trailing dot or space, a colon names a stream, and NAME~1
		# can be a short alias of another folder.
		if ($component -match '[<>:"|?*\x00-\x1f]' -or $component -match '[ .]$' -or $component -match '~[0-9]') {
			Die "Install directory contains an unsupported component: $component"
		}
	}
	$requested = $drive + ($components -join '\')

	$paths = @($drive)
	$current = $drive.TrimEnd('\')
	foreach ($component in $components) {
		$current = "$current\$component"
		$paths += $current
	}
	Assert-ExistingPath $paths

	$broad = @($drive)
	foreach ($name in 'SystemRoot', 'ProgramFiles', 'ProgramFiles(x86)', 'ProgramW6432', 'ProgramData', 'PUBLIC', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'TEMP', 'TMP') {
		$value = [Environment]::GetEnvironmentVariable($name)
		if ($value) { $broad += $value }
	}
	if ($env:USERPROFILE) { $broad += (Split-Path -Parent $env:USERPROFILE) }
	foreach ($candidate in $broad) {
		if ([string]::Equals($requested.TrimEnd('\'), $candidate.Replace('/', '\').TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
			Die "Refusing broad or system install directory: $requested"
		}
	}

	if (Test-PathEntry $requested) {
		if (Test-ReparsePoint $requested) { Die "Install directory is a symlink or junction: $requested" }
		if (-not (Test-Directory $requested)) { Die "Install directory is not a directory: $requested" }
		$security = Get-PathSecurity $requested
		if ($OwnerSids -notcontains $security.Owner) { Die "Install directory is not owned by the current user: $requested" }
		if ($security.NullDacl) { Die "Install directory grants everyone full access: $requested" }
		foreach ($rule in $security.Rules) {
			if ($PrivateSids -notcontains $rule.Sid) {
				Die "Install directory is accessible by another user or group: $requested ($($rule.Sid))"
			}
		}
	}
	$script:Dir = $requested
}

function New-PrivateSecurity([switch]$Directory) {
	if ($Directory) {
		$security = New-Object System.Security.AccessControl.DirectorySecurity
		$inheritance = [System.Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
	} else {
		$security = New-Object System.Security.AccessControl.FileSecurity
		$inheritance = [System.Security.AccessControl.InheritanceFlags]::None
	}
	$security.SetAccessRuleProtection($true, $false)
	$identity = New-Object System.Security.Principal.SecurityIdentifier $UserSid
	# An elevated run would otherwise leave Administrators as the owner on
	# Windows Server, and a later standard-user rerun would refuse the state.
	$security.SetOwner($identity)
	$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
		$identity,
		[System.Security.AccessControl.FileSystemRights]::FullControl,
		$inheritance,
		[System.Security.AccessControl.PropagationFlags]::None,
		[System.Security.AccessControl.AccessControlType]::Allow)
	$security.AddAccessRule($rule)
	$security
}

# Equivalent of chmod 0600/0700: a protected ACL that grants only this user.
function Set-PrivateAcl([string]$Path, [switch]$Directory) {
	$security = New-PrivateSecurity -Directory:$Directory
	if ($Directory) { $info = New-Object System.IO.DirectoryInfo $Path } else { $info = New-Object System.IO.FileInfo $Path }
	if ($info.PSObject.Methods['SetAccessControl']) {
		$info.SetAccessControl($security)
	} else {
		[System.IO.FileSystemAclExtensions]::SetAccessControl($info, $security)
	}
}

# Creates the directory with its private ACL in one step, like mkdir under umask 077.
function New-PrivateDirectory([string]$Path) {
	$security = New-PrivateSecurity -Directory
	if ([System.IO.Directory].GetMethod('CreateDirectory', [type[]]@([string], [System.Security.AccessControl.DirectorySecurity]))) {
		[void][System.IO.Directory]::CreateDirectory($Path, $security)
	} else {
		[void][System.IO.FileSystemAclExtensions]::Create((New-Object System.IO.DirectoryInfo $Path), $security)
	}
}

function Initialize-InstallDirectory {
	Assert-InstallDirectory
	if (Test-PathEntry $script:Dir) { return }
	$current = ''
	$created = $false
	foreach ($component in $script:Dir.Split([char]'\')) {
		if ($current) { $current = "$current\$component" } else { $current = $component }
		if ($current.Length -le 2) { continue }
		if (Test-PathEntry $current) {
			if ((Test-ReparsePoint $current) -or -not (Test-Directory $current)) {
				Die "Install directory changed during setup: $current"
			}
			continue
		}
		try {
			New-PrivateDirectory $current
		} catch {
			if (Test-PathEntry $current) { Die "Install directory changed during setup; refusing to reuse it: $current" }
			Die "Could not create the install directory ${current}: $($_.Exception.Message)"
		}
		$created = $true
	}
	if (-not $created) { Die "Install directory was not created safely: $($script:Dir)" }
	Assert-InstallDirectory
	Set-PrivateAcl $script:Dir -Directory
}

function Assert-PrivateFile([string]$File, [string]$Description) {
	if (-not (Test-PathEntry $File)) { Die "$Description is not a regular file: $File" }
	if (Test-ReparsePoint $File) { Die "Refusing symlink for ${Description}: $File" }
	if (Test-Directory $File) { Die "$Description is not a regular file: $File" }
	if ($OwnerSids -notcontains (Get-PathSecurity $File).Owner) { Die "$Description is not owned by the current user: $File" }
	if ((Get-Item -LiteralPath $File -Force).LinkType -eq 'HardLink') { Die "$Description has multiple hard links: $File" }
	Set-PrivateAcl $File
}

function Write-NewFile([string]$Path, [string]$Content) {
	$bytes = (New-Object System.Text.UTF8Encoding $false).GetBytes($Content)
	$stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
	try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
}

function Read-FileText([string]$Path) {
	[System.IO.File]::ReadAllText($Path).TrimEnd([char[]]@([char]13, [char]10))
}

function Initialize-State {
	$idFile = Join-Path $script:Dir 'install-id'
	$stateFile = Join-Path $script:Dir 'install-state'
	if (-not (Test-PathEntry $idFile)) {
		$bytes = New-Object byte[] 16
		$random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
		try { $random.GetBytes($bytes) } finally { $random.Dispose() }
		try {
			Write-NewFile $idFile (($bytes | ForEach-Object { $_.ToString('x2') }) -join '')
		} catch {
			Die 'Could not create a private installer identity.'
		}
	}
	Assert-PrivateFile $idFile 'Installer identity'
	$script:InstallId = Read-FileText $idFile
	if ($script:InstallId -cnotmatch '^[0-9a-f]{32}$') { Die "Installer identity has invalid contents: $idFile" }
	if (Test-PathEntry $stateFile) {
		Assert-PrivateFile $stateFile 'Installer lifecycle state'
		if ((Read-FileText $stateFile) -cne "contract=1 lifecycle=established volume=$DataVolume install-id=$($script:InstallId)") {
			Die "Installer lifecycle state has invalid contents: $stateFile"
		}
		$script:VolumeEstablished = $true
	}
}

function Set-InstallEstablished {
	if ($script:VolumeEstablished) { return }
	$stateFile = Join-Path $script:Dir 'install-state'
	$tempFile = Join-Path $script:Dir ('.install-state.' + [guid]::NewGuid().ToString('N'))
	try {
		Write-NewFile $tempFile "contract=1 lifecycle=established volume=$DataVolume install-id=$($script:InstallId)`n"
	} catch {
		Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
		Die 'Could not write installer lifecycle state.'
	}
	if (Test-PathEntry $stateFile) {
		Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
		Die "Installer lifecycle state appeared during setup: $stateFile"
	}
	try {
		[System.IO.File]::Move($tempFile, $stateFile)
	} catch {
		Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
		Die 'Could not commit installer lifecycle state.'
	}
	Assert-PrivateFile $stateFile 'Installer lifecycle state'
	$script:VolumeEstablished = $true
}

function Initialize-Environment {
	$script:EnvFile = Join-Path $script:Dir 'autogpt.env'
	if (-not (Test-PathEntry $script:EnvFile)) {
		Write-NewFile $script:EnvFile "# Private runtime configuration for the AutoGPT appliance.`nAUTOGPT_PUBLIC_URL=http://localhost:3000`n"
	}
	Assert-PrivateFile $script:EnvFile 'Runtime configuration'
	$script:EnvHash = (Get-FileHash -LiteralPath $script:EnvFile -Algorithm SHA256).Hash.ToLowerInvariant()
	if ($script:EnvHash -cnotmatch '^[0-9a-f]{64}$') { Die 'Could not hash the runtime configuration.' }
	Ok "Private runtime configuration: $($script:EnvFile)"
}

function Invoke-Preflight {
	Step 'Hardware pre-flight checks'
	Info "Host OS: windows ($Arch); the selected Docker daemon is validated separately."
	$ramGb = 0
	try {
		$ramGb = [int][math]::Floor((Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
	} catch {
		$ramGb = 0
	}
	if ($ramGb -eq 0) { Warn 'Could not determine installed RAM.' }
	elseif ($ramGb -lt $MinRamGb) { Warn "Only $ramGb GB RAM; $MinRamGb GB or more is recommended." }
	else { Ok "$ramGb GB RAM" }

	$target = $script:Dir
	while ($target.Length -gt 3 -and -not (Test-PathEntry $target)) { $target = Split-Path -Parent $target }
	$freeGb = 0
	try {
		$freeGb = [int][math]::Floor((New-Object System.IO.DriveInfo ($target.Substring(0, 1))).AvailableFreeSpace / 1GB)
	} catch {
		$freeGb = 0
	}
	if ($freeGb -lt $MinDiskGb) {
		Warn "Only $freeGb GB free on the state filesystem. Docker may use another storage root; verify at least $MinDiskGb GB there."
	} else {
		Ok "$freeGb GB free on the state filesystem (Docker storage is validated by the image pull)"
	}
}

function ConvertTo-ApplianceArch([string]$Value) {
	if ($Value -ceq 'amd64' -or $Value -ceq 'x86_64') { return 'amd64' }
	if ($Value -ceq 'arm64' -or $Value -ceq 'aarch64') { return 'arm64' }
	''
}

# Windows PowerShell 5.1 strips the double quotes inside arguments it passes to
# native programs, which breaks Go templates such as {{index .Config.Labels "x"}}.
# Quote each argument with the standard Windows rules instead.
function ConvertTo-NativeArgument([string]$Value) {
	if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') { return $Value }
	$builder = New-Object System.Text.StringBuilder
	[void]$builder.Append('"')
	$slashes = 0
	foreach ($char in $Value.ToCharArray()) {
		if ($char -eq [char]'\') {
			$slashes++
			continue
		}
		if ($char -eq [char]'"') {
			[void]$builder.Append([char]'\', 2 * $slashes + 1)
		} elseif ($slashes -gt 0) {
			[void]$builder.Append([char]'\', $slashes)
		}
		$slashes = 0
		[void]$builder.Append($char)
	}
	[void]$builder.Append([char]'\', 2 * $slashes)
	[void]$builder.Append('"')
	$builder.ToString()
}

function Invoke-DockerCapture([string[]]$Arguments) {
	$startInfo = New-Object System.Diagnostics.ProcessStartInfo
	$startInfo.FileName = $script:DockerExe
	$startInfo.Arguments = @($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
	$startInfo.UseShellExecute = $false
	$startInfo.CreateNoWindow = $true
	$startInfo.RedirectStandardOutput = $true
	$startInfo.RedirectStandardError = $true
	$startInfo.StandardOutputEncoding = New-Object System.Text.UTF8Encoding $false
	$startInfo.StandardErrorEncoding = New-Object System.Text.UTF8Encoding $false
	$process = [System.Diagnostics.Process]::Start($startInfo)
	$output = $process.StandardOutput.ReadToEndAsync()
	$errors = $process.StandardError.ReadToEndAsync()
	$process.WaitForExit()
	[pscustomobject]@{
		ExitCode = $process.ExitCode
		Output = $output.Result.TrimEnd([char[]]@([char]13, [char]10))
		Errors = $errors.Result
	}
}

function Get-DockerValue([string[]]$Arguments, [string]$Failure) {
	$result = Invoke-DockerCapture $Arguments
	if ($result.ExitCode -ne 0) { Die $Failure }
	$result.Output
}

function Test-LocalPipeEndpoint([string]$Endpoint) {
	if (-not $Endpoint.StartsWith('npipe://', [StringComparison]::Ordinal)) { return $false }
	$Endpoint.Substring(8).Replace('/', '\') -cmatch '^\\\\\.\\pipe\\[^\\]+$'
}

function Assert-Docker {
	Step 'Validating the local Docker daemon'
	$command = Get-Command -Name docker -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
	if (-not $command) {
		Die 'Docker is required. Install Docker Desktop from https://docs.docker.com/desktop/setup/install/windows-install/, start it, then retry.'
	}
	if ([System.IO.Path]::GetExtension($command.Source) -ne '.exe') {
		Die "The docker command must be docker.exe; found $($command.Source)."
	}
	$script:DockerExe = $command.Source
	if ($env:DOCKER_HOST) {
		$endpoint = $env:DOCKER_HOST
	} else {
		$context = Get-DockerValue @('context', 'show') 'Could not read the selected Docker context.'
		$endpoint = Get-DockerValue @('context', 'inspect', $context, '--format', '{{(index .Endpoints "docker").Host}}') 'Could not inspect the selected Docker context.'
	}
	if (-not (Test-LocalPipeEndpoint $endpoint)) {
		Die "Refusing non-local Docker endpoint: ${endpoint}. Select a local npipe:// context, such as Docker Desktop's desktop-linux."
	}
	if ((Invoke-DockerCapture @('info')).ExitCode -ne 0) {
		Die 'The local Docker daemon is not running or is not accessible. Start Docker Desktop and retry.'
	}
	$platform = @((Get-DockerValue @('info', '--format', '{{.OSType}} {{.Architecture}}') 'Could not inspect the Docker daemon platform.').Trim() -split '\s+')
	$daemonOs = $platform[0]
	$rawArch = ''
	if ($platform.Count -gt 1) { $rawArch = $platform[1] }
	if ($daemonOs -cne 'linux') {
		$mode = $daemonOs
		if (-not $mode) { $mode = 'an unknown mode' }
		Die "Docker is using $mode; switch Docker Desktop to Linux containers and retry."
	}
	$script:DaemonArch = ConvertTo-ApplianceArch $rawArch
	if (-not $script:DaemonArch) {
		if (-not $rawArch) { $rawArch = 'unknown' }
		Die "Unsupported Docker daemon architecture: ${rawArch}. Use amd64 or arm64."
	}
	Ok "Local Docker Linux daemon ($($script:DaemonArch))"
}

function Assert-Image {
	$title = Get-DockerValue @('image', 'inspect', '--format', '{{index .Config.Labels "org.opencontainers.image.title"}}', $script:Image) 'Could not inspect appliance image title.'
	$source = Get-DockerValue @('image', 'inspect', '--format', '{{index .Config.Labels "org.opencontainers.image.source"}}', $script:Image) 'Could not inspect appliance image source.'
	$revision = Get-DockerValue @('image', 'inspect', '--format', '{{index .Config.Labels "org.opencontainers.image.revision"}}', $script:Image) 'Could not inspect appliance image revision.'
	if ($title -cne $ExpectedTitle) { Die "Image $($script:Image) is not the AutoGPT appliance (unexpected OCI title)." }
	if ($source -cne $ExpectedSource) { Die "Image $($script:Image) is not from the expected AutoGPT source repository." }
	if ($revision -cnotmatch '^[0-9a-f]{40}$') { Die "Image $($script:Image) has an invalid source revision label." }
	$platform = @((Get-DockerValue @('image', 'inspect', '--format', '{{.Os}} {{.Architecture}}', $script:Image) 'Could not inspect appliance image platform.').Trim() -split '\s+')
	$imageOs = $platform[0]
	$rawArch = ''
	if ($platform.Count -gt 1) { $rawArch = $platform[1] }
	if ($imageOs -cne 'linux') { Die "Image $($script:Image) is not a Linux appliance image." }
	$imageArch = ConvertTo-ApplianceArch $rawArch
	if (-not $imageArch) {
		if (-not $rawArch) { $rawArch = 'unknown' }
		Die "Image $($script:Image) has unsupported architecture: ${rawArch}."
	}
	if ($imageArch -cne $script:DaemonArch) {
		Die "Image architecture $imageArch does not match Docker daemon architecture $($script:DaemonArch)."
	}

	$digestOutput = Get-DockerValue @('image', 'inspect', '--format', '{{range .RepoDigests}}{{println .}}{{end}}', $script:Image) 'Could not resolve the appliance repository digest.'
	$digests = @()
	foreach ($line in ($digestOutput -split "`n")) {
		$line = $line.TrimEnd([char]13)
		if (-not $line) { continue }
		if (-not ($line.StartsWith('docker.io/significantgravitas/autogpt@', [StringComparison]::Ordinal) -or $line.StartsWith('significantgravitas/autogpt@', [StringComparison]::Ordinal))) {
			Die "Image $($script:Image) resolved to an unexpected repository digest: $line"
		}
		$digest = $line.Substring($line.LastIndexOf('@') + 1)
		if ($digest -cnotmatch '^sha256:[0-9a-f]{64}$') { Die "Image $($script:Image) has an invalid repository digest." }
		$digests += $digest
	}
	$unique = @($digests | Sort-Object -Unique -CaseSensitive)
	if ($unique.Count -ne 1) {
		Die "Image $($script:Image) must resolve to exactly one native repository digest; found $($unique.Count)."
	}
	$script:ImageDigest = "$DeployImage@$($unique[0])"
	Ok "Appliance identity: $($script:ImageDigest)"
}

function Initialize-Image {
	Step "Pulling published appliance image: $($script:Image)"
	& $script:DockerExe pull $script:Image | Out-Host
	if ($LASTEXITCODE -ne 0) {
		Die "Published appliance image $($script:Image) is unavailable. Retry after publication completes or choose an existing -Release vX.Y.Z."
	}
	Assert-Image
}

function Assert-Volume {
	$format = "{{.Driver}} {{index .Labels `"$OwnerLabelKey`"}} {{index .Labels `"$ContractLabelKey`"}} {{index .Labels `"$InstallIdLabelKey`"}}"
	$expected = "local $OwnerLabelValue $ContractLabelValue $($script:InstallId)"
	if ((Invoke-DockerCapture @('volume', 'inspect', $DataVolume)).ExitCode -eq 0) {
		$metadata = Get-DockerValue @('volume', 'inspect', '--format', $format, $DataVolume) 'Could not inspect the persistent volume.'
		if ($metadata -cne $expected) { Die "Refusing existing unowned or incompatible volume: $DataVolume" }
		return
	}
	if ($script:VolumeEstablished) {
		Die "Installer state already exists but persistent volume $DataVolume is missing. Refusing to create an empty replacement; restore the volume or use a new -Dir."
	}
	$create = Invoke-DockerCapture @('volume', 'create', '--driver', 'local',
		'--label', "$OwnerLabelKey=$OwnerLabelValue",
		'--label', "$ContractLabelKey=$ContractLabelValue",
		'--label', "$InstallIdLabelKey=$($script:InstallId)",
		$DataVolume)
	if ($create.ExitCode -ne 0) { Die 'Could not create the persistent volume.' }
	$metadata = Get-DockerValue @('volume', 'inspect', '--format', $format, $DataVolume) 'Could not verify the persistent volume.'
	if ($metadata -cne $expected) { Die "Persistent volume ownership verification failed: $DataVolume" }
}

function Get-ContainerValue([string]$Format, [string]$Failure) {
	Get-DockerValue @('container', 'inspect', '--format', $Format, $ContainerName) $Failure
}

function Use-ExistingContainer {
	$labels = Get-ContainerValue "{{index .Config.Labels `"$OwnerLabelKey`"}} {{index .Config.Labels `"$ContractLabelKey`"}} {{index .Config.Labels `"$InstallIdLabelKey`"}} {{index .Config.Labels `"$EnvHashLabelKey`"}}" "Could not inspect existing container $ContainerName."
	if ($labels -cne "$OwnerLabelValue $ContractLabelValue $($script:InstallId) $($script:EnvHash)") {
		Die "Refusing existing container ${ContainerName}: installer ownership or configuration identity does not match."
	}

	$containerImage = Get-ContainerValue '{{.Config.Image}}' 'Could not inspect existing container image.'
	if ($containerImage -cne $script:ImageDigest) {
		Die "Refusing existing container ${ContainerName}: immutable appliance digest does not match $($script:ImageDigest)."
	}

	$runtime = Get-ContainerValue '{{.HostConfig.Privileged}}|{{.HostConfig.NetworkMode}}|{{.HostConfig.RestartPolicy.Name}}|{{.Config.StopTimeout}}|{{.HostConfig.ShmSize}}|{{.HostConfig.LogConfig.Type}}|{{index .HostConfig.LogConfig.Config "max-size"}}|{{index .HostConfig.LogConfig.Config "max-file"}}|{{len .HostConfig.Ulimits}}|{{range .HostConfig.Ulimits}}{{.Name}}:{{.Soft}}:{{.Hard}}{{end}}|{{len .Mounts}}|{{range .Mounts}}{{.Type}}:{{.Name}}:{{.Destination}}:{{.RW}}{{end}}|{{len .HostConfig.PortBindings}}|{{range $key, $bindings := .HostConfig.PortBindings}}{{$key}}={{range $bindings}}{{.HostIp}}:{{.HostPort}}{{end}}{{end}}' 'Could not inspect existing container runtime contract.'
	if ($runtime -cne 'false|bridge|unless-stopped|360|2147483648|json-file|50m|5|1|nofile:65536:65536|1|volume:autogpt-platform-data:/data:true|1|3000/tcp=127.0.0.1:3000') {
		Die "Refusing existing container ${ContainerName}: critical runtime contract has drifted."
	}

	$process = Get-ContainerValue '{{json .Config.Entrypoint}}|{{json .Config.Cmd}}' 'Could not inspect existing container process contract.'
	if ($process -cne '["/usr/bin/tini","--","/opt/autogpt/single-container/entrypoint.sh"]|["/usr/bin/supervisord","-n","-c","/opt/autogpt/single-container/supervisor/supervisord.conf"]') {
		Die "Refusing existing container ${ContainerName}: entrypoint or command was overridden."
	}

	$state = Get-ContainerValue '{{.State.Status}}' 'Could not inspect existing container state.'
	if ($state -ceq 'running') {
		$health = Get-ContainerValue '{{if .State.Health}}{{.State.Health.Status}}{{end}}' 'Could not inspect existing container health.'
		if ($health -ceq 'healthy') {
			Ok 'Existing installer-owned appliance already matches the requested immutable contract.'
		} elseif ($health -ceq 'starting') {
			Info 'Existing installer-owned appliance is still starting; continuing to the health wait.'
		} else {
			if (-not $health) { $health = 'none' }
			Die "Existing installer-owned container is running but not healthy (health=$health)."
		}
	} elseif ($state -ceq 'created' -or $state -ceq 'exited') {
		Step 'Starting the existing installer-owned appliance'
		& $script:DockerExe start $ContainerName | Out-Null
		if ($LASTEXITCODE -ne 0) { Die 'Could not start the existing installer-owned appliance.' }
	} else {
		Die "Existing installer-owned container is in unsupported state: $state"
	}
}

function Start-Appliance {
	Assert-Volume
	Set-InstallEstablished
	if ((Invoke-DockerCapture @('container', 'inspect', $ContainerName)).ExitCode -eq 0) {
		Use-ExistingContainer
		return
	}
	Step 'Starting the AutoGPT appliance'
	& $script:DockerExe run --detach `
		--name $ContainerName `
		--label "$OwnerLabelKey=$OwnerLabelValue" `
		--label "$ContractLabelKey=$ContractLabelValue" `
		--label "$InstallIdLabelKey=$($script:InstallId)" `
		--label "$EnvHashLabelKey=$($script:EnvHash)" `
		--restart unless-stopped `
		--stop-timeout 360 `
		--shm-size 2g `
		--ulimit nofile=65536:65536 `
		--log-driver json-file `
		--log-opt max-size=50m `
		--log-opt max-file=5 `
		--env-file $script:EnvFile `
		--publish 127.0.0.1:3000:3000 `
		--volume "${DataVolume}:/data" `
		$script:ImageDigest | Out-Null
	if ($LASTEXITCODE -ne 0) { Die 'Could not start the appliance. The persistent volume was not removed.' }
}

function Wait-Healthy {
	Step 'Waiting for the appliance to become healthy'
	for ($attempt = 1; $attempt -le 180; $attempt++) {
		$status = Invoke-DockerCapture @('inspect', '--format', '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}', $ContainerName)
		if ($status.ExitCode -ne 0) { Die "Could not inspect appliance container ${ContainerName}; it may have been removed." }
		$parts = @($status.Output.Trim() -split '\s+', 2)
		$state = $parts[0]
		$health = ''
		if ($parts.Count -gt 1) { $health = $parts[1] }
		if ($state -ceq 'running' -and $health -ceq 'healthy') {
			Ok 'AutoGPT is healthy'
			return
		}
		if ($state -ceq 'exited' -or $state -ceq 'dead' -or $state -ceq 'removing' -or $health -ceq 'unhealthy') {
			& $script:DockerExe logs --tail 100 $ContainerName | Out-Host
			if (-not $health) { $health = 'none' }
			Die "The appliance failed while starting (state=$state, health=$health)."
		}
		Start-Sleep -Seconds 5
	}
	& $script:DockerExe logs --tail 100 $ContainerName | Out-Host
	Die 'The appliance did not become healthy within 15 minutes.'
}

Say ''
Say '============================================='
Say '         AutoGPT Platform Installer'
Say '============================================='

if ($CheckDirOnly) {
	Assert-InstallDirectory
	Info "Validated install directory: $Dir"
	exit 0
}

Resolve-Release
if ($ResolveOnly) {
	Info "Selected release -> $VerRef -> $Image"
	exit 0
}

Assert-InstallDirectory
if ($SkipPreflight) { Warn 'Hardware pre-flight skipped; Docker locality and appliance identity checks remain mandatory.' } else { Invoke-Preflight }
Assert-Docker
if ($PreflightOnly) {
	Say ''
	Say '(-PreflightOnly: nothing was installed.)'
	exit 0
}

Initialize-InstallDirectory
Initialize-State
Initialize-Environment
Initialize-Image
Info "Selected release -> $VerRef -> $ImageDigest"
Start-Appliance
Wait-Healthy

Say ''
Say '============================================='
Say '  AutoGPT is ready at http://localhost:3000'
Say '============================================='
Say "Configuration: $EnvFile"
Say "Persistent data: Docker volume $DataVolume"
Say "Immutable appliance: $ImageDigest"
Say 'Registration starts open. Create the intended account, promote it, then close signup as documented.'
