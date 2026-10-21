# Tests for the Windows release installer (install.ps1), its install.bat shim,
# and the documented bootstrap command, using fake docker.exe, curl.exe and
# powershell.exe programs. The cases follow test-single-container-handoff.sh.
# Run it with both PowerShell editions:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File autogpt_platform/installer/test-windows-installer.ps1
#   pwsh -NoProfile -File autogpt_platform/installer/test-windows-installer.ps1
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$installerDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $installerDir '..\..')).Path
$installer = Join-Path $installerDir 'install.ps1'
$shim = Join-Path $installerDir 'install.bat'
$powerShellExe = (Get-Process -Id $PID).Path
$userSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$everyoneSid = 'S-1-1-0'
$systemPath = "$env:SystemRoot\System32;$env:SystemRoot"
$junctions = New-Object System.Collections.Generic.List[string]

$fakeSource = @'
using System;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

public static class FakeTool
{
    static string Env(string name, string fallback)
    {
        string value = Environment.GetEnvironmentVariable(name);
        return value == null ? fallback : value;
    }

    static bool Enabled(string name) { return Env(name, "false") == "true"; }

    static string Json(string[] values)
    {
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < values.Length; i++)
        {
            if (i > 0) builder.Append(',');
            builder.Append('"');
            foreach (char c in values[i])
            {
                if (c == '\\' || c == '"') builder.Append('\\').Append(c);
                else if (c < ' ') builder.Append("\\u").Append(((int)c).ToString("x4"));
                else builder.Append(c);
            }
            builder.Append('"');
        }
        return builder.Append(']').ToString();
    }

    static void Record(string variable, string line)
    {
        string path = Environment.GetEnvironmentVariable(variable);
        if (!string.IsNullOrEmpty(path)) File.AppendAllText(path, line + "\n");
    }

    static string ValueAfter(string[] args, string flag)
    {
        int index = Array.IndexOf(args, flag);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    static int Fail(string message, int code)
    {
        Console.Error.WriteLine(message);
        return code;
    }

    static string ConfigLabel(string key) { return "{{index .Config.Labels \"" + key + "\"}}"; }

    static string VolumeLabel(string key) { return "{{index .Labels \"" + key + "\"}}"; }

    static string InstallFile(string name) { return Path.Combine(Env("FAKE_INSTALL_DIR", ""), name); }

    static string Sha256(string path)
    {
        using (SHA256 sha = SHA256.Create())
        {
            StringBuilder builder = new StringBuilder();
            foreach (byte b in sha.ComputeHash(File.ReadAllBytes(path))) builder.Append(b.ToString("x2"));
            return builder.ToString();
        }
    }

    public static int Main(string[] args)
    {
        string name = Path.GetFileNameWithoutExtension(Assembly.GetEntryAssembly().Location).ToLowerInvariant();
        if (name == "curl") return Curl(args);
        if (name == "powershell") return PowerShell(args);
        return Docker(args);
    }

    static int Curl(string[] args)
    {
        Record("FAKE_CURL_LOG", Json(args));
        string output = ValueAfter(args, "-o");
        if (output == null) return Fail("fake curl: no -o", 2);
        if (Env("FAKE_CURL_MODE", "ok") == "fail")
        {
            File.WriteAllText(output, "partial installer\n");
            return 22;
        }
        File.WriteAllText(output, "Write-Host 'fake installer'\n");
        return 0;
    }

    static int PowerShell(string[] args)
    {
        Record("FAKE_POWERSHELL_LOG", Json(args));
        string file = ValueAfter(args, "-File");
        if (file != null && File.Exists(file)) Record("FAKE_POWERSHELL_LOG", "target-existed");
        return int.Parse(Env("FAKE_POWERSHELL_EXIT", "0"));
    }

    static int Docker(string[] args)
    {
        Record("FAKE_DOCKER_LOG", Json(args));
        string a0 = args.Length > 0 ? args[0] : "";
        string a1 = args.Length > 1 ? args[1] : "";
        string format = ValueAfter(args, "--format");
        if (a0 == "context" && a1 == "show")
        {
            Console.WriteLine("default");
            return 0;
        }
        if (a0 == "context" && a1 == "inspect")
        {
            if (format != "{{(index .Endpoints \"docker\").Host}}") return Fail("unexpected context inspect format: " + format, 73);
            Console.WriteLine(Env("FAKE_DOCKER_ENDPOINT", "npipe:////./pipe/dockerDesktopLinuxEngine"));
            return 0;
        }
        if (a0 == "info")
        {
            if (Enabled("FAKE_DOCKER_INFO_FAIL")) return 1;
            if (format == null) return 0;
            if (format != "{{.OSType}} {{.Architecture}}") return Fail("unexpected info format: " + format, 74);
            Console.WriteLine(Env("FAKE_DOCKER_OS", "linux") + " " + Env("FAKE_DOCKER_ARCH", "x86_64"));
            return 0;
        }
        if (a0 == "pull") return Enabled("FAKE_DOCKER_PULL_FAIL") ? 1 : 0;
        if (a0 == "image" && a1 == "inspect")
        {
            if (format == ConfigLabel("org.opencontainers.image.title"))
            {
                Console.WriteLine(Env("FAKE_IMAGE_TITLE", "AutoGPT Platform single-container"));
                return 0;
            }
            if (format == ConfigLabel("org.opencontainers.image.source"))
            {
                Console.WriteLine(Env("FAKE_IMAGE_SOURCE", "https://github.com/Significant-Gravitas/AutoGPT"));
                return 0;
            }
            if (format == ConfigLabel("org.opencontainers.image.revision"))
            {
                Console.WriteLine(Env("FAKE_IMAGE_REVISION", new string('a', 40)));
                return 0;
            }
            if (format == "{{.Os}} {{.Architecture}}")
            {
                Console.WriteLine(Env("FAKE_IMAGE_OS", "linux") + " " + Env("FAKE_IMAGE_ARCH", "amd64"));
                return 0;
            }
            if (format == "{{range .RepoDigests}}{{println .}}{{end}}")
            {
                string digests = Env("FAKE_IMAGE_DIGESTS", "significantgravitas/autogpt@sha256:" + new string('b', 64) + "\\n");
                if (digests != "none") Console.Write(digests.Replace("\\n", "\n"));
                return 0;
            }
            return Fail("unexpected image inspect format: " + format, 71);
        }
        if (a0 == "container" && a1 == "inspect")
        {
            if (!Enabled("FAKE_CONTAINER_EXISTS")) return 1;
            if (format == null) return 0;
            if (format == ConfigLabel("com.agpt.installer.owner") + " " + ConfigLabel("com.agpt.installer.contract") + " " + ConfigLabel("com.agpt.installer.install-id") + " " + ConfigLabel("com.agpt.installer.env-sha256"))
            {
                string labels = Env("FAKE_CONTAINER_LABELS", null);
                if (labels == null) labels = "autogpt-platform-installer 1 " + File.ReadAllText(InstallFile("install-id")).Trim() + " " + Sha256(InstallFile("autogpt.env"));
                Console.WriteLine(labels);
                return 0;
            }
            if (format == "{{.Config.Image}}")
            {
                Console.WriteLine(Env("FAKE_CONTAINER_IMAGE", "docker.io/significantgravitas/autogpt@sha256:" + new string('b', 64)));
                return 0;
            }
            if (format.StartsWith("{{.HostConfig.Privileged}}|") && format.Contains("{{index .HostConfig.LogConfig.Config \"max-size\"}}") && format.Contains("{{range $key, $bindings := .HostConfig.PortBindings}}"))
            {
                Console.WriteLine(Env("FAKE_CONTAINER_RUNTIME", "false|bridge|unless-stopped|360|2147483648|json-file|50m|5|1|nofile:65536:65536|1|volume:autogpt-platform-data:/data:true|1|3000/tcp=127.0.0.1:3000"));
                return 0;
            }
            if (format == "{{json .Config.Entrypoint}}|{{json .Config.Cmd}}")
            {
                Console.WriteLine(Env("FAKE_CONTAINER_PROCESS", "[\"/usr/bin/tini\",\"--\",\"/opt/autogpt/single-container/entrypoint.sh\"]|[\"/usr/bin/supervisord\",\"-n\",\"-c\",\"/opt/autogpt/single-container/supervisor/supervisord.conf\"]"));
                return 0;
            }
            if (format == "{{.State.Status}}")
            {
                Console.WriteLine(Env("FAKE_CONTAINER_STATE", "running"));
                return 0;
            }
            if (format == "{{if .State.Health}}{{.State.Health.Status}}{{end}}")
            {
                Console.WriteLine(Env("FAKE_CONTAINER_HEALTH", "healthy"));
                return 0;
            }
            return Fail("unexpected container inspect format: " + format, 72);
        }
        if (a0 == "volume" && a1 == "inspect")
        {
            string statePath = Env("FAKE_VOLUME_STATE", "");
            string state = File.Exists(statePath) ? File.ReadAllText(statePath).Trim() : "";
            if (state == "") return 1;
            if (format == null) return 0;
            if (format != "{{.Driver}} " + VolumeLabel("com.agpt.installer.owner") + " " + VolumeLabel("com.agpt.installer.contract") + " " + VolumeLabel("com.agpt.installer.install-id")) return Fail("unexpected volume inspect format: " + format, 75);
            if (state == "unowned") Console.WriteLine("local <no value> <no value> <no value>");
            else Console.WriteLine("local autogpt-platform-installer 1 " + File.ReadAllText(InstallFile("install-id")).Trim());
            return 0;
        }
        if (a0 == "volume" && a1 == "create")
        {
            File.WriteAllText(Env("FAKE_VOLUME_STATE", ""), "owned\n");
            Console.WriteLine("autogpt-platform-data");
            return 0;
        }
        if (a0 == "run")
        {
            if (Enabled("FAKE_DOCKER_RUN_FAIL")) return 1;
            string envFile = ValueAfter(args, "--env-file");
            if (envFile != InstallFile("autogpt.env")) return Fail("unexpected --env-file: " + envFile, 76);
            Console.WriteLine("fake-container-id");
            return 0;
        }
        if (a0 == "start")
        {
            Console.WriteLine(a1);
            return 0;
        }
        if (a0 == "inspect")
        {
            if (Enabled("FAKE_HEALTH_INSPECT_FAIL")) return 1;
            if (format != "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}") return Fail("unexpected inspect format: " + format, 77);
            Console.WriteLine(Env("FAKE_HEALTH_STATUS", "running healthy"));
            return 0;
        }
        if (a0 == "logs") return 0;
        return Fail("unexpected fake docker invocation: " + string.Join(" ", args), 70);
    }
}
'@

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

function Invoke-Program([string]$FilePath, [string[]]$Arguments, [hashtable]$Environment) {
	$startInfo = New-Object System.Diagnostics.ProcessStartInfo
	$startInfo.FileName = $FilePath
	$startInfo.Arguments = @($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
	$startInfo.UseShellExecute = $false
	$startInfo.CreateNoWindow = $true
	$startInfo.RedirectStandardOutput = $true
	$startInfo.RedirectStandardError = $true
	foreach ($key in @($Environment.Keys)) {
		if ($null -eq $Environment[$key]) {
			[void]$startInfo.EnvironmentVariables.Remove($key)
		} else {
			$startInfo.EnvironmentVariables[$key] = [string]$Environment[$key]
		}
	}
	$process = [System.Diagnostics.Process]::Start($startInfo)
	$stdout = $process.StandardOutput.ReadToEndAsync()
	$stderr = $process.StandardError.ReadToEndAsync()
	$process.WaitForExit()
	[pscustomobject]@{ ExitCode = $process.ExitCode; Output = $stdout.Result + $stderr.Result }
}

function Assert-Contains([string]$Expected, [string]$Actual) {
	if (-not $Actual.Contains($Expected)) { throw "missing expected text: $Expected`nactual output:`n$Actual" }
}

function Assert-NotContains([string]$Unexpected, [string]$Actual) {
	if ($Actual.Contains($Unexpected)) { throw "unexpected text: $Unexpected`nactual output:`n$Actual" }
}

function Assert-Succeeded($Result, [string]$What) {
	if ($Result.ExitCode -ne 0) { throw "$What failed with exit code $($Result.ExitCode):`n$($Result.Output)" }
}

function Assert-FailsWith([string]$Expected, $Result) {
	if ($Result.ExitCode -eq 0) { throw "command unexpectedly succeeded; expected: $Expected`n$($Result.Output)" }
	Assert-Contains $Expected $Result.Output
}

function Invoke-Installer([hashtable]$Environment = @{}, [string[]]$Arguments = @()) {
	$merged = @{
		PATH = "$fakeBin;$systemPath"
		FAKE_DOCKER_LOG = $dockerLog
		FAKE_VOLUME_STATE = $volumeState
		DOCKER_HOST = $null
		DOCKER_CONTEXT = $null
	}
	foreach ($key in @($Environment.Keys)) { $merged[$key] = $Environment[$key] }
	Invoke-Program $powerShellExe (@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $installer) + $Arguments) $merged
}

function Invoke-FakeInstall([string]$InstallDir, [hashtable]$Environment = @{}, [string[]]$Arguments = @()) {
	$merged = @{ FAKE_INSTALL_DIR = $InstallDir }
	foreach ($key in @($Environment.Keys)) { $merged[$key] = $Environment[$key] }
	Invoke-Installer $merged (@('-SkipPreflight', '-Dir', $InstallDir) + $Arguments)
}

# Windows PowerShell 5.1 returns a JSON array as one object; piping unrolls it.
function ConvertFrom-ArgumentLog([string]$Line) {
	@(ConvertFrom-Json $Line | ForEach-Object { $_ })
}

function Get-DockerLog {
	if (-not (Test-Path -LiteralPath $dockerLog)) { return '' }
	$calls = foreach ($line in [System.IO.File]::ReadAllLines($dockerLog)) { (ConvertFrom-ArgumentLog $line) -join ' ' }
	@($calls) -join "`n"
}

function Get-RunCall {
	@((Get-DockerLog) -split "`n" | Where-Object { $_.StartsWith('run ') }) -join "`n"
}

function Clear-File([string]$Path, [string]$Content = '') {
	[System.IO.File]::WriteAllText($Path, $Content)
}

function New-PrivateSecurity {
	$security = New-Object System.Security.AccessControl.DirectorySecurity
	$security.SetAccessRuleProtection($true, $false)
	$security.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
				(New-Object System.Security.Principal.SecurityIdentifier $userSid),
				[System.Security.AccessControl.FileSystemRights]::FullControl,
				[System.Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit',
				[System.Security.AccessControl.PropagationFlags]::None,
				[System.Security.AccessControl.AccessControlType]::Allow)))
	$security
}

function New-PrivateDirectory([string]$Path) {
	if ([System.IO.Directory].GetMethod('CreateDirectory', [type[]]@([string], [System.Security.AccessControl.DirectorySecurity]))) {
		[void][System.IO.Directory]::CreateDirectory($Path, (New-PrivateSecurity))
	} else {
		[void][System.IO.FileSystemAclExtensions]::Create((New-Object System.IO.DirectoryInfo $Path), (New-PrivateSecurity))
	}
}

function Get-DirectorySecurity([string]$Path) {
	$info = New-Object System.IO.DirectoryInfo $Path
	if ($info.PSObject.Methods['GetAccessControl']) { return $info.GetAccessControl() }
	[System.IO.FileSystemAclExtensions]::GetAccessControl($info)
}

function Add-AllowRule([string]$Path, [string]$Sid, [string]$Rights) {
	$info = New-Object System.IO.DirectoryInfo $Path
	$security = Get-DirectorySecurity $Path
	$security.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
				(New-Object System.Security.Principal.SecurityIdentifier $Sid),
				[System.Security.AccessControl.FileSystemRights]$Rights,
				[System.Security.AccessControl.AccessControlType]::Allow)))
	if ($info.PSObject.Methods['SetAccessControl']) { $info.SetAccessControl($security) } else { [System.IO.FileSystemAclExtensions]::SetAccessControl($info, $security) }
}

function New-Junction([string]$Link, [string]$Target) {
	New-Item -ItemType Junction -Path $Link -Target $Target | Out-Null
	$junctions.Add($Link)
}

function New-FileSymlink([string]$Link, [string]$Target) {
	try {
		New-Item -ItemType SymbolicLink -Path $Link -Target $Target -ErrorAction Stop | Out-Null
	} catch {
		& "$env:SystemRoot\System32\cmd.exe" /d /c mklink $Link $Target 2>&1 | Out-Null
	}
	Test-Path -LiteralPath $Link
}

function Assert-PrivateAcl([string]$Path) {
	$acl = Get-Acl -LiteralPath $Path
	$rules = @($acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]))
	if (-not $acl.AreAccessRulesProtected -or $rules.Count -ne 1 -or $rules[0].IdentityReference.Value -ne $userSid -or $rules[0].AccessControlType -ne 'Allow') {
		throw "$Path is not private to the current user: $($acl.GetSecurityDescriptorSddlForm('Access'))"
	}
}

# A test root must pass the installer's own ancestor checks. Try the places a
# standard user can usually create a private folder.
function Select-TestRoot {
	$name = 'agpt-installer-test-' + [guid]::NewGuid().ToString('N').Substring(0, 12)
	$candidates = @()
	foreach ($base in @("$env:SystemDrive\", $env:TEMP, $env:LOCALAPPDATA, $env:RUNNER_TEMP)) {
		if ($base -and (Test-Path -LiteralPath $base)) { $candidates += (Join-Path $base $name) }
	}
	$reasons = @()
	foreach ($candidate in $candidates) {
		$probe = Invoke-Program $powerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $installer, '-CheckDirOnly', '-Dir', $candidate) @{}
		if ($probe.ExitCode -eq 0) { return $candidate }
		$reasons += "${candidate}: $($probe.Output.Trim())"
	}
	throw "no folder on this machine passes the installer's ancestor checks:`n$($reasons -join "`n")"
}

$root = Select-TestRoot
New-PrivateDirectory $root
Write-Host "test root: $root"
try {
	$fakeBin = Join-Path $root 'fake-bin'
	New-Item -ItemType Directory -Path $fakeBin | Out-Null
	$csc = "$env:SystemRoot\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
	if (-not (Test-Path -LiteralPath $csc)) { $csc = "$env:SystemRoot\Microsoft.NET\Framework\v4.0.30319\csc.exe" }
	$sourceFile = Join-Path $root 'FakeTool.cs'
	[System.IO.File]::WriteAllText($sourceFile, $fakeSource)
	$compiled = Join-Path $root 'FakeTool.exe'
	$compile = Invoke-Program $csc @('/nologo', '/target:exe', "/out:$compiled", $sourceFile) @{}
	Assert-Succeeded $compile 'compiling the fake tools'
	foreach ($tool in 'docker', 'curl', 'powershell') { Copy-Item -LiteralPath $compiled -Destination (Join-Path $fakeBin "$tool.exe") }
	$dockerLog = Join-Path $root 'docker.log'
	$volumeState = Join-Path $root 'volume.state'

	# Default and explicit directories.
	$defaultLocal = Join-Path $root 'default-local'
	$result = Invoke-Installer @{ LOCALAPPDATA = $defaultLocal } @('-CheckDirOnly')
	Assert-Succeeded $result 'default directory check'
	Assert-Contains "Validated install directory: $defaultLocal\AutoGPT" $result.Output
	$result = Invoke-Installer @{ LOCALAPPDATA = "$defaultLocal\" } @('-CheckDirOnly')
	Assert-Contains "Validated install directory: $defaultLocal\AutoGPT" $result.Output

	# Release selection and flags.
	$result = Invoke-Installer @{} @('-ResolveOnly')
	Assert-Contains 'Selected release -> latest -> docker.io/significantgravitas/autogpt:latest' $result.Output
	$result = Invoke-Installer @{} @('-ResolveOnly', '-Release', 'autogpt-platform-beta-v1.2.3')
	Assert-Contains 'Selected release -> autogpt-platform-beta-v1.2.3 -> docker.io/significantgravitas/autogpt:v1.2.3' $result.Output
	$result = Invoke-Installer @{} @('--resolve-only', '--release=v2.3.4')
	Assert-Contains 'Selected release -> v2.3.4 -> docker.io/significantgravitas/autogpt:v2.3.4' $result.Output
	$result = Invoke-Installer @{} @('-ResolveOnly', '-Release:v3.4.5')
	Assert-Contains 'Selected release -> v3.4.5 -> docker.io/significantgravitas/autogpt:v3.4.5' $result.Output
	# Run inside a session instead of with -File, PowerShell splits -Name:value differently.
	$inSession = "& '$installer' -ResolveOnly -Release:v4.5.6"
	$result = Invoke-Program $powerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $inSession) @{}
	Assert-Contains 'Selected release -> v4.5.6 -> docker.io/significantgravitas/autogpt:v4.5.6' $result.Output
	$sessionDir = Join-Path $root 'session-dir'
	$result = Invoke-Program $powerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', "& '$installer' -CheckDirOnly -Dir:$sessionDir") @{}
	Assert-Contains "Validated install directory: $sessionDir" $result.Output
	Assert-FailsWith 'Expected autogpt-platform-beta-vX.Y.Z or vX.Y.Z' (Invoke-Installer @{} @('-ResolveOnly', '-Release', 'latest'))
	Assert-FailsWith 'Expected autogpt-platform-beta-vX.Y.Z or vX.Y.Z' (Invoke-Installer @{} @('-ResolveOnly', '-Release', 'V1.2.3'))
	Assert-FailsWith 'needs a non-empty value' (Invoke-Installer @{} @('-ResolveOnly', '--release='))
	Assert-FailsWith 'Unknown flag: -Dev' (Invoke-Installer @{} @('-ResolveOnly', '-Dev'))
	Assert-FailsWith 'Unknown flag: --branch' (Invoke-Installer @{} @('-ResolveOnly', '--branch', 'dev'))
	Assert-FailsWith 'needs a value, not another option' (Invoke-Installer @{} @('-Release', '-ResolveOnly'))
	Assert-FailsWith 'needs a value, not another option' (Invoke-Installer @{} @('-Dir', '-ResolveOnly'))
	Assert-FailsWith '-Release needs a value' (Invoke-Installer @{} @('-ResolveOnly', '-Release'))
	$result = Invoke-Installer @{} @('-Help')
	Assert-Succeeded $result '-Help'
	Assert-Contains "curl.exe --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL" $result.Output

	# Unsafe or unsupported locations.
	Assert-FailsWith "Refusing broad or system install directory: $env:SystemDrive\" (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$env:SystemDrive\"))
	$fakeLocal = Join-Path $root 'fake-local'
	Assert-FailsWith 'Refusing broad or system install directory' (Invoke-Installer @{ LOCALAPPDATA = $fakeLocal } @('-CheckDirOnly', '-Dir', $fakeLocal))
	$fakeTemp = Join-Path $root 'fake-temp'
	Assert-FailsWith 'Refusing broad or system install directory' (Invoke-Installer @{ TEMP = $fakeTemp; TMP = $fakeTemp } @('-CheckDirOnly', '-Dir', "$fakeTemp\"))
	Assert-FailsWith "contains an unsafe '.' component" (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$root\."))
	Assert-FailsWith "contains an unsafe '..' component" (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$root\..\x"))
	Assert-FailsWith 'contains redundant path separators' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$root\\x"))
	Assert-FailsWith 'contains redundant path separators' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$($env:SystemDrive)\\Windows"))
	Assert-FailsWith 'unsupported component: x.' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$root\x."))
	Assert-FailsWith 'unsupported component: a:b' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$root\a:b"))
	Assert-FailsWith 'unsupported component: PROGRA~1' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$root\PROGRA~1"))
	Assert-FailsWith 'Refusing a network or device path' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', '\\server\share\autogpt'))
	$unusedDrive = [char[]](90..68) | Where-Object { -not (Test-Path -LiteralPath "${_}:\") } | Select-Object -First 1
	if ($unusedDrive) {
		Assert-FailsWith "Install directory drive does not exist: ${unusedDrive}:\" (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "${unusedDrive}:\autogpt"))
	}
	Assert-FailsWith 'must be a full path' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$($env:SystemDrive)autogpt"))
	Assert-FailsWith 'Install directory cannot be empty' (Invoke-Installer @{} @('-CheckDirOnly', '--dir='))

	$existingDir = Join-Path $root 'existing-install'
	New-Item -ItemType Directory -Path $existingDir | Out-Null
	Assert-Succeeded (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', $existingDir)) 'existing private directory'

	$readableDir = Join-Path $root 'readable-install'
	New-Item -ItemType Directory -Path $readableDir | Out-Null
	Add-AllowRule $readableDir $everyoneSid 'ReadAndExecute'
	Assert-FailsWith 'is accessible by another user or group' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', $readableDir))

	$sharedDir = Join-Path $root 'shared-install'
	New-Item -ItemType Directory -Path $sharedDir | Out-Null
	Add-AllowRule $sharedDir $everyoneSid 'Modify'
	Assert-FailsWith 'writable by another user or group' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', $sharedDir))
	Assert-FailsWith 'writable by another user or group' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$sharedDir\private-state"))

	# Creating new entries only is the sticky-directory case: it cannot replace
	# a folder the installer creates.
	$stickyParent = Join-Path $root 'sticky-parent'
	New-Item -ItemType Directory -Path $stickyParent | Out-Null
	Add-AllowRule $stickyParent $everyoneSid 'CreateFiles, CreateDirectories, ReadAndExecute'
	Assert-Succeeded (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$stickyParent\private-state")) 'create-only parent'

	$junctionTarget = Join-Path $root 'junction-target'
	New-Item -ItemType Directory -Path $junctionTarget | Out-Null
	$junctionDir = Join-Path $root 'junction-install'
	New-Junction $junctionDir $junctionTarget
	Assert-FailsWith 'symlink or junction' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', $junctionDir))
	Assert-FailsWith 'symlink or junction component' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$junctionDir\child"))

	$fileInPath = Join-Path $root 'not-a-directory'
	Clear-File $fileInPath 'x'
	Assert-FailsWith 'component is not a directory' (Invoke-Installer @{} @('-CheckDirOnly', '-Dir', "$fileInPath\child"))

	# Docker locality, platform and identity checks, with the fake docker.exe.
	Clear-File $dockerLog
	Clear-File $volumeState
	$defaultDir = Join-Path $root 'default install'
	Assert-Succeeded (Invoke-FakeInstall $defaultDir) 'default fake install'
	$log = Get-DockerLog
	Assert-Contains 'pull docker.io/significantgravitas/autogpt:latest' $log
	Assert-Contains 'run --detach' $log
	Assert-Contains ('docker.io/significantgravitas/autogpt@sha256:' + ('b' * 64)) (Get-RunCall)
	Assert-NotContains 'docker.io/significantgravitas/autogpt:latest' (Get-RunCall)
	Assert-Contains '--publish 127.0.0.1:3000:3000' $log
	Assert-Contains '--restart unless-stopped' $log
	Assert-Contains '--stop-timeout 360' $log
	Assert-Contains '--shm-size 2g' $log
	Assert-Contains '--ulimit nofile=65536:65536' $log
	Assert-Contains '--volume autogpt-platform-data:/data' $log
	Assert-Contains '--label com.agpt.installer.owner=autogpt-platform-installer' $log
	Assert-Contains '--label com.agpt.installer.contract=1' $log
	if ((Get-RunCall) -cnotmatch '--label com\.agpt\.installer\.env-sha256=[0-9a-f]{64}') { throw "run call has no env-sha256 label:`n$(Get-RunCall)" }
	Assert-PrivateAcl $defaultDir
	foreach ($name in 'autogpt.env', 'install-id', 'install-state') { Assert-PrivateAcl (Join-Path $defaultDir $name) }
	# The user owns the state even after an elevated run, so a standard-user rerun accepts it.
	foreach ($path in @($defaultDir) + @('autogpt.env', 'install-id', 'install-state' | ForEach-Object { Join-Path $defaultDir $_ })) {
		$owner = (Get-Acl -LiteralPath $path).GetOwner([System.Security.Principal.SecurityIdentifier]).Value
		if ($owner -ne $userSid) { throw "$path is owned by $owner, not the current user" }
	}
	if ([System.IO.File]::ReadAllText((Join-Path $defaultDir 'autogpt.env')) -cne "# Private runtime configuration for the AutoGPT appliance.`nAUTOGPT_PUBLIC_URL=http://localhost:3000`n") {
		throw 'autogpt.env does not have the expected LF-terminated contents'
	}

	Clear-File $dockerLog
	Clear-File $volumeState
	$explicitDir = Join-Path $root 'explicit-install'
	Assert-Succeeded (Invoke-FakeInstall $explicitDir @{} @('-Release', 'v1.2.3')) 'explicit release'
	Assert-Contains 'pull docker.io/significantgravitas/autogpt:v1.2.3' (Get-DockerLog)

	$probeDir = Join-Path $root 'endpoint-probe'
	Assert-FailsWith 'Refusing non-local Docker endpoint: tcp://' (Invoke-FakeInstall $probeDir @{ DOCKER_HOST = 'tcp://localhost:2375' })
	Assert-FailsWith 'Refusing non-local Docker endpoint: ssh://' (Invoke-FakeInstall $probeDir @{ FAKE_DOCKER_ENDPOINT = 'ssh://example.invalid' })
	Assert-FailsWith 'Refusing non-local Docker endpoint: npipe:////remote-host/pipe/docker_engine' (Invoke-FakeInstall $probeDir @{ FAKE_DOCKER_ENDPOINT = 'npipe:////remote-host/pipe/docker_engine' })
	Assert-FailsWith 'Refusing non-local Docker endpoint: unix://' (Invoke-FakeInstall $probeDir @{ FAKE_DOCKER_ENDPOINT = 'unix:///var/run/docker.sock' })
	$result = Invoke-FakeInstall $probeDir @{ FAKE_DOCKER_ENDPOINT = 'npipe:////./pipe/docker_engine' } @('-PreflightOnly')
	Assert-Succeeded $result 'default Docker Desktop pipe'
	Assert-Contains 'nothing was installed' $result.Output
	Assert-Succeeded (Invoke-FakeInstall $probeDir @{ DOCKER_HOST = 'npipe:////./pipe/dockerDesktopLinuxEngine' } @('-PreflightOnly')) 'DOCKER_HOST named pipe'
	Assert-FailsWith 'switch Docker Desktop to Linux containers' (Invoke-FakeInstall $probeDir @{ FAKE_DOCKER_OS = 'windows' })
	Assert-FailsWith 'Unsupported Docker daemon architecture' (Invoke-FakeInstall $probeDir @{ FAKE_DOCKER_ARCH = 'riscv64' })
	Assert-FailsWith 'not running or is not accessible' (Invoke-FakeInstall $probeDir @{ FAKE_DOCKER_INFO_FAIL = 'true' })
	# CI images can ship a docker.exe in System32, so search only an empty folder.
	$emptyBin = Join-Path $root 'empty-bin'
	New-Item -ItemType Directory -Path $emptyBin | Out-Null
	Assert-FailsWith 'Docker is required' (Invoke-FakeInstall $probeDir @{ PATH = $emptyBin })
	$cmdBin = Join-Path $root 'cmd-bin'
	New-Item -ItemType Directory -Path $cmdBin | Out-Null
	Clear-File (Join-Path $cmdBin 'docker.cmd') '@exit /b 0'
	Assert-FailsWith 'must be docker.exe' (Invoke-FakeInstall $probeDir @{ PATH = "$cmdBin;$systemPath" })

	Assert-FailsWith 'unexpected OCI title' (Invoke-FakeInstall (Join-Path $root 'wrong-title') @{ FAKE_IMAGE_TITLE = 'Not AutoGPT' })
	Assert-FailsWith 'invalid source revision label' (Invoke-FakeInstall (Join-Path $root 'wrong-revision') @{ FAKE_IMAGE_REVISION = 'not-a-revision' })
	Assert-FailsWith 'exactly one native repository digest; found 0' (Invoke-FakeInstall (Join-Path $root 'missing-digest') @{ FAKE_IMAGE_DIGESTS = 'none' })
	$two = 'significantgravitas/autogpt@sha256:' + ('b' * 64) + '\nsignificantgravitas/autogpt@sha256:' + ('c' * 64) + '\n'
	Assert-FailsWith 'exactly one native repository digest; found 2' (Invoke-FakeInstall (Join-Path $root 'multiple-digest') @{ FAKE_IMAGE_DIGESTS = $two })
	Assert-FailsWith 'unexpected repository digest' (Invoke-FakeInstall (Join-Path $root 'foreign-digest') @{ FAKE_IMAGE_DIGESTS = ('example/other@sha256:' + ('b' * 64) + '\n') })

	Clear-File $volumeState
	$aliasedDir = Join-Path $root 'aliased-digest'
	$aliased = 'significantgravitas/autogpt@sha256:' + ('b' * 64) + '\ndocker.io/significantgravitas/autogpt@sha256:' + ('b' * 64) + '\n'
	Assert-Succeeded (Invoke-FakeInstall $aliasedDir @{ FAKE_IMAGE_DIGESTS = $aliased }) 'aliased digests'
	$installId = [System.IO.File]::ReadAllText((Join-Path $aliasedDir 'install-id'))
	if ([System.IO.File]::ReadAllText((Join-Path $aliasedDir 'install-state')) -cne "contract=1 lifecycle=established volume=autogpt-platform-data install-id=$installId`n") {
		throw 'install-state does not record the established volume'
	}

	Assert-FailsWith 'does not match Docker daemon architecture' (Invoke-FakeInstall (Join-Path $root 'wrong-image-arch') @{ FAKE_IMAGE_ARCH = 'arm64' })
	Assert-FailsWith 'installer ownership or configuration identity does not match' (Invoke-FakeInstall (Join-Path $root 'container-collision') @{ FAKE_CONTAINER_EXISTS = 'true'; FAKE_CONTAINER_LABELS = 'unowned 0 wrong wrong' })

	# Reruns against the default install.
	Clear-File $dockerLog
	Clear-File $volumeState "owned`n"
	$result = Invoke-FakeInstall $defaultDir @{ FAKE_CONTAINER_EXISTS = 'true' }
	Assert-Succeeded $result 'exact-contract rerun'
	Assert-Contains 'already matches the requested immutable contract' $result.Output
	Assert-NotContains 'run --detach' (Get-DockerLog)
	Assert-Contains 'container inspect --format' (Get-DockerLog)

	Clear-File $dockerLog
	Assert-Succeeded (Invoke-FakeInstall $defaultDir @{ FAKE_CONTAINER_EXISTS = 'true'; FAKE_CONTAINER_STATE = 'exited' }) 'start an exited container'
	Assert-Contains 'start autogpt' (Get-DockerLog)

	Clear-File $dockerLog
	Assert-Succeeded (Invoke-FakeInstall $defaultDir @{ FAKE_CONTAINER_EXISTS = 'true'; FAKE_CONTAINER_HEALTH = 'starting' }) 'container still starting'
	Assert-NotContains 'run --detach' (Get-DockerLog)
	Assert-Contains 'inspect --format' (Get-DockerLog)

	Assert-FailsWith 'running but not healthy (health=unhealthy)' (Invoke-FakeInstall $defaultDir @{ FAKE_CONTAINER_EXISTS = 'true'; FAKE_CONTAINER_HEALTH = 'unhealthy' })
	Assert-FailsWith 'immutable appliance digest does not match' (Invoke-FakeInstall $defaultDir @{ FAKE_CONTAINER_EXISTS = 'true'; FAKE_CONTAINER_IMAGE = 'docker.io/significantgravitas/autogpt:latest' })
	Assert-FailsWith 'entrypoint or command was overridden' (Invoke-FakeInstall $defaultDir @{ FAKE_CONTAINER_EXISTS = 'true'; FAKE_CONTAINER_PROCESS = '["/bin/sh"]|null' })

	Clear-File $volumeState
	Assert-FailsWith 'Could not inspect appliance container autogpt; it may have been removed' (Invoke-FakeInstall (Join-Path $root 'health-inspect-failure') @{ FAKE_HEALTH_INSPECT_FAIL = 'true' })
	Clear-File $volumeState
	Assert-FailsWith 'failed while starting (state=running, health=unhealthy)' (Invoke-FakeInstall (Join-Path $root 'health-unhealthy') @{ FAKE_HEALTH_STATUS = 'running unhealthy' })
	Clear-File $volumeState
	Assert-FailsWith 'failed while starting (state=exited, health=none)' (Invoke-FakeInstall (Join-Path $root 'health-exited') @{ FAKE_HEALTH_STATUS = 'exited' })

	$driftDir = Join-Path $root 'runtime-drift'
	New-Item -ItemType Directory -Path $driftDir | Out-Null
	Clear-File (Join-Path $driftDir 'install-id') ('a' * 32)
	Clear-File (Join-Path $driftDir 'autogpt.env') "AUTOGPT_PUBLIC_URL=http://localhost:3000`n"
	Assert-FailsWith 'critical runtime contract has drifted' (Invoke-FakeInstall $driftDir @{ FAKE_CONTAINER_EXISTS = 'true'; FAKE_CONTAINER_RUNTIME = 'true|host|no|0|0|none||||0||0||0|' })

	$missingVolumeDir = Join-Path $root 'missing-prior-volume'
	New-Item -ItemType Directory -Path $missingVolumeDir | Out-Null
	Clear-File (Join-Path $missingVolumeDir 'install-id') ('a' * 32)
	Clear-File (Join-Path $missingVolumeDir 'autogpt.env') "AUTOGPT_PUBLIC_URL=http://localhost:3000`n"
	Clear-File (Join-Path $missingVolumeDir 'install-state') ('contract=1 lifecycle=established volume=autogpt-platform-data install-id=' + ('a' * 32) + "`n")
	Clear-File $volumeState
	Assert-FailsWith 'Installer state already exists but persistent volume autogpt-platform-data is missing' (Invoke-FakeInstall $missingVolumeDir)

	$retryDir = Join-Path $root 'retry-after-pull'
	Clear-File $volumeState
	Assert-FailsWith 'Published appliance image' (Invoke-FakeInstall $retryDir @{ FAKE_DOCKER_PULL_FAIL = 'true' })
	if (Test-Path -LiteralPath (Join-Path $retryDir 'install-state')) { throw 'a failed pull recorded an established install' }
	Assert-Succeeded (Invoke-FakeInstall $retryDir) 'retry after a failed pull'
	if (-not (Test-Path -LiteralPath (Join-Path $retryDir 'install-state'))) { throw 'the retried install did not record its volume' }
	Clear-File $volumeState
	Assert-FailsWith 'Installer state already exists but persistent volume autogpt-platform-data is missing' (Invoke-FakeInstall $retryDir)

	$runFailureDir = Join-Path $root 'run-failure'
	Clear-File $volumeState
	Assert-FailsWith 'Could not start the appliance' (Invoke-FakeInstall $runFailureDir @{ FAKE_DOCKER_RUN_FAIL = 'true' })
	if (-not (Test-Path -LiteralPath (Join-Path $runFailureDir 'install-state'))) { throw 'a failed run lost the established volume record' }
	Clear-File $volumeState
	Assert-FailsWith 'Installer state already exists but persistent volume autogpt-platform-data is missing' (Invoke-FakeInstall $runFailureDir)

	Clear-File $volumeState "unowned`n"
	Assert-FailsWith 'Refusing existing unowned or incompatible volume' (Invoke-FakeInstall (Join-Path $root 'unowned-volume'))

	Clear-File $volumeState
	$symlinkEnvDir = Join-Path $root 'symlink-env'
	New-Item -ItemType Directory -Path $symlinkEnvDir | Out-Null
	$outsideEnv = Join-Path $root 'outside.env'
	Clear-File $outsideEnv "AUTOGPT_PUBLIC_URL=http://localhost:3000`n"
	if (New-FileSymlink (Join-Path $symlinkEnvDir 'autogpt.env') $outsideEnv) {
		Assert-FailsWith 'Refusing symlink for Runtime configuration' (Invoke-FakeInstall $symlinkEnvDir)
	} else {
		Write-Host 'skipped: this account cannot create file symlinks (needs Developer Mode or administrator)'
	}

	$hardlinkEnvDir = Join-Path $root 'hardlink-env'
	New-Item -ItemType Directory -Path $hardlinkEnvDir | Out-Null
	Clear-File (Join-Path $hardlinkEnvDir 'autogpt.env') "AUTOGPT_PUBLIC_URL=http://localhost:3000`n"
	New-Item -ItemType HardLink -Path (Join-Path $root 'env-copy') -Target (Join-Path $hardlinkEnvDir 'autogpt.env') | Out-Null
	Assert-FailsWith 'Runtime configuration has multiple hard links' (Invoke-FakeInstall $hardlinkEnvDir)

	$hardlinkStateDir = Join-Path $root 'hardlink-state'
	New-Item -ItemType Directory -Path $hardlinkStateDir | Out-Null
	Clear-File (Join-Path $hardlinkStateDir 'install-id') ('a' * 32)
	New-Item -ItemType HardLink -Path (Join-Path $root 'state-copy') -Target (Join-Path $hardlinkStateDir 'install-id') | Out-Null
	Assert-FailsWith 'Installer identity has multiple hard links' (Invoke-FakeInstall $hardlinkStateDir)

	$badIdDir = Join-Path $root 'bad-install-id'
	New-Item -ItemType Directory -Path $badIdDir | Out-Null
	Clear-File (Join-Path $badIdDir 'install-id') 'not-an-id'
	Assert-FailsWith 'Installer identity has invalid contents' (Invoke-FakeInstall $badIdDir)

	# The bootstrap command from docs/platform/installer.md must not run a
	# failed download, and must run a successful one from a unique temp file.
	$doc = [System.IO.File]::ReadAllText((Join-Path $repoRoot 'docs\platform\installer.md'))
	$section = [regex]::Match($doc, '(?ms)^### Windows\r?\n.*?^```powershell\r?\n(.*?)^```')
	if (-not $section.Success) { throw 'docs/platform/installer.md has no "### Windows" section with a powershell block' }
	$bootstrapFile = Join-Path $root 'bootstrap.ps1'
	[System.IO.File]::WriteAllText($bootstrapFile, $section.Groups[1].Value)
	$bootstrapTemp = Join-Path $root 'bootstrap-temp'
	New-Item -ItemType Directory -Path $bootstrapTemp | Out-Null
	$curlLog = Join-Path $root 'curl.log'
	$powerShellLog = Join-Path $root 'powershell.log'
	$bootstrapEnvironment = @{
		PATH = "$fakeBin;$systemPath"
		TEMP = $bootstrapTemp
		TMP = $bootstrapTemp
		FAKE_CURL_LOG = $curlLog
		FAKE_POWERSHELL_LOG = $powerShellLog
		FAKE_CURL_MODE = 'fail'
	}
	[void](Invoke-Program $powerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $bootstrapFile) $bootstrapEnvironment)
	if (Test-Path -LiteralPath $powerShellLog) { throw 'the documented bootstrap ran a partial download' }
	if (@(Get-ChildItem -LiteralPath $bootstrapTemp -Force).Count -ne 0) { throw 'the documented bootstrap left its partial download behind' }
	$curlArgs = ConvertFrom-ArgumentLog ([System.IO.File]::ReadAllLines($curlLog)[0])
	$installerPath = $curlArgs[7]
	if (($curlArgs[0..6] -join ' ') -cne '--proto =https --proto-redir =https --tlsv1.2 -fsSL -o' -or $curlArgs[8] -cne 'https://setup.agpt.co/install.ps1' -or $curlArgs.Count -ne 9) {
		throw "unexpected bootstrap curl arguments: $($curlArgs -join ' ')"
	}
	if ($installerPath -notmatch ('^' + [regex]::Escape($bootstrapTemp) + '\\autogpt-install-[0-9a-f]{32}\.ps1$')) { throw "unexpected bootstrap temp file: $installerPath" }

	$bootstrapEnvironment['FAKE_CURL_MODE'] = 'ok'
	[void](Invoke-Program $powerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $bootstrapFile) $bootstrapEnvironment)
	$ran = [System.IO.File]::ReadAllLines($powerShellLog)
	$ranArgs = ConvertFrom-ArgumentLog $ran[0]
	if (($ranArgs[0..3] -join ' ') -cne '-NoProfile -ExecutionPolicy Bypass -File' -or $ranArgs.Count -ne 5 -or $ran[1] -cne 'target-existed') {
		throw "the documented bootstrap did not run the downloaded installer as expected: $($ran -join ' | ')"
	}
	if (@(Get-ChildItem -LiteralPath $bootstrapTemp -Force).Count -ne 0) { throw 'the documented bootstrap left the installer behind' }

	# install.bat runs install.ps1 the same way and passes its arguments on.
	$shimCopy = Join-Path $root 'install.bat'
	Copy-Item -LiteralPath $shim -Destination $shimCopy
	$shimTemp = Join-Path $root 'shim-temp'
	New-Item -ItemType Directory -Path $shimTemp | Out-Null
	Remove-Item -LiteralPath $curlLog, $powerShellLog -Force
	$shimEnvironment = @{
		PATH = "$fakeBin;$systemPath"
		TEMP = $shimTemp
		TMP = $shimTemp
		FAKE_CURL_LOG = $curlLog
		FAKE_POWERSHELL_LOG = $powerShellLog
		FAKE_CURL_MODE = 'fail'
	}
	$result = Invoke-Program "$env:SystemRoot\System32\cmd.exe" @('/d', '/c', $shimCopy, '-Release', 'v1.2.3') $shimEnvironment
	if ($result.ExitCode -ne 1) { throw "install.bat returned $($result.ExitCode) after a failed download" }
	if (Test-Path -LiteralPath $powerShellLog) { throw 'install.bat ran a partial download' }
	if (@(Get-ChildItem -LiteralPath $shimTemp -Force).Count -ne 0) { throw 'install.bat left its partial download behind' }
	$curlArgs = ConvertFrom-ArgumentLog ([System.IO.File]::ReadAllLines($curlLog)[0])
	if (($curlArgs[0..6] -join ' ') -cne '--proto =https --proto-redir =https --tlsv1.2 -fsSL -o' -or $curlArgs[8] -cne 'https://setup.agpt.co/install.ps1' -or $curlArgs.Count -ne 9) {
		throw "unexpected install.bat curl arguments: $($curlArgs -join ' ')"
	}

	$shimEnvironment['FAKE_CURL_MODE'] = 'ok'
	$shimEnvironment['FAKE_POWERSHELL_EXIT'] = '3'
	$result = Invoke-Program "$env:SystemRoot\System32\cmd.exe" @('/d', '/c', $shimCopy, '-Release', 'v1.2.3') $shimEnvironment
	if ($result.ExitCode -ne 3) { throw "install.bat did not pass on the installer's exit code: $($result.ExitCode)" }
	$ran = [System.IO.File]::ReadAllLines($powerShellLog)
	$ranArgs = ConvertFrom-ArgumentLog $ran[0]
	if (($ranArgs[0..3] -join ' ') -cne '-NoProfile -ExecutionPolicy Bypass -File' -or ($ranArgs[5..6] -join ' ') -cne '-Release v1.2.3' -or $ranArgs.Count -ne 7 -or $ran[1] -cne 'target-existed') {
		throw "install.bat did not run the downloaded installer as expected: $($ran -join ' | ')"
	}
	if ($ranArgs[4] -notmatch ('^' + [regex]::Escape($shimTemp) + '\\autogpt-install-[0-9]+\.ps1$')) { throw "unexpected install.bat temp file: $($ranArgs[4])" }
	if (@(Get-ChildItem -LiteralPath $shimTemp -Force).Count -ne 0) { throw 'install.bat left the installer behind' }

	Write-Host "installer PowerShell tests passed ($($PSVersionTable.PSEdition) $($PSVersionTable.PSVersion))"
} finally {
	foreach ($junction in $junctions) { [System.IO.Directory]::Delete($junction) }
	Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
