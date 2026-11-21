# Proves that the installed app never opens a socket the Windows firewall
# would ask the user about.
#
#   firewall.ps1 -Phase begin -InstallDir <dir> -StatePath <file> -NodePath <node.exe>
#   firewall.ps1 -Phase check -InstallDir <dir> -StatePath <file>
#
# `begin` CHANGES THIS MACHINE: it turns the firewall and its notifications
# on for every profile, enlarges the Security log and turns on auditing of
# "Filtering Platform Connection". It is meant for a CI machine that is
# thrown away. It needs an administrator.
#
# Why the audit log and not a list of listening sockets: the sockets that
# caused the prompt lived for microseconds inside the Erlang VM
# (build/erlang_patches.py). A sample never sees them. The audit log records
# every listen (event 5154) and every bind (5158) with the program's path and
# the local address.
#
# `check` prints one JSON object: { "violations": [...], "notes": [...] }.

param(
    [Parameter(Mandatory = $true)][ValidateSet('begin', 'check')][string]$Phase,
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$StatePath,
    [string]$NodePath
)

$ErrorActionPreference = 'Stop'
$FirewallLog = 'Microsoft-Windows-Windows Firewall With Advanced Security/Firewall'
# "Filtering Platform Connection". Its display name is translated; this is not.
$ConnectionAudit = '{0CCE9226-69AE-11D9-BED3-505054503030}'
# Rule added, rule changed, rule deleted, and "blocked, could not notify the user".
$FirewallEventIds = 2004, 2005, 2006, 2011

function Test-Loopback([string]$Address) {
    return $Address -eq '::1' -or $Address.StartsWith('127.')
}

# Audit events name a program by device path (\device\harddiskvolume3\users\...),
# so programs are matched on the part after the drive letter.
function Get-PathTail([string]$Path) {
    return ([IO.Path]::GetFullPath($Path)).Substring(2).TrimEnd('\').ToLowerInvariant()
}

function Get-SocketEvents([datetime]$Since) {
    $found = @(Get-WinEvent -FilterHashtable @{ LogName = 'Security'; Id = 5154, 5158; StartTime = $Since } -ErrorAction SilentlyContinue)
    foreach ($entry in $found) {
        $data = @{}
        foreach ($item in ([xml]$entry.ToXml()).Event.EventData.Data) { $data[$item.Name] = $item.'#text' }
        [pscustomobject]@{
            Id          = $entry.Id
            Application = ([string]$data['Application']).ToLowerInvariant()
            Address     = [string]$data['SourceAddress']
            Port        = [string]$data['SourcePort']
            Protocol    = [string]$data['Protocol']
        }
    }
}

function Get-RulesNaming([string]$Directory) {
    return @(Get-NetFirewallApplicationFilter -All |
            Where-Object { $_.Program -like "$Directory*" } |
            ForEach-Object { $_.InstanceID })
}

# A program nobody has made a rule for listens on every interface, connects
# out and sends a datagram. The listen has to show up in the audit log, or
# nothing this script concludes afterwards means anything. What its outgoing
# sockets log decides whether bind events can be told from listeners here.
function Invoke-Control([string]$Node) {
    $directory = Join-Path ([IO.Path]::GetTempPath()) ('autogpt-e2e-control-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $directory | Out-Null
    try {
        $program = Join-Path $directory 'node.exe'
        Copy-Item -LiteralPath $Node -Destination $program
        $script = Join-Path $directory 'control.js'
        Set-Content -LiteralPath $script -Encoding ASCII -Value @'
const net = require('net');
const dgram = require('dgram');
const server = net.createServer((socket) => socket.end());
server.listen(0, '0.0.0.0', () => {
  const port = server.address().port;
  console.log(port);
  net.connect(port, '127.0.0.1').on('error', () => {});
  const datagram = dgram.createSocket('udp4');
  datagram.send('x', port, '127.0.0.1', () => datagram.close());
  setTimeout(() => process.exit(0), 3000);
});
'@
        $since = Get-Date
        $port = [string](& $program $script)
        # The folder's own name, not the whole path: the temporary directory
        # is given in its short form on some machines (C:\Users\RUNNER~1\...
        # on GitHub's), and the audit log names programs by their long path.
        $tail = ('\' + (Split-Path -Leaf $directory) + '\node.exe').ToLowerInvariant()
        $deadline = (Get-Date).AddSeconds(30)
        do {
            Start-Sleep -Seconds 2
            $seen = @(Get-SocketEvents $since)
            $events = @($seen | Where-Object { $_.Application.EndsWith($tail) })
            # The program listens once, and the folder it runs from is its
            # alone, so any listen logged for it is that one. The port is
            # taken from the event: what the program printed did not compare
            # equal to it on GitHub's machines.
            $listens = @($events | Where-Object { $_.Id -eq 5154 })
        } until ($listens.Count -gt 0 -or (Get-Date) -gt $deadline)
        if ($listens.Count -eq 0) {
            # What was logged instead, so that the next failure explains itself.
            $programs = @($seen | ForEach-Object { "$($_.Id) $($_.Application) $($_.Address):$($_.Port)" } | Sort-Object -Unique | Select-Object -First 12)
            $policy = (& auditpol.exe /get "/subcategory:$ConnectionAudit" /r | Select-Object -Last 1)
            throw ("firewall instrumentation is not working on this machine: a program listening on 0.0.0.0:$port " +
                "($program) produced no audit event 5154. Socket events since it started: $($seen.Count)" +
                $(if ($programs.Count) { ", from: " + ($programs -join '; ') } else { '' }) +
                ". Audit policy: $policy")
        }
        $listenPorts = @($listens | ForEach-Object { $_.Port })
        $outgoing = @($events | Where-Object { $_.Id -eq 5158 -and $listenPorts -notcontains $_.Port -and -not (Test-Loopback $_.Address) })
        $logged = @($events | ForEach-Object { "$($_.Id) $($_.Address):$($_.Port)/$($_.Protocol)" } | Sort-Object -Unique)
        return @{
            bindDiscriminates = $outgoing.Count -eq 0
            control           = "listen logged on $($listenPorts -join ',') (the program printed '$port'); $($outgoing.Count) bind event(s) with a non-loopback address from outgoing sockets; all of its events: $($logged -join ' ')"
        }
    }
    finally {
        Remove-Item -LiteralPath $directory -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Start-Check {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'the firewall check needs an administrator'
    }
    if (-not $NodePath) { throw '-NodePath is required for -Phase begin' }

    Set-NetFirewallProfile -All -Enabled True -NotifyOnListen True
    & wevtutil.exe sl Security /ms:268435456
    if ($LASTEXITCODE -ne 0) { throw "wevtutil exited with $LASTEXITCODE" }
    & auditpol.exe /set "/subcategory:$ConnectionAudit" /success:enable /failure:enable | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "auditpol exited with $LASTEXITCODE" }

    $since = Get-Date
    $control = Invoke-Control $NodePath
    $state = @{
        since             = $since.ToString('o')
        bindDiscriminates = $control.bindDiscriminates
        control           = $control.control
        rulesBefore       = [string[]](Get-RulesNaming $InstallDir)
        profiles          = [string[]](Get-NetFirewallProfile | ForEach-Object { "$($_.Name): enabled=$($_.Enabled) notifyOnListen=$($_.NotifyOnListen) inbound=$($_.DefaultInboundAction)" })
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
    $state | ConvertTo-Json
}

function Complete-Check {
    $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    $since = [datetime]::Parse($state.since, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::RoundtripKind)
    $tail = (Get-PathTail $InstallDir) + '\'
    $violations = New-Object System.Collections.Generic.List[string]
    $notes = New-Object System.Collections.Generic.List[string]
    $notes.Add("control: $($state.control)")

    $exposed = @(Get-SocketEvents $since | Where-Object { $_.Application.Contains($tail) -and -not (Test-Loopback $_.Address) })
    foreach ($entry in @($exposed | Where-Object { $_.Id -eq 5154 })) {
        $violations.Add("listened on $($entry.Address):$($entry.Port) (protocol $($entry.Protocol)): $($entry.Application)")
    }
    $binds = @($exposed | Where-Object { $_.Id -eq 5158 })
    if ($state.bindDiscriminates) {
        foreach ($entry in $binds) {
            $violations.Add("bound $($entry.Address):$($entry.Port) (protocol $($entry.Protocol)): $($entry.Application)")
        }
    }
    elseif ($binds.Count -gt 0) {
        # On this image an ordinary outgoing connection logs the same bind
        # event, so it cannot be held against the app.
        $notes.Add("$($binds.Count) bind event(s) with a non-loopback address not counted: outgoing sockets log the same event here")
    }

    $needle = $InstallDir.ToLowerInvariant()
    $logged = @(Get-WinEvent -FilterHashtable @{ LogName = $FirewallLog; Id = $FirewallEventIds; StartTime = $since } -ErrorAction SilentlyContinue)
    foreach ($entry in $logged) {
        if ($entry.ToXml().ToLowerInvariant().Contains($needle)) {
            $violations.Add("firewall event $($entry.Id): $(($entry.Message -split "`n")[0].Trim())")
        }
    }

    foreach ($rule in (Get-RulesNaming $InstallDir)) {
        if ($state.rulesBefore -notcontains $rule) { $violations.Add("a firewall rule now names the app: $rule") }
    }

    @{
        violations = [string[]]@($violations | Select-Object -Unique -First 50)
        notes      = [string[]]$notes
    } | ConvertTo-Json
}

if ($Phase -eq 'begin') { Start-Check } else { Complete-Check }
