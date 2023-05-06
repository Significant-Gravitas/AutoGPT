# powershell color library (hash lookup table)
$global:pscolors = @{
    NC = "`e[0m";
    B = "`e[0;30m"; R = "`e[0;31m"; G = "`e[0;32m"; Y = "`e[0;33m"; BL = "`e[0;34m"; P = "`e[0;35m"; C = "`e[0;36m"; W = "`e[0;37m";
    BB = "`e[1;30m"; BR = "`e[1;31m"; BG = "`e[1;32m"; BY = "`e[1;33m"; BBL = "`e[1;34m"; BP = "`e[1;35m"; BC = "`e[1;36m"; BW = "`e[1;37m";
    UB = "`e[4;30m"; UR = "`e[4;31m"; UG = "`e[4;32m"; UY = "`e[4;33m"; UBL = "`e[4;34m"; UP = "`e[4;35m"; UC = "`e[4;36m"; UW = "`e[4;37m";
    OB = "`e[40m"; OR = "`e[41m"; OG = "`e[42m"; OY = "`e[43m"; OBL = "`e[44m"; OP = "`e[45m"; OC = "`e[46m"; OW = "`e[47m";
    IB = "`e[0;90m"; IR = "`e[0;91m"; IG = "`e[0;92m"; IY = "`e[0;93m"; IBL = "`e[0;94m"; IP = "`e[0;95m"; IC = "`e[0;96m"; IW = "`e[0;97m";
    BIB = "`e[1;90m"; BIR = "`e[1;91m"; BIG = "`e[1;92m"; BIY = "`e[1;93m"; BIBL = "`e[1;94m"; BIP = "`e[1;95m"; BIC = "`e[1;96m"; BIW = "`e[1;97m";
    OIB = "`e[0;100m"; OIR = "`e[0;101m"; OIG = "`e[0;102m"; OIY = "`e[0;103m"; OIBL = "`e[0;104m"; OIP = "`e[0;105m"; OIC = "`e[0;106m"; OIW = "`e[0;107m"
}

# Usage instructions
function Display-Usage() {
    [CmdletBinding()]
    param()
    Write-Host "Usage: $args[0] [-h] [-v] [-e VENV_PATH] [-c] [-s] [-C AI_SETTINGS] [-l CONTINUOUS_LIMIT] [-S] [-d] [-g] [-G] [-m MEMORY_TYPE] [-b BROWSER_NAME] [-A] [-N] [-w WORKSPACE_DIR] [-I] [args]"
    Write-Host ""
    Write-Host "${COLOR_Y}Command-line Options:${COLOR_NC}"
    Write-Host ""
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-h, --help".PadRight(40), $global:pscolors.NC, "Show this help message and exit.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-v, --virtualenv".PadRight(40), $global:pscolors.NC, "Use Python virtual environment. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-e, --venv-path VENV_PATH".PadRight(40), $global:pscolors.NC, "Set the path to the Python virtual environment.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-c, --continuous".PadRight(40), $global:pscolors.NC, "Enable continuous mode for autogpt. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-s, --skip-reprompt".PadRight(40), $global:pscolors.NC, "Skip the re-prompting messages at the beginning of the script.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-C, --ai-settings AI_SETTINGS".PadRight(40), $global:pscolors.NC, "Specify which ai_settings.yaml file to use.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-l, --continuous-limit CONTINUOUS_LIMIT".PadRight(40), $global:pscolors.NC, "Define the number of times to run in continuous mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-S, --speak".PadRight(40), $global:pscolors.NC, "Enable Speak Mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-d, --debug".PadRight(40), $global:pscolors.NC, "Enable Debug Mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-g, --gpt3only".PadRight(40), $global:pscolors.NC, "Enable GPT3.5 Only Mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-G, --gpt4only".PadRight(40), $global:pscolors.NC, "Enable GPT4 Only Mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-m, --use-memory MEMORY_TYPE".PadRight(40), $global:pscolors.NC, "Define which Memory backend to use.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-b, --browser-name BROWSER_NAME".PadRight(40), $global:pscolors.NC, "Specify which web-browser to use when using selenium to scrape the web.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-A, --allow-downloads".PadRight(40), $global:pscolors.NC, "Set to true to allow Auto-GPT to download files natively. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-N, --skip-news".PadRight(40), $global:pscolors.NC, "Set to true to suppress the output of latest news on startup. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-w, --workspace-dir WORKSPACE_DIR".PadRight(40), $global:pscolors.NC, "Specifies which directory to use for the workspace.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "-I, --install-plugin-deps".PadRight(40), $global:pscolors.NC, "Set to true to install external dependencies for 3rd party plugins. (default: false)")  
    Write-Host ""
    Write-Host "${COLOR_Y}Environment Variables:${COLOR_NC}"
    Write-Host ""
    Write-Host $("  {0}{1}{2}" -f $global:pscolors.BL, "You can set these values instead of passing command line arguments,", $global:pscolors.NC)
    Write-Host $("  {0}{1}{2}" -f $global:pscolors.BL, "before executing this script.", $global:pscolors.NC)
    Write-Host ""
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C,  "USE_VENV".PadRight(30), $global:pscolors.NC, "Set to true to use Python virtual environment. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "VENV_PATH".PadRight(30), $global:pscolors.NC, "Set the path to the Python virtual environment.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "CONTINUOUS".PadRight(30), $global:pscolors.NC, "Set to true to enable continuous mode for autogpt. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "SKIP_REPROMPT".PadRight(30), $global:pscolors.NC, "Set to true to skip the re-prompting messages at the beginning of the script.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "AI_SETTINGS".PadRight(30), $global:pscolors.NC, "Specify which ai_settings.yaml file to use.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "CONTINUOUS_LIMIT".PadRight(30), $global:pscolors.NC, "Define the number of times to run in continuous mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "SPEAK".PadRight(30), $global:pscolors.NC, "Set to true to enable Speak Mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "DEBUG".PadRight(30), $global:pscolors.NC, "Set to true to enable Debug Mode.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "GPT3_ONLY".PadRight(30), $global:pscolors.NC, "Set to true to use GPT-3 API only for autogpt. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "GPT4_ONLY".PadRight(30), $global:pscolors.NC, "Set to true to use GPT-4 API only for autogpt. (default: false)")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "MEMORY_TYPE".PadRight(30), $global:pscolors.NC, "Set to define which Memory backend to use.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "BROWSER_NAME".PadRight(30), $global:pscolors.NC, "Set to define which web-browser to use when using selenium to scrape the web.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "BROWSER_NAME".PadRight(30), $global:pscolors.NC, "Set to define which web-browser to use when using selenium to scrape the web.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "ALLOW_DOWNLOADS".PadRight(30), $global:pscolors.NC, "Set to true to allow Auto-GPT to download files natively.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "SKIP_NEWS".PadRight(30), $global:pscolors.NC, "Set to true to suppress the output of latest news on startup.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "WORKSPACE_DIR".PadRight(30), $global:pscolors.NC, "Specifies which directory to use for the workspace.")
    Write-Host $("  {0}{1}{2}{3}" -f $global:pscolors.C, "INSTALL_PLUGIN_DEPS".PadRight(30), $global:pscolors.NC, "Set to true to install external dependencies for 3rd party plugins.")
    Exit 1
}

# Parse Command-line arguments
function Parse-Arguments() {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true, Position=0)][string[]]$args = @()
    )

    $params = Get-DefaultParams

    while ($args) {
        switch -regex ($args[0]) {
            "-h|--help" { Display-Usage; return }
            "-v|--virtualenv" { $params.USE_VENV = $true; break }
            "-e|--venv-path" { $params.VENV_PATH = $args[1]; $args = $args[2..($args.Length - 1)]; break }
            "-c|--continuous" { $params.CONTINUOUS = $true; break }
            "-s|--skip-reprompt" { $params.SKIP_REPROMPT = $true; break }
            "-C|--ai-settings" { $params.AI_SETTINGS = $args[1]; $args = $args[2..($args.Length - 1)]; break }
            "-l|--continuous-limit" { $params.CONTINUOUS_LIMIT = $args[1]; $args = $args[2..($args.Length - 1)]; break }
            "-S|--speak" { $params.SPEAK = $true; break }
            "-d|--debug" { $params.DEBUG = $true; break }
            "-g|--gpt3only" { $params.GPT3_ONLY = $true; break }
            "-G|--gpt4only" { $params.GPT4_ONLY = $true; break }
            "-m|--use-memory" { $params.MEMORY_TYPE = $args[1]; $args = $args[2..($args.Length - 1)]; break }
            "-b|--browser-name" { $params.BROWSER_NAME = $args[1]; $args = $args[2..($args.Length - 1)]; break }
            "-A|--allow-downloads" { $params.ALLOW_DOWNLOADS = $true; break }
            "-N|--skip-news" { $params.SKIP_NEWS = $true; break }
            "-w|--workspace-directory" { $params.WORKSPACE_DIR = $args[1]; $args = $args[2..($args.Length - 1)]; break }
            "-I|--install-plugin-deps" { $params.INSTALL_PLUGIN_DEPS = $true; break }
            default { break }
        }
        $args = $args[1..($args.Length - 1)]
    }
    return $params
}

function Get-DefaultParams() {
    [CmdletBinding()]
    param()
    $params = @{
        USE_VENV = $env:USE_VENV -eq 'true'
        VENV_PATH = $env:VENV_PATH
        CONTINUOUS = $env:CONTINUOUS -eq 'true'
        SKIP_REPROMPT = $env:SKIP_REPROMPT -eq 'true'
        AI_SETTINGS = $env:AI_SETTINGS
        CONTINUOUS_LIMIT = $env:CONTINUOUS_LIMIT
        SPEAK = $env:SPEAK -eq 'true'
        DEBUG = $env:DEBUG -eq 'true'
        GPT3_ONLY = $env:GPT3_ONLY -eq 'true'
        GPT4_ONLY = $env:GPT4_ONLY -eq 'true'
        MEMORY_TYPE = $env:MEMORY_TYPE
        BROWSER_NAME = $env:BROWSER_NAME
        ALLOW_DOWNLOADS = $env:ALLOW_DOWNLOADS -eq 'true'
        SKIP_NEWS = $env:SKIP_NEWS -eq 'true'
        WORKSPACE_DIR = $env:WORKSPACE_DIR
        INSTALL_PLUGIN_DEPS = $env:INSTALL_PLUGIN_DEPS -eq 'true'
    }
    return $params
}

function Build-PyCommand() {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true, Position=0)][string]$pythonCmd,
        [Parameter(Mandatory = $true, Position=1)][string[]]$args
    )
    $command = "$pythonCmd -m autogpt"
    if ($args.CONTINUOUS) { $command += " --continuous" }
    if ($args.GPT3_ONLY) { $command += " --gpt3only" }
    if ($args.GPT4_ONLY) { $command += " --gpt4only" }
    if ($args.SKIP_REPROMPT) { $command += " --skip-reprompt" }
    if ($args.AI_SETTINGS) { $command += " --ai-settings $($args.AI_SETTINGS)" }
    if ($args.CONTINUOUS_LIMIT) { $command += " --continuous-limit $($args.CONTINUOUS_LIMIT)" }
    if ($args.SPEAK) { $command += " --speak" }
    if ($args.DEBUG) { $command += " --debug" }
    if ($args.MEMORY_TYPE) { $command += " --use-memory $($args.MEMORY_TYPE)" }
    if ($args.BROWSER_NAME) { $command += " --browser-name $($args.BROWSER_NAME)" }
    if ($args.ALLOW_DOWNLOADS) { $command += " --allow-downloads" }
    if ($args.SKIP_NEWS) { $command += " --skip-news" }
    if ($args.WORKSPACE_DIR) { $command += " --workspace-directory $($args.WORKSPACE_DIR)" }
    if ($args.INSTALL_PLUGIN_DEPS) { $command += " --install-plugin-deps" }
    $command += " $($args['--'])"
    return $command
}

# Null-fix args
if ($args) {
    $params = Parse-Arguments $args
} else {
    $params = Get-DefaultParams
}

# Use virtual environment if requested
if ($params.USE_VENV -and [string]::IsNullOrEmpty($params.VENV_PATH)) {
    Write-Host "Using Python virtual environment: $($params.VENV_PATH)"
    . "$($params.VENV_PATH)/bin/activate"
}

# Check missing requirements
$pythonCmd = (Get-Command python3 -ErrorAction SilentlyContinue).Source ?? (Get-Command python -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrEmpty($pythonCmd)) {
    Write-Error "Python not found. Please install Python."
    return
}

# Install missing requirements
$requirementsFile = Join-Path $PSScriptRoot "requirements.txt"
$checkResult = & $pythonCmd (Join-Path $PSScriptRoot "scripts/check_requirements.py") $requirementsFile
if ($checkResult.ExitCode -eq 1) {
    Write-Host "Installing missing packages..."
    & $pythonCmd -m pip install -r $requirementsFile
}

# Run the command
$command = Build-PyCommand $pythonCmd $params
Invoke-Expression $command

# Deactivate virtual environment if used
if ($params.USE_VENV -and [string]::IsNullOrEmpty($params.VENV_PATH)) {
    Write-Host "Deactivating Python virtual environment: $($params.VENV_PATH)"
    deactivate
}

Read-Host "Press Enter to continue..."
