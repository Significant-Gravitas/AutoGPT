# Download LanguageTool
Invoke-WebRequest -Uri "https://languagetool.org/download/LanguageTool-5.5.zip" -OutFile "$env:USERPROFILE\Downloads\LanguageTool-5.5.zip"

# Extract LanguageTool
Expand-Archive -Path "$env:USERPROFILE\Downloads\LanguageTool-5.5.zip" -DestinationPath "$env:USERPROFILE\LanguageTool"

# Install the LanguageTool extension for Visual Studio Code
code --install-extension "adamvoss.vscode-languagetool"

# Configure the LanguageTool extension
$settingsPath = "$env:APPDATA\Code\User\settings.json"
$settings = Get-Content $settingsPath | ConvertFrom-Json
$settings."languagetool.serverPath" = "$env:USERPROFILE\LanguageTool\LanguageTool-5.5"
Set-Content -Path $settingsPath -Value ($settings | ConvertTo-Json -Depth 4)
