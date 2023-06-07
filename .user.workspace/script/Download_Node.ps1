# Set the installer file name and download URL
$installerFileName = "nodejs-installer.msi"
$installerUrl = "https://nodejs.org/dist/v18.15.0/node-v18.15.0-x64.msi"

# Download the installer file
Invoke-WebRequest -Uri $installerUrl -OutFile $installerFileName

# Install Node.js and npm
Start-Process -FilePath msiexec.exe -ArgumentList "/i $installerFileName" -Wait

# Remove the installer file
Remove-Item $installerFileName