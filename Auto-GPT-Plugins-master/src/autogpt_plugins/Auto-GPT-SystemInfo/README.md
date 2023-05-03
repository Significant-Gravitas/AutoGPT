# System Information Plugin for Auto-GPT

## Overview

This plugin adds an extra line to the prompt, serving as a hint for the AI to use shell commands likely supported by the current system.
By incorporating this plugin, you can ensure that the AI model provides more accurate and system-specific shell commands, improving its overall performance and usefulness.

**Example Prompt Included in the Context:**

   - "Shell commands executed on Linux 64bit Debian GNU/Linux 11 (debian) in zsh"

This helps the model provide commands compatible with the current operating system, enabling it to use system-specific package managers and other utilities.

## Installation

Download this repository as a .zip file, copy it to ./plugins/, and rename it to Auto-GPT-SystemInfo.zip.

To download it directly from your Auto-GPT directory, you can run this command on Linux or MacOS:

```
curl -o ./plugins/Auto-GPT-SystemInfo.zip https://github.com/hdkiller/Auto-GPT-SystemInfo/archive/refs/heads/master.zip 
```

In PowerShell:

```
Invoke-WebRequest -Uri "https://github.com/hdkiller/Auto-GPT-SystemInfo/archive/refs/heads/master.zip" -OutFile "./plugins/Auto-GPT-SystemInfo.zip"
```

## Configuration

This plugin is designed to enhance the capabilities of executing shell commands in Auto-GPT, which is only activated when the EXECUTE_LOCAL_COMMANDS=True setting is enabled in the .env file. If that environment variable is not set, the plugin will display a warning and will not inject the operating system details into the context to prevent encouraging the model from executing shell commands.

## Compatibility

The plugin supports Linux, macOS, and Windows.