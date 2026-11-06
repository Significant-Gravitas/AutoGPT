# Which running programs the installer closes before it installs, updates or
# uninstalls. electron-builder asks for this file by name (`nsis.include` in
# electron-builder.config.js).
#
# electron-builder's own check (templates/nsis/include/
# allowOnlyOneInstallerInstance.nsh) closes every program whose path starts
# with the install directory, compared as text: "...\Programs\autogpt" is
# also the start of "...\Programs\autogpt-voice". The normal app's installer
# would stop a running variant and its database with it, on every update,
# and a variant's would stop any variant whose slug begins with its own.
#
# The three macros below are electron-builder's, with two changes. The path
# must start with the install directory followed by a backslash. And the
# programs found are counted as a list: PowerShell gives no count for a
# single one, so the template's check does not see the last program of an
# app, such as a runtime whose window has gone. Defining
# customCheckAppRunning is how the template is told to use these macros
# instead of its own. test/packaging.test.js compares them with the
# template, so an electron-builder that changes its check is noticed.

# The template leaves these two out when customCheckAppRunning is defined.
!include "getProcessInfo.nsh"
Var pid

!macro customCheckAppRunning
  Push $0
  System::Call 'Kernel32::SetEnvironmentVariable(t "AUTOGPT_DESKTOP_INSTALL_DIR", t "$INSTDIR\")i.r0'
  ${if} $0 == 0
    Pop $0
    Abort "Could not prepare the install directory for checking running programs."
  ${endIf}
  Pop $0
  !insertmacro IS_POWERSHELL_AVAILABLE
  !insertmacro OWN_CHECK_APP_RUNNING
!macroend

!macro OWN_FIND_PROCESS _FILE _RETURN
  ${if} $IsPowerShellAvailable == 0
    nsExec::Exec `"$PowerShellPath" -C "if (@(Get-CimInstance -ClassName Win32_Process | ? {$$_.Path -and $$_.Path.StartsWith($$env:AUTOGPT_DESKTOP_INSTALL_DIR, 'CurrentCultureIgnoreCase')}).Count -gt 0) { exit 0 } else { exit 1 }"`
    Pop ${_RETURN}
  ${else}
    !ifdef INSTALL_MODE_PER_ALL_USERS
      # exact match via findstr anchored to the start of each CSV line
      nsExec::Exec `"$CmdPath" /C tasklist /FI "IMAGENAME eq ${_FILE}" /FO CSV /NH | "$SYSDIR\findstr.exe" /B /I /C:"\"${_FILE}\""`
      Pop ${_RETURN}
    !else
      # find process owned by current user — anchored exact match
      nsExec::Exec `"$CmdPath" /C tasklist /FI "USERNAME eq %USERNAME%" /FI "IMAGENAME eq ${_FILE}" /FO CSV /NH | "$SYSDIR\findstr.exe" /B /I /C:"\"${_FILE}\""`
      Pop ${_RETURN}
    !endif
  ${endIf}
!macroend

!macro OWN_KILL_PROCESS _FILE _FORCE
  Push $0
  ${if} ${_FORCE} == 1
    ${if} $IsPowerShellAvailable == 0
      StrCpy $0 "-Force"
    ${else}
      StrCpy $0 "/F"
    ${endIf}
  ${else}
    StrCpy $0 ""
  ${endIf}

  ${if} $IsPowerShellAvailable == 0
    nsExec::Exec `"$PowerShellPath" -C "Get-CimInstance -ClassName Win32_Process | ? {$$_.Path -and $$_.Path.StartsWith($$env:AUTOGPT_DESKTOP_INSTALL_DIR, 'CurrentCultureIgnoreCase')} | % { Stop-Process -Id $$_.ProcessId $0 }"`
  ${else}
    !ifdef INSTALL_MODE_PER_ALL_USERS
      nsExec::Exec `taskkill /IM "${_FILE}" /FI "PID ne $pid"`
    !else
      nsExec::Exec `"$CmdPath" /C taskkill $0 /IM "${_FILE}" /FI "PID ne $pid" /FI "USERNAME eq %USERNAME%"`
    !endif
  ${endIf}
  Pop $0
!macroend

!macro OWN_CHECK_APP_RUNNING
  ${GetProcessInfo} 0 $pid $1 $2 $3 $4
  ${if} $3 != "${APP_EXECUTABLE_FILENAME}"
    ${if} ${isUpdated}
      # allow app to exit without explicit kill
      Sleep 300
    ${endIf}

    !insertmacro OWN_FIND_PROCESS "${APP_EXECUTABLE_FILENAME}" $R0
    ${if} $R0 == 0
      ${if} ${isUpdated}
        # allow app to exit without explicit kill
        Sleep 1000
        Goto doStopProcess
      ${endIf}
      MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION "$(appRunning)" /SD IDOK IDOK doStopProcess
      Quit

      doStopProcess:

      DetailPrint "$(appClosing)"

      !insertmacro OWN_KILL_PROCESS "${APP_EXECUTABLE_FILENAME}" 0
      # to ensure that files are not "in-use"
      Sleep 300

      # Retry counter
      StrCpy $R1 0

      loop:
        IntOp $R1 $R1 + 1

        !insertmacro OWN_FIND_PROCESS "${APP_EXECUTABLE_FILENAME}" $R0
        ${if} $R0 == 0
          # wait to give a chance to exit gracefully
          Sleep 1000
          !insertmacro OWN_KILL_PROCESS "${APP_EXECUTABLE_FILENAME}" 1 # 1 = force kill
          !insertmacro OWN_FIND_PROCESS "${APP_EXECUTABLE_FILENAME}" $R0
          ${if} $R0 == 0
            DetailPrint `Waiting for "${PRODUCT_NAME}" to close.`
            Sleep 2000
          ${else}
            Goto not_running
          ${endIf}
        ${else}
          Goto not_running
        ${endIf}

        # App likely running with elevated permissions.
        # Ask user to close it manually
        ${if} $R1 > 1
          MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "$(appCannotBeClosed)" /SD IDCANCEL IDRETRY loop
          Quit
        ${else}
          Goto loop
        ${endIf}
      not_running:
    ${endIf}
  ${endIf}
!macroend

# How an update removes the version it replaces.
#
# electron-builder's uninstaller (templates/nsis/uninstaller.nsh), run by the
# next version's installer, moves the old install out of the way one file at
# a time before it deletes it, so that a file still in use leaves everything
# where it was. For this app that is a rename for each of its twenty thousand
# files, each one looked at by the virus scanner.
#
# One rename of the directory does the same: Windows refuses it while any
# program inside is running or any file inside is open, and then nothing has
# moved. It goes to a sibling, not to the installer's temporary directory:
# a directory cannot be renamed onto another drive, and %TEMP% may be on
# one. When Windows refuses, the removal is electron-builder's own, word for
# word (test/packaging.test.js compares it with the template), which also
# covers an uninstaller that had to be run from inside the directory.
#
# Windows also refuses while the directory is any program's working
# directory, and two programs have it for theirs: this uninstaller (its
# un.onInit goes there), and the new version's installer that is waiting for
# it. An update starts that installer from the running app without naming a
# working directory (electron-updater, BaseUpdater.spawnLog), so it has the
# app's, and the app's is the install directory whenever it was started from
# its shortcut, the relaunch after an update included. customInit below takes
# the installer out of it; the first lines of customRemoveFiles take the
# uninstaller out. Without customInit the rename was refused on every real
# update, and only an installer started from somewhere else got it.
#
# This is compiled into the uninstaller, so it works from the second update
# on: the version that is already installed is removed by its own.

# At the end of the installer's .onInit. Nothing between here and the old
# version's removal writes by a relative path, and the template goes back to
# $INSTDIR before it writes the new files (installSection.nsh).
!macro customInit
  SetOutPath $TEMP
!macroend

!macro customRemoveFiles
  ${if} ${isUpdated}
    # The uninstaller's working directory is $INSTDIR, which would be reason
    # enough for Windows to refuse.
    SetOutPath $TEMP
    # What an earlier update could not finish deleting.
    RMDir /r "$INSTDIR.old-install"
    ClearErrors
    Rename "$INSTDIR" "$INSTDIR.old-install"
    ${if} ${Errors}
      ClearErrors
      DetailPrint "The old version could not be moved away in one piece; moving it file by file."
      SetOutPath $INSTDIR
      CreateDirectory "$PLUGINSDIR\old-install"

      Push ""
      Call un.atomicRMDir
      Pop $R0

      ${if} $R0 != 0
        DetailPrint "File is busy, aborting: $R0"

        # Attempt to restore previous directory
        Push ""
        Call un.restoreFiles
        Pop $R0

        Abort `Can't rename "$INSTDIR" to "$PLUGINSDIR\old-install".`
      ${endif}

      # Move out of $INSTDIR so it can be removed
      SetOutPath $TEMP
      # Remove all files (or remaining shallow directories from the block above)
      RMDir /r $INSTDIR
    ${else}
      DetailPrint "Moved the old version away in one piece."
      RMDir /r "$INSTDIR.old-install"
      ${if} ${Errors}
        # A file in it was still held (a scanner, the indexer). Once more;
        # what is left then goes with the next update or the uninstall.
        Sleep 2000
        RMDir /r "$INSTDIR.old-install"
        ClearErrors
      ${endif}
    ${endif}
  ${else}
    # What an update could not finish deleting goes with the app: nothing
    # else would ever remove it.
    ${if} ${FileExists} "$INSTDIR.old-install\*.*"
      RMDir /r "$INSTDIR.old-install"
      ClearErrors
    ${endif}
    # Move out of $INSTDIR so it can be removed
    SetOutPath $TEMP
    # Remove all files (or remaining shallow directories from the block above)
    RMDir /r $INSTDIR
  ${endif}
!macroend
