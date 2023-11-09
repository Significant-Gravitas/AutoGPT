import win32ras

stateStrings = {
    win32ras.RASCS_OpenPort: "OpenPort",
    win32ras.RASCS_PortOpened: "PortOpened",
    win32ras.RASCS_ConnectDevice: "ConnectDevice",
    win32ras.RASCS_DeviceConnected: "DeviceConnected",
    win32ras.RASCS_AllDevicesConnected: "AllDevicesConnected",
    win32ras.RASCS_Authenticate: "Authenticate",
    win32ras.RASCS_AuthNotify: "AuthNotify",
    win32ras.RASCS_AuthRetry: "AuthRetry",
    win32ras.RASCS_AuthCallback: "AuthCallback",
    win32ras.RASCS_AuthChangePassword: "AuthChangePassword",
    win32ras.RASCS_AuthProject: "AuthProject",
    win32ras.RASCS_AuthLinkSpeed: "AuthLinkSpeed",
    win32ras.RASCS_AuthAck: "AuthAck",
    win32ras.RASCS_ReAuthenticate: "ReAuthenticate",
    win32ras.RASCS_Authenticated: "Authenticated",
    win32ras.RASCS_PrepareForCallback: "PrepareForCallback",
    win32ras.RASCS_WaitForModemReset: "WaitForModemReset",
    win32ras.RASCS_WaitForCallback: "WaitForCallback",
    win32ras.RASCS_Projected: "Projected",
    win32ras.RASCS_StartAuthentication: "StartAuthentication",
    win32ras.RASCS_CallbackComplete: "CallbackComplete",
    win32ras.RASCS_LogonNetwork: "LogonNetwork",
    win32ras.RASCS_Interactive: "Interactive",
    win32ras.RASCS_RetryAuthentication: "RetryAuthentication",
    win32ras.RASCS_CallbackSetByCaller: "CallbackSetByCaller",
    win32ras.RASCS_PasswordExpired: "PasswordExpired",
    win32ras.RASCS_Connected: "Connected",
    win32ras.RASCS_Disconnected: "Disconnected",
}


def TestCallback(hras, msg, state, error, exterror):
    print("Callback called with ", hras, msg, stateStrings[state], error, exterror)


def test(rasName="_ Divert Off"):
    return win32ras.Dial(None, None, (rasName,), TestCallback)
