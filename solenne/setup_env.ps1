pip install -r requirements.txt
pip install pillow pytesseract
python -m pip install pywin32-postinstall-script --install
$vers = python - <<'PY'
import numpy, httpx
print(numpy.__version__)
print(httpx.__version__)
PY
$lines = $vers -split "`n"
if ([version]$lines[0] -ge [version]'2.0') {
    Write-Host 'WARNING: NumPy >= 2 not supported'
}
if ([version]$lines[1] -ge [version]'0.28') {
    Write-Host 'WARNING: httpx >= 0.28 not supported'
}

function solenne_service {
    param([string]$action)
    $svc = 'SolenneService'
    switch ($action) {
        'install' { python solenne/windows_service.py --startup=auto install }
        'remove'  { python solenne/windows_service.py remove }
        'start'   { python solenne/windows_service.py start }
        'stop'    { python solenne/windows_service.py stop }
        default { Write-Host 'Usage: solenne_service install|remove|start|stop' }
    }
}

function solenne_set_key {
    param([string]$key)
    python - <<'PY'
import sys
try:
    import keyring
except Exception:
    sys.exit('keyring not available')
keyring.set_password('Solenne', 'OPENAI_API_KEY', sys.argv[1])
print('Key stored')
PY
    $key
}
