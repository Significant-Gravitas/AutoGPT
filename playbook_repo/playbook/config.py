import os
import sys

# Try to import from credentials.py
try:
    # Attempt relative import if running as a package
    from . import credentials
except ImportError:
    # Fallback or if not running as package
    credentials = None

BRAIN_API_URL = getattr(credentials, "BRAIN_API_URL", None) if credentials else None
BRAIN_API_KEY = getattr(credentials, "BRAIN_API_KEY", None) if credentials else None

# Fallback to environment variables
if not BRAIN_API_URL:
    BRAIN_API_URL = os.environ.get("BRAIN_API_URL")

if not BRAIN_API_KEY:
    BRAIN_API_KEY = os.environ.get("BRAIN_API_KEY")

# Validation
if not BRAIN_API_URL or not BRAIN_API_KEY:
    print("MISSING_KEYS: Please provide playbook/credentials.py or set BRAIN_API_KEY in environment.")
    sys.exit(1)

def get_masked_info():
    url_masked = BRAIN_API_URL[:10] + "..." if BRAIN_API_URL else "MISSING"
    return f"URL={url_masked}, KEY_PRESENT=true"
