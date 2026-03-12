"""
Launch the ABN Consulting AI Co-Navigator API server.

Usage:
    python -m autogpt.coaching_server
    uvicorn autogpt.coaching_server:app --host 0.0.0.0 --port 8080 --reload
"""
import os

from dotenv import load_dotenv

load_dotenv()  # Load .env before any Config singleton is instantiated

from autogpt.coaching.api import app  # noqa: E402 — must follow load_dotenv()

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("COACHING_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
