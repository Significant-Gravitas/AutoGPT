import httpx
import json
import os
import datetime
import base64
from . import config
from .utils import get_logger

logger = get_logger("brain_client")

class BrainClient:
    def __init__(self):
        self.base_url = config.BRAIN_API_URL.rstrip('/')
        self.api_key = config.BRAIN_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = 30.0

    def _create_blocker(self, response_text, status_code, context, headers=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        log_filename = f"logs/key_validation_error_{timestamp}.json"

        error_data = {
            "context": context,
            "status_code": status_code,
            "headers": dict(headers) if headers else {},
            "content": response_text
        }

        with open(log_filename, "w") as f:
            json.dump(error_data, f, indent=2)

        blocker_content = f"""# KEY VALIDATION BLOCKER

CRITICAL FAILURE in {context}
Timestamp: {timestamp}
Status: {status_code}
Log: {log_filename}

Response:
```
{response_text[:2000]}
```
"""
        with open("KEY_VALIDATION_BLOCKER.md", "w") as f:
            f.write(blocker_content)

        logger.error(f"BLOCKER CREATED: {context} failed with {status_code}")
        # Raise RuntimeError to stop the flow
        raise RuntimeError(f"CRITICAL: {context} failed. See KEY_VALIDATION_BLOCKER.md")

    def _validate_response(self, response, required_fields, context):
        if response.status_code != 200:
            self._create_blocker(response.text, response.status_code, context, response.headers)

        try:
            data = response.json()
        except json.JSONDecodeError:
            self._create_blocker(response.text, response.status_code, f"{context} - Invalid JSON", response.headers)
            return # Unreachable

        missing = [f for f in required_fields if f not in data]
        if missing:
            self._create_blocker(response.text, response.status_code, f"{context} - Schema Mismatch (missing {missing})", response.headers)

        return data

    def check_health(self):
        """Sync health check for startup."""
        url = f"{self.base_url}/extract"
        # Synthetic payload
        payload = {
            "session_id": "health-check-probe",
            "file": {
                "filename": "health_probe.txt",
                "content_b64": base64.b64encode(b"HEALTH CHECK").decode('utf-8')
            }
        }

        logger.info(f"Probing Brain at {url}...")
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload, headers=self.headers)
            self._validate_response(response, ["status", "session_id", "ui_state"], "Health Probe")
            logger.info("Health probe successful.")
            return True
        except httpx.RequestError as e:
            self._create_blocker(str(e), 0, "Health Probe Connection Error")
        except RuntimeError:
            raise

    async def extract(self, session_id, filename, content_b64):
        url = f"{self.base_url}/extract"
        payload = {
            "session_id": session_id,
            "file": {
                "filename": filename,
                "content_b64": content_b64
            }
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=self.headers)

        return self._validate_response(response, ["status", "session_id", "extracted", "ui_state"], "POST /extract")

    async def train(self, session_id, mapping):
        url = f"{self.base_url}/train"
        payload = {
            "session_id": session_id,
            "mapping": mapping,
            "example_execution": {"preview": True}
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=self.headers)

        return self._validate_response(response, ["status", "session_id", "summary", "ui_state"], "POST /train")

    async def finish_training(self, session_id):
        url = f"{self.base_url}/finish_training"
        payload = {
            "session_id": session_id,
            "confirm": True
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=self.headers)

        return self._validate_response(response, ["status", "session_id", "execution_plan"], "POST /finish_training")

    async def execute(self, session_id, files):
        # files is list of {filename, content_b64}
        url = f"{self.base_url}/execute"
        payload = {
            "session_id": session_id,
            "files": files,
            "execution_options": {"dry_run": False}
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=self.headers)

        return self._validate_response(response, ["status", "session_id", "results"], "POST /execute")

    async def get_opportunities(self):
        url = f"{self.base_url}/opportunities"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=self.headers)

        return self._validate_response(response, ["status", "opportunities"], "GET /opportunities")
