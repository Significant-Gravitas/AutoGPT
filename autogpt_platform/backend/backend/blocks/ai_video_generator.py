import logging
from typing import Literal, Optional, Dict, Any, Set
import requests
import time
from autogpt_libs.supabase_integration_credentials_store.types import APIKeyCredentials
from pydantic import SecretStr
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import CredentialsField, CredentialsMetaInput, SchemaField

logger = logging.getLogger(__name__)


class AIVideoGeneratorBlock(Block):
    class Input(BlockSchema):
        video_description: str = SchemaField(
            description="Description of the video to generate.",
            placeholder="A dog running in a field.",
        )
        credentials: CredentialsMetaInput[Literal["fal"], Literal["api_key"]] = (
            CredentialsField(
                provider="fal",
                supported_credential_types={"api_key"},
                description="The FAL integration can be used with any API key with sufficient permissions for video generation.",
            )
        )

    class Output(BlockSchema):
        video_url: str = SchemaField(description="The URL of the generated video.")
        error: str = SchemaField(
            description="Error message if video generation failed."
        )

    def __init__(self):
        test_credentials = APIKeyCredentials(
            id="01234567-89ab-cdef-0123-456789abcdef",
            provider="fal",
            api_key=SecretStr("test-api-key"),
            title="Mock FAL API Key",
            expires_at=None,
        )

        test_credentials_input = {
            "provider": test_credentials.provider,
            "id": test_credentials.id,
            "type": test_credentials.type,
            "title": test_credentials.title,
        }

        super().__init__(
            id="530cf046-2ce0-4854-ae2c-659db17c7a46",
            description="A block that takes in a description of a video and generates a video using the FAL Mochi API.",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "video_description": "A dog running in a field.",
                "credentials": test_credentials_input,
            },
            test_credentials=test_credentials,
            test_output=[("video_url", "https://fal.media/files/.../video.mp4")],
            test_mock={
                "generate_video": lambda *args, **kwargs: "https://fal.media/files/.../video.mp4"
            },
        )

    @staticmethod
    def generate_video(api_key: str, prompt: str) -> str:
        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json"
        }
        
        # Keep track of logs we've already seen
        seen_logs: Set[str] = set()
        
        try:
            # Submit request to queue
            submit_response = requests.post(
                "https://queue.fal.run/fal-ai/mochi-v1",
                headers=headers,
                json={
                    "prompt": prompt,
                    "enable_prompt_expansion": True
                }
            )
            submit_response.raise_for_status()
            request_data = submit_response.json()
            
            # Get request_id and urls from initial response
            request_id = request_data.get("request_id")
            status_url = request_data.get("status_url")
            result_url = request_data.get("response_url")
            
            if not all([request_id, status_url, result_url]):
                raise ValueError("Missing required data in submission response")
            
            # Poll for status with exponential backoff
            max_attempts = 30
            attempt = 0
            base_wait_time = 5
            
            while attempt < max_attempts:
                status_response = requests.get(
                    f"{status_url}?logs=1",
                    headers=headers
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                
                # Process new logs only
                logs = status_data.get('logs', [])
                if logs and isinstance(logs, list):
                    for log in logs:
                        if isinstance(log, dict):
                            # Create a unique key for this log entry
                            log_key = f"{log.get('timestamp', '')}-{log.get('message', '')}"
                            if log_key not in seen_logs:
                                seen_logs.add(log_key)
                                message = log.get('message', '')
                                if message:
                                    logger.debug(f"[AIVideoGeneratorBlock] - [{log.get('level', 'INFO')}] [{log.get('source', '')}] [{log.get('timestamp', '')}] {message}")
                
                status = status_data.get("status")
                if status == "COMPLETED":
                    # Get the final result using the result_url
                    result_response = requests.get(
                        result_url,
                        headers=headers
                    )
                    result_response.raise_for_status()
                    result_data = result_response.json()
                    
                    # Extract video URL from response
                    if "video" not in result_data or not isinstance(result_data["video"], dict):
                        raise ValueError("Invalid response format - missing video data")
                    
                    video_url = result_data["video"].get("url")
                    if not video_url:
                        raise ValueError("No video URL in response")
                    
                    return video_url
                    
                elif status == "FAILED":
                    error_msg = status_data.get("error", "No error details provided")
                    raise RuntimeError(f"Video generation failed: {error_msg}")
                elif status == "IN_QUEUE":
                    position = status_data.get("queue_position", "unknown")
                    logger.debug(f"[AIVideoGeneratorBlock] - Status: In queue, position: {position}")
                elif status == "IN_PROGRESS":
                    logger.debug("[AIVideoGeneratorBlock] - Status: Request is being processed...")
                else:
                    logger.info(f"[AIVideoGeneratorBlock] - Status: Unknown status: {status}")
                
                wait_time = min(base_wait_time * (2 ** attempt), 60)
                time.sleep(wait_time)
                attempt += 1
            
            raise RuntimeError("Maximum polling attempts reached")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            video_url = self.generate_video(
                api_key=credentials.api_key.get_secret_value(),
                prompt=input_data.video_description,
            )
            yield "video_url", video_url
        except Exception as e:
            error_message = str(e)
            yield "error", [error_message]