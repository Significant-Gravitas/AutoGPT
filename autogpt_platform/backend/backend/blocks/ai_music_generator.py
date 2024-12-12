import logging
import time
from enum import Enum
from typing import Literal

import replicate
from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="replicate",
    api_key=SecretStr("mock-replicate-api-key"),
    title="Mock Replicate API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


# Model version enum
class MusicGenModelVersion(str, Enum):
    STEREO_LARGE = "stereo-large"
    MELODY_LARGE = "melody-large"
    LARGE = "large"


# Audio format enum
class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


# Normalization strategy enum
class NormalizationStrategy(str, Enum):
    LOUDNESS = "loudness"
    CLIP = "clip"
    PEAK = "peak"
    RMS = "rms"


class AIMusicGeneratorBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REPLICATE], Literal["api_key"]
        ] = CredentialsField(
            description="The Replicate integration can be used with "
            "any API key with sufficient permissions for the blocks it is used on.",
        )
        prompt: str = SchemaField(
            description="A description of the music you want to generate",
            placeholder="e.g., 'An upbeat electronic dance track with heavy bass'",
            title="Prompt",
        )
        music_gen_model_version: MusicGenModelVersion = SchemaField(
            description="Model to use for generation",
            default=MusicGenModelVersion.STEREO_LARGE,
            title="Model Version",
        )
        duration: int = SchemaField(
            description="Duration of the generated audio in seconds",
            default=8,
            title="Duration",
        )
        temperature: float = SchemaField(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity",
            default=1.0,
            title="Temperature",
        )
        top_k: int = SchemaField(
            description="Reduces sampling to the k most likely tokens",
            default=250,
            title="Top K",
        )
        top_p: float = SchemaField(
            description="Reduces sampling to tokens with cumulative probability of p. When set to 0 (default), top_k sampling is used",
            default=0.0,
            title="Top P",
        )
        classifier_free_guidance: int = SchemaField(
            description="Increases the influence of inputs on the output. Higher values produce lower-variance outputs that adhere more closely to inputs",
            default=3,
            title="Classifier Free Guidance",
        )
        output_format: AudioFormat = SchemaField(
            description="Output format for generated audio",
            default=AudioFormat.WAV,
            title="Output Format",
        )
        normalization_strategy: NormalizationStrategy = SchemaField(
            description="Strategy for normalizing audio",
            default=NormalizationStrategy.LOUDNESS,
            title="Normalization Strategy",
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="URL of the generated audio file")
        error: str = SchemaField(description="Error message if the model run failed")

    def __init__(self):
        super().__init__(
            id="44f6c8ad-d75c-4ae1-8209-aad1c0326928",
            description="This block generates music using Meta's MusicGen model on Replicate.",
            categories={BlockCategory.AI},
            input_schema=AIMusicGeneratorBlock.Input,
            output_schema=AIMusicGeneratorBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "prompt": "An upbeat electronic dance track with heavy bass",
                "music_gen_model_version": MusicGenModelVersion.STEREO_LARGE,
                "duration": 8,
                "temperature": 1.0,
                "top_k": 250,
                "top_p": 0.0,
                "classifier_free_guidance": 3,
                "output_format": AudioFormat.WAV,
                "normalization_strategy": NormalizationStrategy.LOUDNESS,
            },
            test_output=[
                (
                    "result",
                    "https://replicate.com/output/generated-audio-url.wav",
                ),
            ],
            test_mock={
                "run_model": lambda api_key, music_gen_model_version, prompt, duration, temperature, top_k, top_p, classifier_free_guidance, output_format, normalization_strategy: "https://replicate.com/output/generated-audio-url.wav",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        max_retries = 3
        retry_delay = 5  # seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"[AIMusicGeneratorBlock] - Running model (attempt {attempt + 1})"
                )
                result = self.run_model(
                    api_key=credentials.api_key,
                    music_gen_model_version=input_data.music_gen_model_version,
                    prompt=input_data.prompt,
                    duration=input_data.duration,
                    temperature=input_data.temperature,
                    top_k=input_data.top_k,
                    top_p=input_data.top_p,
                    classifier_free_guidance=input_data.classifier_free_guidance,
                    output_format=input_data.output_format,
                    normalization_strategy=input_data.normalization_strategy,
                )
                if result and result != "No output received":
                    yield "result", result
                    return
                else:
                    last_error = "Model returned empty or invalid response"
                    raise ValueError(last_error)
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"[AIMusicGeneratorBlock] - Error: {last_error}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

        # If we've exhausted all retries, yield the error
        yield "error", f"Failed after {max_retries} attempts. Last error: {last_error}"

    def run_model(
        self,
        api_key: SecretStr,
        music_gen_model_version: MusicGenModelVersion,
        prompt: str,
        duration: int,
        temperature: float,
        top_k: int,
        top_p: float,
        classifier_free_guidance: int,
        output_format: AudioFormat,
        normalization_strategy: NormalizationStrategy,
    ):
        # Initialize Replicate client with the API key
        client = replicate.Client(api_token=api_key.get_secret_value())

        # Run the model with parameters
        output = client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": prompt,
                "music_gen_model_version": music_gen_model_version,
                "duration": duration,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "classifier_free_guidance": classifier_free_guidance,
                "output_format": output_format,
                "normalization_strategy": normalization_strategy,
            },
        )

        # Handle the output
        if isinstance(output, list) and len(output) > 0:
            result_url = output[0]  # If output is a list, get the first element
        elif isinstance(output, str):
            result_url = output  # If output is a string, use it directly
        else:
            result_url = (
                "No output received"  # Fallback message if output is not as expected
            )

        return result_url
