import asyncio
import logging
from typing import Any

from replicate.client import Client as ReplicateClient
from replicate.helpers import FileOutput

logger = logging.getLogger(__name__)

ReplicateOutputs = FileOutput | list[FileOutput] | list[str] | str | list[dict]


def extract_result(output: ReplicateOutputs) -> str:
    result = (
        "Unable to process result. Please contact us with the models and inputs used"
    )
    # Check if output is a list or a string and extract accordingly; otherwise, assign a default message
    if isinstance(output, list) and len(output) > 0:
        # we could use something like all(output, FileOutput) but it will be slower so we just type ignore
        if isinstance(output[0], FileOutput):
            result = output[0].url  # If output is a list, get the first element
        elif isinstance(output[0], str):
            result = "".join(
                output  # type: ignore we're already not a file output here
            )  # type:ignore If output is a list and a str, join the elements the first element. Happens if its text
        elif isinstance(output[0], dict):
            result = str(output[0])
        else:
            logger.error(
                "Replicate generated a new output type that's not a file output or a str in a replicate block"
            )
    elif isinstance(output, FileOutput):
        result = output.url  # If output is a FileOutput, use the url
    elif isinstance(output, str):
        result = output  # If output is a string (for some reason due to their janky type hinting), use it directly
    else:
        result = "No output received"  # Fallback message if output is not as expected
        logger.error(
            "We somehow didn't get an output from a replicate block. This is almost certainly an error"
        )

    return result


async def run_replicate_with_retry(
    client: ReplicateClient,
    model: str,
    input_params: dict[str, Any],
    wait: bool = False,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    last_error = None
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            output = await client.async_run(
                model, input=input_params, wait=wait, **kwargs
            )

            # Check for failed status in response
            is_failed = False
            if isinstance(output, dict) and output.get("status") == "failed":
                is_failed = True
            elif hasattr(output, "status") and getattr(output, "status") == "failed":
                is_failed = True

            if is_failed:
                # Try to get error message
                error_msg = "Replicate prediction failed"
                if isinstance(output, dict):
                    error = output.get("error")
                    if error:
                        error_msg = f"{error_msg}: {error}"
                elif hasattr(output, "error"):
                    error = getattr(output, "error")
                    if error:
                        error_msg = f"{error_msg}: {error}"

                raise RuntimeError(error_msg)

            return output

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Replicate attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Replicate failed after {max_retries} attempts: {str(e)}")
                raise last_error
