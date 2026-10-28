import logging

from replicate.client import Client as ReplicateClient
from replicate.exceptions import ReplicateError
from replicate.helpers import FileOutput
from replicate.prediction import Prediction

logger = logging.getLogger(__name__)

ReplicateOutputs = FileOutput | list[FileOutput] | list[str] | str | list[dict]


class ReplicateModelNotRunnableError(ValueError):
    """The model has no version we can run: it doesn't exist or has none."""


async def create_unpinned_prediction(
    client: ReplicateClient, model_name: str, model_inputs: dict
) -> Prediction:
    """Start a prediction for a model given without a version hash.

    Replicate's model-level endpoint (``/v1/models/{owner}/{name}/predictions``)
    only serves official models and answers 404 for community models, so on
    a 404 we run the model's latest published version instead.
    """
    try:
        return await client.predictions.async_create(
            model=model_name, input=model_inputs
        )
    except ReplicateError as e:
        if e.status != 404:
            raise

    version = await _resolve_latest_version(client, model_name)
    return await client.predictions.async_create(version=version, input=model_inputs)


async def _resolve_latest_version(client: ReplicateClient, model_name: str) -> str:
    try:
        model = await client.models.async_get(model_name)
    except ReplicateError as e:
        if e.status != 404:
            raise
        raise ReplicateModelNotRunnableError(
            f"Replicate model '{model_name}' was not found. Check that the name "
            "is 'owner/model-name' and that your API key can access it."
        ) from e

    if model.latest_version is None:
        raise ReplicateModelNotRunnableError(
            f"Replicate model '{model_name}' has no published version to run. "
            "Set 'version' to a version hash from the model's Versions tab on "
            "Replicate."
        )
    return model.latest_version.id


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
