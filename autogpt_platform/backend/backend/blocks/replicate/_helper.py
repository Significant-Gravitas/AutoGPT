import logging
import re

from pydantic import BaseModel, ValidationError
from replicate.client import Client as ReplicateClient
from replicate.exceptions import ReplicateError
from replicate.helpers import FileOutput
from replicate.prediction import Prediction

logger = logging.getLogger(__name__)

ReplicateOutputs = FileOutput | list[FileOutput] | list[str] | str | list[dict]

_MODEL_NAME = re.compile(r"[\w.-]+/[\w.-]+")
_VERSION = re.compile(r"[A-Za-z0-9]+")


class ReplicateModelNotRunnableError(ValueError):
    """The block can't run this model or version, so no prediction was made."""


async def start_prediction(
    client: ReplicateClient,
    model_name: str,
    version: str | None,
    model_inputs: dict,
    *,
    priced: bool,
) -> Prediction:
    """Start a run; a model the block has no price for runs only if it is a
    community model, which Replicate bills by run time."""
    if not _MODEL_NAME.fullmatch(model_name):
        raise ReplicateModelNotRunnableError(
            f"Replicate model name '{model_name}' must look like 'owner/model-name'."
        )
    if version:
        await _require_version_of(client, model_name, version)
        if not priced:
            await _require_community_model(client, model_name)
        return await client.predictions.async_create(
            version=version, input=model_inputs
        )
    if priced:
        return await create_unpinned_prediction(client, model_name, model_inputs)
    listing = await _require_community_model(client, model_name)
    return await client.predictions.async_create(
        version=_latest_version_id(listing, model_name), input=model_inputs
    )


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

    listing = await _get_listing(client, model_name)
    return await client.predictions.async_create(
        version=_latest_version_id(listing, model_name), input=model_inputs
    )


class _VersionRef(BaseModel):
    id: str


class _ModelListing(BaseModel):
    is_official: bool
    latest_version: _VersionRef | None = None


async def _require_community_model(
    client: ReplicateClient, model_name: str
) -> _ModelListing:
    listing = await _get_listing(client, model_name)
    if listing.is_official:
        raise ReplicateModelNotRunnableError(
            f"Replicate model '{model_name}' is an official model with its own "
            "price, and the Replicate block can't run it yet. Use another model, "
            "or a block made for this one."
        )
    return listing


async def _require_version_of(
    client: ReplicateClient, model_name: str, version: str
) -> None:
    """Replicate runs a version by its hash alone, so check it is the named model's."""
    if not _VERSION.fullmatch(version):
        raise ReplicateModelNotRunnableError(
            f"Version '{version}' is not a Replicate version hash."
        )
    try:
        await client._async_request(
            "GET", f"/v1/models/{model_name}/versions/{version}"
        )
    except ReplicateError as e:
        if e.status != 404:
            raise
        raise ReplicateModelNotRunnableError(
            f"Version '{version}' was not found for Replicate model "
            f"'{model_name}'. Leave 'version' empty to run the model's latest "
            "version."
        ) from e


async def _get_listing(client: ReplicateClient, model_name: str) -> _ModelListing:
    try:
        # The SDK's Model drops `is_official`, so read the listing directly.
        response = await client._async_request("GET", f"/v1/models/{model_name}")
    except ReplicateError as e:
        if e.status != 404:
            raise
        raise ReplicateModelNotRunnableError(
            f"Replicate model '{model_name}' was not found. Check that the name "
            "is 'owner/model-name' and that your API key can access it."
        ) from e
    try:
        return _ModelListing.model_validate(response.json())
    except ValidationError as e:
        raise RuntimeError(
            f"Replicate returned an unexpected listing for model '{model_name}'"
        ) from e


def _latest_version_id(listing: _ModelListing, model_name: str) -> str:
    if listing.latest_version is None:
        raise ReplicateModelNotRunnableError(
            f"Replicate model '{model_name}' has no published version to run. "
            "Set 'version' to a version hash from the model's Versions tab on "
            "Replicate."
        )
    return listing.latest_version.id


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
