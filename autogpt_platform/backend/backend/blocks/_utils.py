import logging
import os

from backend.integrations.providers import ProviderName

from ._base import AnyBlockSchema

logger = logging.getLogger(__name__)


def is_block_auth_configured(
    block_cls: type[AnyBlockSchema],
) -> bool:
    """
    Check if a block has a valid authentication method configured at runtime.

    For example if a block is an OAuth-only block and there env vars are not set,
    do not show it in the UI.

    """
    from backend.sdk.registry import AutoRegistry

    # Create an instance to access input_schema
    try:
        block = block_cls()
    except Exception as e:
        # If we can't create a block instance, assume it's not OAuth-only
        logger.error(f"Error creating block instance for {block_cls.__name__}: {e}")
        return True
    logger.debug(
        f"Checking if block {block_cls.__name__} has a valid provider configured"
    )

    # Get all credential inputs from input schema
    credential_inputs = block.input_schema.get_credentials_fields_info()
    required_inputs = block.input_schema.get_required_fields()
    if not credential_inputs:
        logger.debug(
            f"Block {block_cls.__name__} has no credential inputs - Treating as valid"
        )
        return True

    # Check credential inputs
    if len(required_inputs.intersection(credential_inputs.keys())) == 0:
        logger.debug(
            f"Block {block_cls.__name__} has only optional credential inputs"
            " - will work without credentials configured"
        )

    # Check if the credential inputs for this block are correctly configured
    for field_name, field_info in credential_inputs.items():
        provider_names = field_info.provider
        if not provider_names:
            logger.warning(
                f"Block {block_cls.__name__} "
                f"has credential input '{field_name}' with no provider options"
                " - Disabling"
            )
            return False

        # If a field has multiple possible providers, each one needs to be usable to
        # prevent breaking the UX
        for _provider_name in provider_names:
            provider_name = _provider_name.value
            if provider_name in ProviderName.__members__.values():
                logger.debug(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"provider '{provider_name}' is part of the legacy provider system"
                    " - Treating as valid"
                )
                break

            provider = AutoRegistry.get_provider(provider_name)
            if not provider:
                logger.warning(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"refers to unknown provider '{provider_name}' - Disabling"
                )
                return False

            # Check the provider's supported auth types
            if field_info.supported_types != provider.supported_auth_types:
                logger.warning(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"has mismatched supported auth types (field <> Provider): "
                    f"{field_info.supported_types} != {provider.supported_auth_types}"
                )

            if not (supported_auth_types := provider.supported_auth_types):
                # No auth methods are been configured for this provider
                logger.warning(
                    f"Block {block_cls.__name__} credential input '{field_name}' "
                    f"provider '{provider_name}' "
                    "has no authentication methods configured - Disabling"
                )
                return False

            # Check if provider supports OAuth
            if "oauth2" in supported_auth_types:
                # Check if OAuth environment variables are set
                if (oauth_config := provider.oauth_config) and bool(
                    os.getenv(oauth_config.client_id_env_var)
                    and os.getenv(oauth_config.client_secret_env_var)
                ):
                    logger.debug(
                        f"Block {block_cls.__name__} credential input '{field_name}' "
                        f"provider '{provider_name}' is configured for OAuth"
                    )
                else:
                    logger.error(
                        f"Block {block_cls.__name__} credential input '{field_name}' "
                        f"provider '{provider_name}' "
                        "is missing OAuth client ID or secret - Disabling"
                    )
                    return False

        logger.debug(
            f"Block {block_cls.__name__} credential input '{field_name}' is valid; "
            f"supported credential types: {', '.join(field_info.supported_types)}"
        )

    return True
