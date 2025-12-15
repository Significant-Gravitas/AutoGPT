# Import the provider builder to ensure it's registered
from backend.sdk.registry import AutoRegistry

from .triggers import GenericWebhookTriggerBlock, generic_webhook

# Ensure the SDK registry is patched to include our webhook manager
AutoRegistry.patch_integrations()

__all__ = ["GenericWebhookTriggerBlock", "generic_webhook"]
