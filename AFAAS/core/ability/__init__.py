import warnings

warnings.warn(
    "AFAAS.core.ability is deprecated and will be removed in a future version. "
    "Use AFAAS.core.tools instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Set up everything from tools to be accessible via ability
