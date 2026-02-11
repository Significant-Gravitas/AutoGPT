def _rebuild_library_agent_preset() -> None:
    """Deferred model rebuild to avoid circular import."""
    from backend.api.features.library.model import LibraryAgentPreset

    from .graph import NodeModel
    from .integrations import Webhook  # noqa: F401

    # Resolve Webhook forward references
    NodeModel.model_rebuild()
    LibraryAgentPreset.model_rebuild()


_rebuild_library_agent_preset()
