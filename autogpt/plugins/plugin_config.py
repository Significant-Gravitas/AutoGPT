from typing import Any


class PluginConfig:
    """Class for holding configuration of a single plugin"""

    def __init__(self, name: str, enabled: bool = False, config: dict[str, Any] = None):
        self.name = name
        self.enabled = enabled
        # Arbitray config options for this plugin. API keys or plugin-specific options live here.
        self.config = config or {}

    def __repr__(self):
        return f"PluginConfig('{self.name}', {self.enabled}, {str(self.config)}"
