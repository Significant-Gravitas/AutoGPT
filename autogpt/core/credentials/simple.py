from typing import Dict

from autogpt.core.configuration import Configuration


class CredentialsManager:
    configuration_defaults = {
        "credentials": {
            "openai": {
                "api_key": "YOUR_API_KEY",
                "azure_configuration": {
                    "api_type": "azure",
                    "api_base": "YOUR_AZURE_API_BASE",
                    "api_version": "YOUR_AZURE_API_VERSION",
                    "deployment_ids": {
                        "fast_model": "YOUR_FAST_LLM_MODEL_DEPLOYMENT_ID",
                        "smart_model": "YOUR_SMART_LLM_MODEL_DEPLOYMENT_ID",
                        "embedding_model": "YOUR_EMBEDDING_MODEL_DEPLOYMENT_ID",
                    },
                },
            },
        },
    }

    def __init__(self, configuration: Configuration):
        self._configuration = configuration.credentials

    def add_credentials(self, service_name: str, credentials: Dict):
        self._configuration[service_name] = credentials
        # TODO: Save to file.

    def get_credentials(self, service_name: str) -> Dict[str, Dict]:
        return self._configuration[service_name].copy()

    def __repr__(self):
        return f"CredentialsManager(services={list(self._configuration)}"
