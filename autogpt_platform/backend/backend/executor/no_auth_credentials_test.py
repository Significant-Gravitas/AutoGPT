from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.executor.no_auth_credentials import get_no_auth_credentials
from backend.integrations.credentials_store import ollama_credentials
from backend.integrations.providers import ProviderName


def _model_for(provider: ProviderName) -> LLMModel:
    return next(model for model in LLMModel if model.metadata.provider == provider)


def test_ollama_credentials_are_discriminated_as_no_auth() -> None:
    field_info = AITextGeneratorBlock.Input.get_credentials_fields_info()["credentials"]
    ollama_model = _model_for(ProviderName.OLLAMA)

    discriminated = field_info.discriminate(ollama_model.value)

    assert not discriminated.supported_types
    assert "credentials" in AITextGeneratorBlock.Input.get_required_fields()
    assert (
        get_no_auth_credentials(field_info, {"model": ollama_model.value})
        is ollama_credentials
    )


def test_authenticated_llm_provider_stays_required() -> None:
    field_info = AITextGeneratorBlock.Input.get_credentials_fields_info()["credentials"]
    openai_model = _model_for(ProviderName.OPENAI)

    discriminated = field_info.discriminate(openai_model.value)

    assert discriminated.supported_types == frozenset({"api_key"})
    assert "credentials" in AITextGeneratorBlock.Input.get_required_fields()
    assert get_no_auth_credentials(field_info, {"model": openai_model.value}) is None
