import pytest
from pytest_mock import MockerFixture

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


def test_unresolved_llm_discriminator_defers_to_runtime() -> None:
    field_info = AITextGeneratorBlock.Input.get_credentials_fields_info()["credentials"]

    assert get_no_auth_credentials(field_info, {}) is ollama_credentials


@pytest.mark.asyncio
async def test_preflight_allows_model_supplied_by_upstream_link(
    mocker: MockerFixture,
) -> None:
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "linked-model-node"
    mock_node.credentials_optional = False
    mock_node.input_default = {}
    mock_node.input_links = [mocker.Mock(sink_name="model")]
    mock_node.block = mocker.MagicMock()
    mock_node.block.input_schema = AITextGeneratorBlock.Input

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]
    mock_store = mocker.patch(
        "backend.executor.utils.get_integration_credentials_store"
    )

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="test-user-id",
        nodes_input_masks=None,
    )

    assert errors == {}
    assert nodes_to_skip == set()
    mock_store.assert_not_called()
