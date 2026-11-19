from pydantic import TypeAdapter

from backend.blocks.stagehand.blocks import StagehandRecommendedLLMModel


def test_a_node_saved_on_claude_sonnet_4_5_runs_on_sonnet_4_6():
    """Anthropic retires Sonnet 4.5 on 2026-11-30. A saved node naming it
    still validates, as Sonnet 4.6, so the workflow keeps running."""
    model = TypeAdapter(StagehandRecommendedLLMModel).validate_python(
        "claude-sonnet-4-5-20250929"
    )

    assert model is StagehandRecommendedLLMModel.CLAUDE_4_6_SONNET
    assert model.provider_name == "anthropic/claude-sonnet-4-6"


def test_the_model_list_no_longer_offers_claude_sonnet_4_5():
    assert "claude-sonnet-4-5-20250929" not in {
        model.value for model in StagehandRecommendedLLMModel
    }
