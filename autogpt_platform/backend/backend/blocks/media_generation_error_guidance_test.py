from backend.blocks.ai_image_customizer import (
    TEST_CREDENTIALS as REPLICATE_TEST_CREDENTIALS,
)
from backend.blocks.ai_image_customizer import (
    TEST_CREDENTIALS_INPUT as REPLICATE_TEST_CREDENTIALS_INPUT,
)
from backend.blocks.ai_image_customizer import AIImageCustomizerBlock, GeminiImageModel
from backend.blocks.ai_image_generator_block import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AIImageGeneratorBlock,
    ImageGenModel,
    ImageSize,
    ImageStyle,
)
from backend.blocks.ai_shortform_video_block import _missing_project_id_message
from backend.blocks.fal._auth import TEST_CREDENTIALS as FAL_TEST_CREDENTIALS
from backend.blocks.fal._auth import (
    TEST_CREDENTIALS_INPUT as FAL_TEST_CREDENTIALS_INPUT,
)
from backend.blocks.fal.ai_video_generator import AIVideoGeneratorBlock, FalModel
from backend.blocks.flux_kontext import TEST_CREDENTIALS as FLUX_TEST_CREDENTIALS
from backend.blocks.flux_kontext import (
    TEST_CREDENTIALS_INPUT as FLUX_TEST_CREDENTIALS_INPUT,
)
from backend.blocks.flux_kontext import (
    AIImageEditorBlock,
    AspectRatio,
    ImageEditorModel,
)
from backend.blocks.ideogram import TEST_CREDENTIALS as IDEOGRAM_TEST_CREDENTIALS
from backend.blocks.ideogram import (
    TEST_CREDENTIALS_INPUT as IDEOGRAM_TEST_CREDENTIALS_INPUT,
)
from backend.blocks.ideogram import AspectRatio as IdeogramAspectRatio
from backend.blocks.ideogram import (
    ColorPalettePreset,
    IdeogramModelBlock,
    IdeogramModelName,
    MagicPromptOption,
    StyleType,
    UpscaleOption,
)
from backend.blocks.replicate._auth import (
    TEST_CREDENTIALS as REPLICATE_FLUX_TEST_CREDENTIALS,
)
from backend.blocks.replicate._auth import (
    TEST_CREDENTIALS_INPUT as REPLICATE_FLUX_TEST_CREDENTIALS_INPUT,
)
from backend.blocks.replicate._helper import (
    NO_OUTPUT_MESSAGE,
    UNPROCESSABLE_OUTPUT_MESSAGE,
)
from backend.blocks.replicate.flux_advanced import (
    ImageType,
    ReplicateFluxAdvancedModelBlock,
    ReplicateFluxModelName,
)
from backend.blocks.talking_head import _missing_clip_id_message
from backend.data.execution import ExecutionContext


async def test_image_generator_storage_error_does_not_suggest_model_fallback(
    monkeypatch,
):
    async def generate_image(self, input_data, credentials):
        return "data:image/png;base64,AAAA"

    async def store_media_file(*args, **kwargs):
        raise ValueError("Workspace storage timed out")

    monkeypatch.setattr(AIImageGeneratorBlock, "generate_image", generate_image)
    monkeypatch.setattr(
        "backend.blocks.ai_image_generator_block.store_media_file", store_media_file
    )

    outputs = await _run_image_generator()

    assert outputs == [("error", "Workspace storage timed out")]


async def test_image_generator_provider_error_suggests_model_fallback(monkeypatch):
    async def generate_image(self, input_data, credentials):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(AIImageGeneratorBlock, "generate_image", generate_image)

    outputs = await _run_image_generator()

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another image generation model" in outputs[0][1]


async def _run_image_generator():
    block = AIImageGeneratorBlock()
    outputs = []
    async for output in block.run(
        AIImageGeneratorBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            prompt="A test image",
            model=ImageGenModel.NANO_BANANA_2,
            size=ImageSize.SQUARE,
            style=ImageStyle.ANY,
        ),
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(user_id="user", graph_exec_id="exec"),
    ):
        outputs.append(output)

    return outputs


async def test_image_customizer_provider_error_suggests_model_fallback(monkeypatch):
    async def run_model(*args, **kwargs):
        raise RuntimeError("Provider rate limit exceeded")

    monkeypatch.setattr(AIImageCustomizerBlock, "run_model", run_model)

    block = AIImageCustomizerBlock()
    outputs = []
    async for output in block.run(
        AIImageCustomizerBlock.Input(
            credentials=REPLICATE_TEST_CREDENTIALS_INPUT,
            prompt="Make it pop",
            model=GeminiImageModel.NANO_BANANA,
            images=[],
        ),
        credentials=REPLICATE_TEST_CREDENTIALS,
        execution_context=ExecutionContext(user_id="user", graph_exec_id="exec"),
    ):
        outputs.append(output)

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another image generation model" in outputs[0][1]


async def test_ideogram_provider_error_suggests_model_fallback(monkeypatch):
    async def run_model(*args, **kwargs):
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(IdeogramModelBlock, "run_model", run_model)

    block = IdeogramModelBlock()
    outputs = []
    async for output in block.run(
        IdeogramModelBlock.Input(
            credentials=IDEOGRAM_TEST_CREDENTIALS_INPUT,
            ideogram_model_name=IdeogramModelName.V2,
            prompt="A futuristic city",
            aspect_ratio=IdeogramAspectRatio.ASPECT_1_1,
            upscale=UpscaleOption.NO_UPSCALE,
            magic_prompt_option=MagicPromptOption.AUTO,
            seed=None,
            style_type=StyleType.AUTO,
            negative_prompt=None,
            color_palette_name=ColorPalettePreset.NONE,
            custom_color_palette=None,
        ),
        credentials=IDEOGRAM_TEST_CREDENTIALS,
    ):
        outputs.append(output)

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another image generation model" in outputs[0][1]


async def test_fal_video_generator_provider_error_suggests_model_fallback(monkeypatch):
    async def generate_video(*args, **kwargs):
        raise RuntimeError("API request failed: gateway timeout")

    monkeypatch.setattr(AIVideoGeneratorBlock, "generate_video", generate_video)

    block = AIVideoGeneratorBlock()
    outputs = []
    async for output in block.run(
        AIVideoGeneratorBlock.Input(
            credentials=FAL_TEST_CREDENTIALS_INPUT,
            prompt="A dog running",
            model=FalModel.MOCHI,
        ),
        credentials=FAL_TEST_CREDENTIALS,
        execution_context=ExecutionContext(user_id="user", graph_exec_id="exec"),
    ):
        outputs.append(output)

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another video generation model or block" in outputs[0][1]


async def test_flux_kontext_provider_error_suggests_model_fallback(monkeypatch):
    async def run_model(*args, **kwargs):
        raise RuntimeError("Replicate model overloaded")

    monkeypatch.setattr(AIImageEditorBlock, "run_model", run_model)

    block = AIImageEditorBlock()
    outputs = []
    async for output in block.run(
        AIImageEditorBlock.Input(
            credentials=FLUX_TEST_CREDENTIALS_INPUT,
            prompt="Add a hat",
            model=ImageEditorModel.FLUX_KONTEXT_PRO,
            input_image=None,
            aspect_ratio=AspectRatio.MATCH_INPUT_IMAGE,
            seed=None,
        ),
        credentials=FLUX_TEST_CREDENTIALS,
        execution_context=ExecutionContext(user_id="user", graph_exec_id="exec"),
    ):
        outputs.append(output)

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another image generation model" in outputs[0][1]


async def test_replicate_flux_provider_error_suggests_model_fallback(monkeypatch):
    async def run_model(*args, **kwargs):
        raise RuntimeError("503 Replicate temporarily unavailable")

    monkeypatch.setattr(ReplicateFluxAdvancedModelBlock, "run_model", run_model)

    block = ReplicateFluxAdvancedModelBlock()
    outputs = []
    async for output in block.run(
        ReplicateFluxAdvancedModelBlock.Input(
            credentials=REPLICATE_FLUX_TEST_CREDENTIALS_INPUT,
            replicate_model_name=ReplicateFluxModelName.FLUX_SCHNELL,
            prompt="A beautiful landscape",
            seed=42,
            steps=25,
            guidance=3.0,
            interval=2.0,
            aspect_ratio="1:1",
            output_format=ImageType.PNG,
            output_quality=80,
            safety_tolerance=2,
        ),
        credentials=REPLICATE_FLUX_TEST_CREDENTIALS,
    ):
        outputs.append(output)

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another image generation model" in outputs[0][1]


async def test_replicate_flux_unprocessable_output_suggests_model_fallback(monkeypatch):
    async def run_model(*args, **kwargs):
        return UNPROCESSABLE_OUTPUT_MESSAGE

    monkeypatch.setattr(ReplicateFluxAdvancedModelBlock, "run_model", run_model)

    block = ReplicateFluxAdvancedModelBlock()
    outputs = []
    async for output in block.run(
        ReplicateFluxAdvancedModelBlock.Input(
            credentials=REPLICATE_FLUX_TEST_CREDENTIALS_INPUT,
            replicate_model_name=ReplicateFluxModelName.FLUX_SCHNELL,
            prompt="A beautiful landscape",
            seed=42,
            steps=25,
            guidance=3.0,
            interval=2.0,
            aspect_ratio="1:1",
            output_format=ImageType.PNG,
            output_quality=80,
            safety_tolerance=2,
        ),
        credentials=REPLICATE_FLUX_TEST_CREDENTIALS,
    ):
        outputs.append(output)

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another image generation model" in outputs[0][1]


async def test_replicate_flux_no_output_suggests_model_fallback(monkeypatch):
    async def run_model(*args, **kwargs):
        return NO_OUTPUT_MESSAGE

    monkeypatch.setattr(ReplicateFluxAdvancedModelBlock, "run_model", run_model)

    block = ReplicateFluxAdvancedModelBlock()
    outputs = []
    async for output in block.run(
        ReplicateFluxAdvancedModelBlock.Input(
            credentials=REPLICATE_FLUX_TEST_CREDENTIALS_INPUT,
            replicate_model_name=ReplicateFluxModelName.FLUX_SCHNELL,
            prompt="A beautiful landscape",
            seed=42,
            steps=25,
            guidance=3.0,
            interval=2.0,
            aspect_ratio="1:1",
            output_format=ImageType.PNG,
            output_quality=80,
            safety_tolerance=2,
        ),
        credentials=REPLICATE_FLUX_TEST_CREDENTIALS,
    ):
        outputs.append(output)

    assert outputs, "block produced no outputs"
    assert outputs[0][0] == "error"
    assert "try another image generation model" in outputs[0][1]


def test_revid_missing_project_id_preserves_provider_response_detail():
    message = _missing_project_id_message({"error": "Bad Request: invalid input"})

    assert message == (
        "Failed to create video: No project ID returned: Bad Request: invalid input"
    )
    assert "try another video generation model or block" not in message


def test_d_id_missing_clip_id_preserves_provider_response_detail():
    message = _missing_clip_id_message({"message": "Bad Request: invalid presenter"})

    assert (
        message == "Clip creation returned no clip ID: Bad Request: invalid presenter"
    )
    assert "try another video generation model or block" not in message
