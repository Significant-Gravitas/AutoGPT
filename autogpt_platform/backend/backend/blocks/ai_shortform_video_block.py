import logging
import time
from enum import Enum
from typing import List, Literal, Optional

from pydantic import SecretStr

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import requests

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="revid",
    api_key=SecretStr("mock-revid-api-key"),
    title="Mock Revid API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


class AudioTrack(str, Enum):
    OBSERVER = ("Observer",)
    FUTURISTIC_BEAT = ("Futuristic Beat",)
    SCIENCE_DOCUMENTARY = ("Science Documentary",)
    HOTLINE = ("Hotline",)
    BLADERUNNER_2049 = ("Bladerunner 2049",)
    A_FUTURE = ("A Future",)
    ELYSIAN_EMBERS = ("Elysian Embers",)
    INSPIRING_CINEMATIC = ("Inspiring Cinematic",)
    BLADERUNNER_REMIX = ("Bladerunner Remix",)
    IZZAMUZZIC = ("Izzamuzzic",)
    NAS = ("Nas",)
    PARIS_ELSE = ("Paris - Else",)
    SNOWFALL = ("Snowfall",)
    BURLESQUE = ("Burlesque",)
    CORNY_CANDY = ("Corny Candy",)
    HIGHWAY_NOCTURNE = ("Highway Nocturne",)
    I_DONT_THINK_SO = ("I Don't Think So",)
    LOSING_YOUR_MARBLES = ("Losing Your Marbles",)
    REFRESHER = ("Refresher",)
    TOURIST = ("Tourist",)
    TWIN_TYCHES = ("Twin Tyches",)
    DONT_STOP_ME_ABSTRACT_FUTURE_BASS = ("Dont Stop Me Abstract Future Bass",)

    @property
    def audio_url(self):
        audio_urls = {
            AudioTrack.OBSERVER: "https://cdn.tfrv.xyz/audio/observer.mp3",
            AudioTrack.FUTURISTIC_BEAT: "https://cdn.tfrv.xyz/audio/_futuristic-beat.mp3",
            AudioTrack.SCIENCE_DOCUMENTARY: "https://cdn.tfrv.xyz/audio/_science-documentary.mp3",
            AudioTrack.HOTLINE: "https://cdn.tfrv.xyz/audio/_hotline.mp3",
            AudioTrack.BLADERUNNER_2049: "https://cdn.tfrv.xyz/audio/_bladerunner-2049.mp3",
            AudioTrack.A_FUTURE: "https://cdn.tfrv.xyz/audio/a-future.mp3",
            AudioTrack.ELYSIAN_EMBERS: "https://cdn.tfrv.xyz/audio/elysian-embers.mp3",
            AudioTrack.INSPIRING_CINEMATIC: "https://cdn.tfrv.xyz/audio/inspiring-cinematic-ambient.mp3",
            AudioTrack.BLADERUNNER_REMIX: "https://cdn.tfrv.xyz/audio/bladerunner-remix.mp3",
            AudioTrack.IZZAMUZZIC: "https://cdn.tfrv.xyz/audio/_izzamuzzic.mp3",
            AudioTrack.NAS: "https://cdn.tfrv.xyz/audio/_nas.mp3",
            AudioTrack.PARIS_ELSE: "https://cdn.tfrv.xyz/audio/_paris-else.mp3",
            AudioTrack.SNOWFALL: "https://cdn.tfrv.xyz/audio/_snowfall.mp3",
            AudioTrack.BURLESQUE: "https://cdn.tfrv.xyz/audio/burlesque.mp3",
            AudioTrack.CORNY_CANDY: "https://cdn.tfrv.xyz/audio/corny-candy.mp3",
            AudioTrack.HIGHWAY_NOCTURNE: "https://cdn.tfrv.xyz/audio/highway-nocturne.mp3",
            AudioTrack.I_DONT_THINK_SO: "https://cdn.tfrv.xyz/audio/i-dont-think-so.mp3",
            AudioTrack.LOSING_YOUR_MARBLES: "https://cdn.tfrv.xyz/audio/losing-your-marbles.mp3",
            AudioTrack.REFRESHER: "https://cdn.tfrv.xyz/audio/refresher.mp3",
            AudioTrack.TOURIST: "https://cdn.tfrv.xyz/audio/tourist.mp3",
            AudioTrack.TWIN_TYCHES: "https://cdn.tfrv.xyz/audio/twin-tynches.mp3",
            AudioTrack.DONT_STOP_ME_ABSTRACT_FUTURE_BASS: "https://cdn.revid.ai/audio/_dont-stop-me-abstract-future-bass.mp3",
        }
        return audio_urls[self]


class GenerationPreset(str, Enum):
    LEONARDO = ("Default",)
    ANIME = ("Anime",)
    REALISM = ("Realist",)
    ILLUSTRATION = ("Illustration",)
    SKETCH_COLOR = ("Sketch Color",)
    SKETCH_BW = ("Sketch B&W",)
    PIXAR = ("Pixar",)
    INK = ("Japanese Ink",)
    RENDER_3D = ("3D Render",)
    LEGO = ("Lego",)
    SCIFI = ("Sci-Fi",)
    RECRO_CARTOON = ("Retro Cartoon",)
    PIXEL_ART = ("Pixel Art",)
    CREATIVE = ("Creative",)
    PHOTOGRAPHY = ("Photography",)
    RAYTRACED = ("Raytraced",)
    ENVIRONMENT = ("Environment",)
    FANTASY = ("Fantasy",)
    ANIME_SR = ("Anime Realism",)
    MOVIE = ("Movie",)
    STYLIZED_ILLUSTRATION = ("Stylized Illustration",)
    MANGA = ("Manga",)
    DEFAULT = ("DEFAULT",)


class Voice(str, Enum):
    LILY = "Lily"
    DANIEL = "Daniel"
    BRIAN = "Brian"
    JESSICA = "Jessica"
    CHARLOTTE = "Charlotte"
    CALLUM = "Callum"
    EVA = "Eva"

    @property
    def voice_id(self):
        voice_id_map = {
            Voice.LILY: "pFZP5JQG7iQjIQuC4Bku",
            Voice.DANIEL: "onwK4e9ZLuTAKqWW03F9",
            Voice.BRIAN: "nPczCjzI2devNBz1zQrb",
            Voice.JESSICA: "cgSgspJ2msm6clMCkdW9",
            Voice.CHARLOTTE: "XB0fDUnXU5powFXDhCwa",
            Voice.CALLUM: "N2lVS1w4EtoT3dr4eOWO",
            Voice.EVA: "FGY2WhTYpPnrIDTdsKH5",
        }
        return voice_id_map[self]

    def __str__(self):
        return self.value


class VisualMediaType(str, Enum):
    STOCK_VIDEOS = ("stockVideo",)
    MOVING_AI_IMAGES = ("movingImage",)
    AI_VIDEO = ("aiVideo",)


logger = logging.getLogger(__name__)

class _RevidMixin:
    """Utility mix‑in that bundles the shared webhook / polling helpers."""

    def create_webhook(self):
        url = "https://webhook.site/token"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers)
        webhook_data = response.json()
        return webhook_data["uuid"], f"https://webhook.site/{webhook_data['uuid']}"

    def create_video(self, api_key: SecretStr, payload: dict) -> dict:
        url = "https://www.revid.ai/api/public/v2/render"
        headers = {"key": api_key.get_secret_value()}
        response = requests.post(url, json=payload, headers=headers)
        logger.debug(
            f"API Response Status Code: {response.status_code}, Content: {response.text}"
        )
        return response.json()

    def check_video_status(self, api_key: SecretStr, pid: str) -> dict:
        url = f"https://www.revid.ai/api/public/v2/status?pid={pid}"
        headers = {"key": api_key.get_secret_value()}
        response = requests.get(url, headers=headers)
        return response.json()

    def wait_for_video(
        self,
        api_key: SecretStr,
        pid: str,
        webhook_token: str,
        max_wait_time: int = 1000,
    ) -> str:
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status = self.check_video_status(api_key, pid)
            logger.debug(f"Video status: {status}")

            if status.get("status") == "ready" and "videoUrl" in status:
                return status["videoUrl"]
            elif status.get("status") == "error":
                error_message = status.get("error", "Unknown error occurred")
                logger.error(f"Video creation failed: {error_message}")
                raise ValueError(f"Video creation failed: {error_message}")
            elif status.get("status") in ["FAILED", "CANCELED"]:
                logger.error(f"Video creation failed: {status.get('message')}")
                raise ValueError(f"Video creation failed: {status.get('message')}")

        logger.error("Video creation timed out")
        raise TimeoutError("Video creation timed out")


class AIShortformVideoCreatorBlock(Block, _RevidMixin):
    """Creates a short‑form text‑to‑video clip using stock or AI imagery."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REVID], Literal["api_key"]
        ] = CredentialsField(
            description="The revid.ai integration can be used with "
            "any API key with sufficient permissions for the blocks it is used on.",
        )
        script: str = SchemaField(
            description="""1. Use short and punctuated sentences\n\n2. Use linebreaks to create a new clip\n\n3. Text outside of brackets is spoken by the AI, and [text between brackets] will be used to guide the visual generation. For example, [close-up of a cat] will show a close-up of a cat.""",
            placeholder="[close-up of a cat] Meow!",
        )
        ratio: str = SchemaField(
            description="Aspect ratio of the video", default="9 / 16"
        )
        resolution: str = SchemaField(
            description="Resolution of the video", default="720p"
        )
        frame_rate: int = SchemaField(description="Frame rate of the video", default=60)
        generation_preset: GenerationPreset = SchemaField(
            description="Generation preset for visual style - only effects AI generated visuals",
            default=GenerationPreset.LEONARDO,
            placeholder=GenerationPreset.LEONARDO,
        )
        background_music: AudioTrack = SchemaField(
            description="Background music track",
            default=AudioTrack.HIGHWAY_NOCTURNE,
            placeholder=AudioTrack.HIGHWAY_NOCTURNE,
        )
        voice: Voice = SchemaField(
            description="AI voice to use for narration",
            default=Voice.LILY,
            placeholder=Voice.LILY,
        )
        video_style: VisualMediaType = SchemaField(
            description="Type of visual media to use for the video",
            default=VisualMediaType.STOCK_VIDEOS,
            placeholder=VisualMediaType.STOCK_VIDEOS,
        )

    class Output(BlockSchema):
        video_url: str = SchemaField(description="The URL of the created video")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="361697fb-0c4f-4feb-aed3-8320c88c771b",
            description="Creates a shortform video using revid.ai",
            categories={BlockCategory.SOCIAL, BlockCategory.AI},
            input_schema=AIShortformVideoCreatorBlock.Input,
            output_schema=AIShortformVideoCreatorBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "script": "[close-up of a cat] Meow!",
                "ratio": "9 / 16",
                "resolution": "720p",
                "frame_rate": 60,
                "generation_preset": GenerationPreset.LEONARDO,
                "background_music": AudioTrack.HIGHWAY_NOCTURNE,
                "voice": Voice.LILY,
                "video_style": VisualMediaType.STOCK_VIDEOS,
            },
            test_output=(
                "video_url",
                "https://example.com/video.mp4",
            ),
            test_mock={
                "create_webhook": lambda: (
                    "test_uuid",
                    "https://webhook.site/test_uuid",
                ),
                "create_video": lambda api_key, payload: {"pid": "test_pid"},
                "wait_for_video": lambda api_key, pid, webhook_token, max_wait_time=1000: "https://example.com/video.mp4",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        webhook_token, webhook_url = self.create_webhook()
        logger.debug(f"Webhook URL: {webhook_url}")

        payload = {
            "frameRate": input_data.frame_rate,
            "resolution": input_data.resolution,
            "frameDurationMultiplier": 18,
            "webhook": webhook_url,
            "creationParams": {
                "mediaType": input_data.video_style,
                "captionPresetName": "Wrap 1",
                "selectedVoice": input_data.voice.voice_id,
                "hasEnhancedGeneration": True,
                "generationPreset": input_data.generation_preset.name,
                "selectedAudio": input_data.background_music.value,
                "origin": "/create",
                "inputText": input_data.script,
                "flowType": "text-to-video",
                "slug": "create-tiktok-video",
                "hasToGenerateVoice": True,
                "hasToTranscript": False,
                "hasToSearchMedia": True,
                "hasAvatar": False,
                "hasWebsiteRecorder": False,
                "hasTextSmallAtBottom": False,
                "ratio": input_data.ratio,
                "sourceType": "contentScraping",
                "selectedStoryStyle": {"value": "custom", "label": "Custom"},
                "hasToGenerateVideos": input_data.video_style != VisualMediaType.STOCK_VIDEOS,
                "audioUrl": input_data.background_music.audio_url,
            },
        }

        logger.debug("Creating video...")
        response = self.create_video(credentials.api_key, payload)
        pid = response.get("pid")

        if not pid:
            logger.error(
                f"Failed to create video: No project ID returned. API Response: {response}"
            )
            raise RuntimeError("Failed to create video: No project ID returned")

        logger.debug(f"Video created with project ID: {pid}. Waiting for completion...")
        video_url = self.wait_for_video(credentials.api_key, pid, webhook_token)
        logger.debug(f"Video ready: {video_url}")
        yield "video_url", video_url


class AIAdMakerVideoCreatorBlock(Block, _RevidMixin):
    """Generates a 30‑second vertical AI advert using optional user‑supplied imagery."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REVID], Literal["api_key"]
        ] = CredentialsField(
            description="Credentials for Revid.ai API access.",
        )
        script: str = SchemaField(
            description="Short advertising copy. Line breaks create new scenes.",
            placeholder="Introducing Foobar – [show product photo] the gadget that does it all.",
        )
        ratio: str = SchemaField(description="Aspect ratio", default="9 / 16")
        target_duration: int = SchemaField(
            description="Desired length of the ad in seconds.", default=30
        )
        voice: Voice = SchemaField(
            description="Narration voice", default=Voice.EVA, placeholder=Voice.EVA
        )
        background_music: AudioTrack = SchemaField(
            description="Background track", default=AudioTrack.DONT_STOP_ME_ABSTRACT_FUTURE_BASS
        )
        input_media_urls: List[str] = SchemaField(
            description="List of image URLs to feature in the advert.", default=[]
        )
        use_only_provided_media: bool = SchemaField(
            description="Restrict visuals to supplied images only.", default=True
        )

    class Output(BlockSchema):
        video_url: str = SchemaField(description="URL of the finished advert")
        error: str = SchemaField(description="Error message on failure")

    def __init__(self):
        super().__init__(
            id="3e3fd845-000e-457f-9f50-9f2f9e278bbd",
            description="Creates an AI‑generated 30‑second advert (text + images)",
            categories={BlockCategory.MARKETING, BlockCategory.AI},
            input_schema=AIAdMakerVideoCreatorBlock.Input,
            output_schema=AIAdMakerVideoCreatorBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "script": "Test product launch!",
                "input_media_urls": [
                    "https://cdn.revid.ai/uploads/1747076315114-image.png",
                ],
            },
            test_output=("video_url", "https://example.com/ad.mp4"),
            test_mock={
                "create_webhook": lambda: (
                    "test_uuid",
                    "https://webhook.site/test_uuid",
                ),
                "create_video": lambda api_key, payload: {"pid": "test_pid"},
                "wait_for_video": lambda api_key, pid, webhook_token, max_wait_time=1000: "https://example.com/ad.mp4",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs):
        webhook_token, webhook_url = self.create_webhook()

        payload = {
            "webhook": webhook_url,
            "creationParams": {
                "targetDuration": input_data.target_duration,
                "ratio": input_data.ratio,
                "mediaType": "aiVideo",
                "inputText": input_data.script,
                "flowType": "text-to-video",
                "slug": "ai-ad-generator",
                "slugNew": "",
                "isCopiedFrom": False,
                "hasToGenerateVoice": True,
                "hasToTranscript": False,
                "hasToSearchMedia": True,
                "hasAvatar": False,
                "hasWebsiteRecorder": False,
                "hasTextSmallAtBottom": False,
                "selectedAudio": input_data.background_music.value,
                "selectedVoice": input_data.voice.voice_id,
                "selectedAvatar": "https://cdn.revid.ai/avatars/young-woman.mp4",
                "selectedAvatarType": "video/mp4",
                "websiteToRecord": "",
                "hasToGenerateCover": True,
                "nbGenerations": 1,
                "disableCaptions": False,
                "mediaMultiplier": "medium",
                "characters": [],
                "captionPresetName": "Revid",
                "sourceType": "contentScraping",
                "selectedStoryStyle": {"value": "custom", "label": "General"},
                "generationPreset": "DEFAULT",
                "hasToGenerateMusic": False,
                "isOptimizedForChinese": False,
                "generationUserPrompt": "",
                "enableNsfwFilter": False,
                "addStickers": False,
                "typeMovingImageAnim": "dynamic",
                "hasToGenerateSoundEffects": False,
                "forceModelType": "gpt-image-1",
                "selectedCharacters": [],
                "lang": "",
                "voiceSpeed": 1,
                "disableAudio": False,
                "disableVoice": False,
                "useOnlyProvidedMedia": input_data.use_only_provided_media,
                "imageGenerationModel": "ultra",
                "videoGenerationModel": "base",
                "hasEnhancedGeneration": True,
                "hasEnhancedGenerationPro": True,
                "inputMedias": [
                    {"url": url, "title": "", "type": "image"}
                    for url in input_data.input_media_urls
                ],
                "hasToGenerateVideos": True,
                "audioUrl": input_data.background_music.audio_url,
                "watermark": None,
            },
        }

        response = self.create_video(credentials.api_key, payload)
        pid = response.get("pid")
        if not pid:
            raise RuntimeError("Failed to create video: No project ID returned")

        video_url = self.wait_for_video(credentials.api_key, pid, webhook_token)
        yield "video_url", video_url


class AIPromptToVideoCreatorBlock(Block, _RevidMixin):
    """Turns a single creative prompt into a fully AI‑generated video."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REVID], Literal["api_key"]
        ] = CredentialsField(description="Revid.ai API credentials")
        prompt: str = SchemaField(
            description="Imaginative prompt describing the desired video.",
            placeholder="A neon‑lit cyberpunk alley with rain‑soaked pavements.",
        )
        ratio: str = SchemaField(default="9 / 16")
        prompt_target_duration: int = SchemaField(default=30)
        voice: Voice = SchemaField(default=Voice.EVA)
        background_music: AudioTrack = SchemaField(
            default=AudioTrack.DONT_STOP_ME_ABSTRACT_FUTURE_BASS
        )

    class Output(BlockSchema):
        video_url: str = SchemaField(description="Rendered video URL")
        error: str = SchemaField(description="Error message if any")

    def __init__(self):
        super().__init__(
            id="46f4099c-ad01-4d79-874c-37a24c937ba3",
            description="Creates an AI video from a single prompt (no line‑breaking script).",
            categories={BlockCategory.AI, BlockCategory.SOCIAL},
            input_schema=AIPromptToVideoCreatorBlock.Input,
            output_schema=AIPromptToVideoCreatorBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "prompt": "Epic time‑lapse of a city skyline from day to night",
            },
            test_output=("video_url", "https://example.com/prompt.mp4"),
            test_mock={
                "create_webhook": lambda: (
                    "test_uuid",
                    "https://webhook.site/test_uuid",
                ),
                "create_video": lambda api_key, payload: {"pid": "test_pid"},
                "wait_for_video": lambda api_key, pid, webhook_token, max_wait_time=1000: "https://example.com/prompt.mp4",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs):
        webhook_token, webhook_url = self.create_webhook()

        payload = {
            "webhook": webhook_url,
            "creationParams": {
                "mediaType": "aiVideo",
                "flowType": "prompt-to-video",
                "slug": "prompt-to-video",
                "slugNew": "",
                "isCopiedFrom": False,
                "hasToGenerateVoice": True,
                "hasToTranscript": False,
                "hasToSearchMedia": True,
                "hasAvatar": False,
                "hasWebsiteRecorder": False,
                "hasTextSmallAtBottom": False,
                "ratio": input_data.ratio,
                "selectedAudio": input_data.background_music.value,
                "selectedVoice": input_data.voice.voice_id,
                "selectedAvatar": "https://cdn.revid.ai/avatars/young-woman.mp4",
                "selectedAvatarType": "video/mp4",
                "websiteToRecord": "",
                "hasToGenerateCover": True,
                "nbGenerations": 1,
                "disableCaptions": False,
                "characters": [],
                "captionPresetName": "Revid",
                "sourceType": "contentScraping",
                "selectedStoryStyle": {"value": "custom", "label": "General"},
                "generationPreset": "DEFAULT",
                "hasToGenerateMusic": False,
                "isOptimizedForChinese": False,
                "generationUserPrompt": input_data.prompt,
                "enableNsfwFilter": False,
                "addStickers": False,
                "typeMovingImageAnim": "dynamic",
                "hasToGenerateSoundEffects": False,
                "promptTargetDuration": input_data.prompt_target_duration,
                "selectedCharacters": [],
                "lang": "",
                "voiceSpeed": 1,
                "disableAudio": False,
                "disableVoice": False,
                "imageGenerationModel": "good",
                "videoGenerationModel": "base",
                "hasEnhancedGeneration": True,
                "hasEnhancedGenerationPro": True,
                "inputMedias": [],
                "hasToGenerateVideos": True,
                "audioUrl": input_data.background_music.audio_url,
                "watermark": None,
            },
        }

        response = self.create_video(credentials.api_key, payload)
        pid = response.get("pid")
        if not pid:
            raise RuntimeError("Failed to create video: No project ID returned")

        video_url = self.wait_for_video(credentials.api_key, pid, webhook_token)
        yield "video_url", video_url


class AIScreenshotToVideoAdBlock(Block, _RevidMixin):
    """Creates an advert where the supplied screenshot is narrated by an AI avatar."""

    class Input(BlockSchema):
        credentials: CredentialsMetaInput[
            Literal[ProviderName.REVID], Literal["api_key"]
        ] = CredentialsField(description="Revid.ai API key")
        script: str = SchemaField(
            description="Narration that will accompany the screenshot.",
            placeholder="Check out these amazing stats!",
        )
        screenshot_url: str = SchemaField(description="Screenshot or image URL to showcase.")
        ratio: str = SchemaField(default="9 / 16")
        target_duration: int = SchemaField(default=30)
        voice: Voice = SchemaField(default=Voice.EVA)
        background_music: AudioTrack = SchemaField(
            default=AudioTrack.DONT_STOP_ME_ABSTRACT_FUTURE_BASS
        )

    class Output(BlockSchema):
        video_url: str = SchemaField(description="Rendered video URL")
        error: str = SchemaField(description="Error, if encountered")

    def __init__(self):
        super().__init__(
            id="9f68982c-3af6-4923-9a97-b50a8c8d2234",
            description="Turns a screenshot into an engaging, avatar‑narrated video advert.",
            categories={BlockCategory.AI, BlockCategory.MARKETING},
            input_schema=AIScreenshotToVideoAdBlock.Input,
            output_schema=AIScreenshotToVideoAdBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "script": "Amazing numbers!",
                "screenshot_url": "https://cdn.revid.ai/uploads/1747080376028-image.png",
            },
            test_output=("video_url", "https://example.com/screenshot.mp4"),
            test_mock={
                "create_webhook": lambda: (
                    "test_uuid",
                    "https://webhook.site/test_uuid",
                ),
                "create_video": lambda api_key, payload: {"pid": "test_pid"},
                "wait_for_video": lambda api_key, pid, webhook_token, max_wait_time=1000: "https://example.com/screenshot.mp4",
            },
            test_credentials=TEST_CREDENTIALS,
        )

    def run(self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs):
        webhook_token, webhook_url = self.create_webhook()

        payload = {
            "webhook": webhook_url,
            "creationParams": {
                "targetDuration": input_data.target_duration,
                "ratio": input_data.ratio,
                "mediaType": "aiVideo",
                "hasAvatar": True,
                "removeAvatarBackground": True,
                "inputText": input_data.script,
                "flowType": "text-to-video",
                "slug": "ai-ad-generator",
                "slugNew": "screenshot-to-video-ad",
                "isCopiedFrom": "ai-ad-generator",
                "hasToGenerateVoice": True,
                "hasToTranscript": False,
                "hasToSearchMedia": True,
                "hasWebsiteRecorder": False,
                "hasTextSmallAtBottom": False,
                "selectedAudio": input_data.background_music.value,
                "selectedVoice": input_data.voice.voice_id,
                "selectedAvatar": "https://cdn.revid.ai/avatars/young-woman.mp4",
                "selectedAvatarType": "video/mp4",
                "websiteToRecord": "",
                "hasToGenerateCover": True,
                "nbGenerations": 1,
                "disableCaptions": False,
                "mediaMultiplier": "medium",
                "characters": [],
                "captionPresetName": "Revid",
                "sourceType": "contentScraping",
                "selectedStoryStyle": {"value": "custom", "label": "General"},
                "generationPreset": "DEFAULT",
                "hasToGenerateMusic": False,
                "isOptimizedForChinese": False,
                "generationUserPrompt": "",
                "enableNsfwFilter": False,
                "addStickers": False,
                "typeMovingImageAnim": "dynamic",
                "hasToGenerateSoundEffects": False,
                "forceModelType": "gpt-image-1",
                "selectedCharacters": [],
                "lang": "",
                "voiceSpeed": 1,
                "disableAudio": False,
                "disableVoice": False,
                "useOnlyProvidedMedia": True,
                "imageGenerationModel": "ultra",
                "videoGenerationModel": "base",
                "hasEnhancedGeneration": True,
                "hasEnhancedGenerationPro": True,
                "inputMedias": [
                    {"url": input_data.screenshot_url, "title": "", "type": "image"}
                ],
                "hasToGenerateVideos": True,
                "audioUrl": input_data.background_music.audio_url,
                "watermark": None,
            },
        }

        response = self.create_video(credentials.api_key, payload)
        pid = response.get("pid")
        if not pid:
            raise RuntimeError("Failed to create video: No project ID returned")

        video_url = self.wait_for_video(credentials.api_key, pid, webhook_token)
        yield "video_url", video_url
