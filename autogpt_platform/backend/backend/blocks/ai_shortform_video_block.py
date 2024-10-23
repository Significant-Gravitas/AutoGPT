import logging
import time
from enum import Enum

import requests
from pydantic import Field

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import BlockSecret, SchemaField, SecretField


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


class Voice(str, Enum):
    LILY = "Lily"
    DANIEL = "Daniel"
    BRIAN = "Brian"
    JESSICA = "Jessica"
    CHARLOTTE = "Charlotte"
    CALLUM = "Callum"

    @property
    def voice_id(self):
        voice_id_map = {
            Voice.LILY: "pFZP5JQG7iQjIQuC4Bku",
            Voice.DANIEL: "onwK4e9ZLuTAKqWW03F9",
            Voice.BRIAN: "nPczCjzI2devNBz1zQrb",
            Voice.JESSICA: "cgSgspJ2msm6clMCkdW9",
            Voice.CHARLOTTE: "XB0fDUnXU5powFXDhCwa",
            Voice.CALLUM: "N2lVS1w4EtoT3dr4eOWO",
        }
        return voice_id_map[self]

    def __str__(self):
        return self.value


class VisualMediaType(str, Enum):
    STOCK_VIDEOS = ("stockVideo",)
    MOVING_AI_IMAGES = ("movingImage",)
    AI_VIDEO = ("aiVideo",)


logger = logging.getLogger(__name__)


class AIShortformVideoCreatorBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="revid_api_key",
            description="Your revid.ai API key",
            placeholder="Enter your revid.ai API key",
        )
        script: str = SchemaField(
            description="""1. Use short and punctuated sentences\n\n2. Use linebreaks to create a new clip\n\n3. Text outside of brackets is spoken by the AI, and [text between brackets] will be used to guide the visual generation. For example, [close-up of a cat] will show a close-up of a cat.""",
            placeholder="[close-up of a cat] Meow!",
        )
        ratio: str = Field(description="Aspect ratio of the video", default="9 / 16")
        resolution: str = Field(description="Resolution of the video", default="720p")
        frame_rate: int = Field(description="Frame rate of the video", default=60)
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
        video_url: str = Field(description="The URL of the created video")
        error: str = Field(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="361697fb-0c4f-4feb-aed3-8320c88c771b",
            description="Creates a shortform video using revid.ai",
            categories={BlockCategory.SOCIAL, BlockCategory.AI},
            input_schema=AIShortformVideoCreatorBlock.Input,
            output_schema=AIShortformVideoCreatorBlock.Output,
            test_input={
                "api_key": "test_api_key",
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
        )

    def create_webhook(self):
        url = "https://webhook.site/token"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        webhook_data = response.json()
        return webhook_data["uuid"], f"https://webhook.site/{webhook_data['uuid']}"

    def create_video(self, api_key: str, payload: dict) -> dict:
        url = "https://www.revid.ai/api/public/v2/render"
        headers = {"key": api_key}
        response = requests.post(url, json=payload, headers=headers)
        logger.debug(
            f"API Response Status Code: {response.status_code}, Content: {response.text}"
        )
        response.raise_for_status()
        return response.json()

    def check_video_status(self, api_key: str, pid: str) -> dict:
        url = f"https://www.revid.ai/api/public/v2/status?pid={pid}"
        headers = {"key": api_key}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def wait_for_video(
        self, api_key: str, pid: str, webhook_token: str, max_wait_time: int = 1000
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

            time.sleep(10)

        logger.error("Video creation timed out")
        raise TimeoutError("Video creation timed out")

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Create a new Webhook.site URL
        webhook_token, webhook_url = self.create_webhook()
        logger.debug(f"Webhook URL: {webhook_url}")

        audio_url = input_data.background_music.audio_url

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
                "selectedAudio": input_data.background_music,
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
                "hasToGenerateVideos": input_data.video_style
                != VisualMediaType.STOCK_VIDEOS,
                "audioUrl": audio_url,
            },
        }

        logger.debug("Creating video...")
        response = self.create_video(input_data.api_key.get_secret_value(), payload)
        pid = response.get("pid")

        if not pid:
            logger.error(
                f"Failed to create video: No project ID returned. API Response: {response}"
            )
            raise RuntimeError("Failed to create video: No project ID returned")
        else:
            logger.debug(
                f"Video created with project ID: {pid}. Waiting for completion..."
            )
            video_url = self.wait_for_video(
                input_data.api_key.get_secret_value(), pid, webhook_token
            )
            logger.debug(f"Video ready: {video_url}")
            yield "video_url", video_url
