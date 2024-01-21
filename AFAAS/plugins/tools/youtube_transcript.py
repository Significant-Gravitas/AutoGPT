from __future__ import annotations

from llama_hub.youtube_transcript import YoutubeTranscriptReader

from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

try:
    import youtube_transcript_api
except ImportError:
    import subprocess

    LOG.info("youtube_transcript_api package is not installed. Installing...")
    subprocess.run(["pip", "install", "youtube_transcript_api"])
    LOG.info("youtube_transcript_api package has been installed.")


@tool(
    name="youtube_transcript",
    description="Provide a transcript of a Youtube Video.",
    parameters={
        "youtube_url": {
            "type": "string",
            "description": "URL of a Youtube Video.",
            "required": True,
        },
    },
    categories=["youtube", "video"],
)
def youtube_transcript(youtube_url: str):
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[youtube_url])

    return [doc.to_langchain_format() for doc in documents]
