from llama_hub.youtube_transcript import YoutubeTranscriptReader

from AFAAS.core.tools.tool_decorator import tool

@tool(
    name="youtube_transcript",
    description="Provide a Transcript of a Youtube Video.",
    parameters = {
        "youtube_url": {
            "type": "string",
            "description": "URL of a Youtube Video.",
            "required": True
        },
    }
)
def youtube_transcript(youtube_url: str):  
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(ytlinks=[youtube_url])

    return [
  doc.to_langchain_format()
  for doc in documents
]
