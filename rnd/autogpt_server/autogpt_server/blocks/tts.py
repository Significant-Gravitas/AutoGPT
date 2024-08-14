from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from pathlib import Path
from openai import OpenAI
from autogpt_server.data.model import BlockSecret, SchemaField, SecretField

class TextToSpeechBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(
            key="openai_api_key",
            description="Your OpenAI API key",
            placeholder="Enter your OpenAI API key",
        )
        text: str = SchemaField(
            description="The text to convert to speech",
            placeholder="Enter the text you want to convert to speech",
        )
        voice: str = SchemaField(
            description="The voice to use for speech synthesis",
            placeholder="alloy",
            default="alloy",
        )
        model: str = SchemaField(
            default="tts-1",
            description="The TTS model to use",
            placeholder="tts-1",
        )
        output_path: str = SchemaField(
            description="The path where the output audio file will be saved",
            placeholder="/path/to/output/speech.mp3",
        )

    class Output(BlockSchema):
        file_path: str = SchemaField(description="The path of the generated audio file")
        file_size: int = SchemaField(description="The size of the generated audio file in bytes")
        duration: float = SchemaField(description="The duration of the generated audio in seconds")
        error: str = SchemaField(description="Error message if the TTS conversion failed")

    def __init__(self):
        super().__init__(
            id="1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
            input_schema=TextToSpeechBlock.Input,
            output_schema=TextToSpeechBlock.Output,
            description="Uses OpenAI to covert the input string into a audio stored as an mp3 in the specified output dir",
            categories=[BlockCategory.AI, BlockCategory.OUTPUT],
            test_input={
                "api_key": "your_test_api_key",
                "text": "Hello, this is a test for text-to-speech conversion.",
                "voice": "alloy",
                "model": "tts-1",
                "output_path": "/tmp/test_speech.mp3",
            },
            test_output=[
                ("file_path", "/tmp/test_speech.mp3"),
                ("file_size", 12345),
                ("duration", 3.5),
            ],
            test_mock={
                "create_speech": lambda *args, **kwargs: MockResponse(),
            },
        )

    def create_speech(self, api_key: str, text: str, voice: str, model: str, output_path: str):
        client = OpenAI(api_key=api_key)
        speech_file_path = Path(output_path)

        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )

        response.stream_to_file(speech_file_path)
        return speech_file_path

    def run(self, input_data: Input) -> BlockOutput:
        try:
            output_file = self.create_speech(
                api_key=input_data.api_key.get_secret_value(),
                text=input_data.text,
                voice=input_data.voice,
                model=input_data.model,
                output_path=input_data.output_path
            )

            file_size = output_file.stat().st_size
            
            # Here we would typically use a library like pydub to get the duration
            # For simplicity, we'll estimate it based on average speech rate
            estimated_duration = len(input_data.text.split()) / 2.5  # Assuming 150 words per minute

            yield "file_path", str(output_file)
            yield "file_size", file_size
            yield "duration", estimated_duration

        except Exception as e:
            yield "error", f"Error occurred during text-to-speech conversion: {str(e)}"

class MockResponse:
    def stream_to_file(self, path):
        # Mock implementation for testing
        with open(path, 'w') as f:
            f.write("Mock audio content")