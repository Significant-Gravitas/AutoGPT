# Text to Speech Module

This module provides a way to convert a given text message into an audible speech. It uses different voice engines depending on the configuration set by the user. The following speech engines are available:

- GTTSVoice: Uses Google's Text-to-Speech API to synthesize speech
- ElevenLabsSpeech: Uses the 11labs text-to-speech API to synthesize speech
- MacOSTTS: Uses the built-in text-to-speech system on macOS
- BrianSpeech: Uses the Brian Speech Synthesis engine to synthesize speech

This module provides the following functions:

## Function: say_text(text: str, voice_index: int = 0) -> None

This function speaks the given text using the given voice index. It takes the following parameters:
- text (str): The text to be converted to speech
- voice_index (int): The index of the voice to be used (default is 0, which selects the default voice)

Example usage:
```python
from tts import say_text
say_text("Hello World!")
```

In the above example, the text "Hello World!" will be converted to speech using the default voice engine.