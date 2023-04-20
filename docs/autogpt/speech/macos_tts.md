# MacOS TTS Voice

This module implements a text-to-speech (TTS) voice for Mac operating system. 

This module defines a class `MacOSTTS` that inherits from `VoiceBase`. The `MacOSTTS` class provides methods to convert text to speech. 

## Class

### MacOSTTS

This is the main class that provides an interface to convert a given text to speech on MacOS. 

```python
MacOSTTS()
```

#### Methods

##### `_setup() -> None`

This method sets up the MacOS specific configurations for the TTS voice. 

##### `_speech(text: str, voice_index: int = 0) -> bool`

This method takes in a string `text` and an optional integer `voice_index` and converts the given `text` to speech. The method returns a boolean value indicating whether the conversion was successful or not. The method supports the following voices:

- voice_index 0 (default): This uses the default voice, which is `Alex`.
- voice_index 1: This uses the premium voice `Ava`.
- voice_index > 1: This uses the voice `Samantha`.

#### Example

```python
tts = MacOSTTS()
tts.speech('Hello World')
tts.speech('Hello World in Ava premium voice', voice_index=1)
tts.speech('Hello World in Samantha voice', voice_index=2)
```