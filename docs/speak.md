# `speech_utils.py` Module Documentation

This module provides functions to generate speech from text using different text-to-speech engines.

## Functions

### `eleven_labs_speech`

`eleven_labs_speech(text: str, voice_index: int = 0) -> bool`

This function uses the ElevenLabs.io API to convert the given `text` parameter to speech. The `text` parameter must be a string. The `voice_index` parameter indicates the index of the voice to be used. The API returns an audio file in MPEG format which is played using the playsound library. If the API call fails, the function falls back to Google Text-to-Speech (GTTS) and plays the resulting file.

### `gtts_speech`

`gtts_speech(text: str) -> None`

This function uses the GTTS library to convert the given `text` parameter to speech. The `text` parameter must be a string. The resulting file is played using the playsound library.

### `macos_tts_speech`

`macos_tts_speech(text: str) -> None`

This function uses the built-in Text-to-Speech functionality of MacOS to convert the given `text` parameter to speech. The `text` parameter must be a string. The resulting speech is played automatically.

### `say_text`

`say_text(text: str, voice_index: int = 0) -> None`

This function is the main entry point for the module. It takes a `text` parameter and an optional `voice_index` parameter (default is 0) as input. It uses a semaphore to ensure that only a limited number of speech requests are processed concurrently. If the `elevenlabs_api_key` variable is not set in the config.py file, or if its value is `False`, the function falls back to using the GTTS library or the `macos_tts_speech` function to convert the text to speech.