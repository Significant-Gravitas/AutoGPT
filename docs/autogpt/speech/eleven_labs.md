# `ElevenLabsSpeech` class documentation

The `ElevenLabsSpeech` class is a sub-class of `VoiceBase`. This class uses [elevenlabs.io's text-to-speech](https://elevenlabs.io/text-to-speech) API to convert text to speech. Uses the API-Key provided in a config file to authenticate requests.

## Methods

`def _setup(self) -> None:`

Set up the `voice`, `API-Key`, `placeholders` etc. These values are required for `ElevenLabsSpeech` class.

`def _use_custom_voice(self, voice, voice_index) -> None:`

Use a custom voice if available and not a placeholder.

- `voice`: The `voice_ID`.
- `voice_index`: The `voice` index.

`def _speech(self, text: str, voice_index: int = 0) -> bool:`

Speak text using elevenlabs.io's API

- `text`: The `text` to speak
- `voice_index` : The `voice` to use. This parameter is optional and defaults to `0`.
- `Returns`: A boolean value `True` if the request was successful, otherwise `False`.