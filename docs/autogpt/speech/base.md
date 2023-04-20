## VoiceBase class

The VoiceBase class is the base class for all voice classes. All the other voice classes should derive from this class to implement text to speech functionality.

### Attributes

- `_url`: a string representing the URL of the API needed for text-to-speech conversion.
- `_headers`: a dictionary containing any specific headers need to be sent along with the requests made to the API.
- `_api_key`: a string representing the API key must be sent along with the requests made to the API.
- `_voices`: a list of strings containing the names of all the available voices.
- `_mutex`: a lock to protect concurrent access to the text-to-speech engine.

### Methods

- `__init__()`: initializes the voice class by creating instance variables.
- `say(text: str, voice_index: int = 0) -> bool`: abstract method that will say the given text using the specified voice.
- `_setup() -> None`: abstract method to setup any necessary pre-requisites like voices, API key, etc.
- `_speech(text: str, voice_index: int = 0) -> bool`: abstract method that will play the given text using the specified voice.

### Example

```python
class MyVoice(VoiceBase):
    def __init__(self):
        self._url = "http://myapi.com/text-to-speech"
        self._headers = {"Content-Type": "application/json"}
        self._api_key = "my_api_key"
        super().__init__()

    def _setup(self) -> None:
        # Setup my voice engine
        pass

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        # Speak text using my API
        pass
```

In the above example, the `MyVoice` class extends the `VoiceBase` class, defining some class variables and implementing the `_setup()` and `_speech()` methods to perform text to speech operations. The `__init__()` method of `VoiceBase` is also called using `super()` to initialize the voice class.