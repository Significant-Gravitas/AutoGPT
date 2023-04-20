## **Module autogpt.speech.brian**

This module contains the `BrianSpeech` class, which is used to convert the given text into speech.

### **Class**

#### BrianSpeech
```python
class BrianSpeech(VoiceBase):
    def _setup(self) -> None:
        pass
    def _speech(self, text: str, _: int = 0) -> bool:
        pass
```
**Description**

The `BrianSpeech` class extends the `VoiceBase` abstract base class to define the Brian speech module for autogpt.

**Methods**
- `_setup(self)`: This method sets up the voices and authentication for the Brian speech module.
- `_speech(self, text: str, _: int = 0) -> bool`: This method takes the text as input and converts it into speech using the streamelements API.

### **Usage**

```python
from autogpt.speech.brian import BrianSpeech

# Create object of BrianSpeech
brian = BrianSpeech()

# Call speak method to convert text into speech
brian.speak("Hello world")
```

Ensure you have a stable internet connection before running the speak method as it requires the use of the `streamelements` API. The method will return a boolean value indicating if the request was successful or not.