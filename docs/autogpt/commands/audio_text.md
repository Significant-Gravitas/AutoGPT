# `read_audio_from_file` function

Converts audio to text for a given filename.

## Arguments
- `filename` (str): The filename of the audio file.

## Returns
- `str`: The text from the audio.


## Example usage
```python
>>> from autogpt.commands.audio import read_audio_from_file
>>> print(read_audio_from_file("audio.wav"))
"The audio says: Hello world"
```

# `read_audio` function

Converts audio to text for given raw audio bytes.

## Arguments
- `audio` (bytes): The raw audio bytes.

## Returns
- `str`: The text from the audio.


## Example usage
```python
>>> from autogpt.commands.audio import read_audio
>>> with open("audio.wav", "rb") as f:
...     audio = f.read()
... 
>>> print(read_audio(audio))
"The audio says: Hello world"
```