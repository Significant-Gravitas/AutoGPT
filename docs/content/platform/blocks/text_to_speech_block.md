## Unreal Text to Speech

### What it is
A block that converts text into speech using the Unreal Speech API.

### What it does
This block takes a text input and generates an audio file of that text being spoken. It allows users to specify the voice they want to use for the speech conversion.

### How it works
The block sends the provided text and voice selection to the Unreal Speech API. The API processes this information and returns a URL where the generated audio file can be accessed.

### Inputs
| Input | Description |
|-------|-------------|
| Text | The text you want to convert into speech. This could be a sentence, paragraph, or any written content you'd like to hear spoken aloud. |
| Voice ID | The identifier for the voice you want to use for the speech. By default, it uses a voice called "Scarlett," but you can change this to other available voices. |
| API Key | Your personal key to access the Unreal Speech API. This is kept secret and secure. |

### Outputs
| Output | Description |
|--------|-------------|
| MP3 URL | The web address where you can access or download the generated audio file in MP3 format. |
| Error | If something goes wrong during the process, this will contain a message explaining what happened. |

### Possible use case
This block could be used in an application that helps visually impaired users consume written content. For example, a news app could use this block to convert articles into audio format, allowing users to listen to the news instead of reading it.