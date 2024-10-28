# Find available voices for D-ID

1. **ElevenLabs**
   - Select any voice from the voice list: https://api.elevenlabs.io/v1/voices
   - Copy the voice_id
   - Use it as a string in the voice_id field in the CreateTalkingAvatarClip Block

2. **Microsoft Azure Voices**
    - Select any voice from the voice gallery: https://speech.microsoft.com/portal/voicegallery
    - Click on the "Sample code" tab on the right
    - Copy the voice name, for example: config.SpeechSynthesisVoiceName ="en-GB-AbbiNeural"
    - Use this string en-GB-AbbiNeural in the voice_id field in the CreateTalkingAvatarClip Block

3. **Amazon Polly Voices**
    - Select any voice from the voice list: https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
    - Copy the voice name / ID
    - Use it as string in the voice_id field in the CreateTalkingAvatarClip Block