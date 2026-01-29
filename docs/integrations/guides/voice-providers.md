# Voice Providers for D-ID

This guide covers the voice providers you can use with the D-ID Create Talking Avatar Video block.

## ElevenLabs

1. Select any voice from the voice list: [https://api.elevenlabs.io/v1/voices](https://api.elevenlabs.io/v1/voices)
2. Copy the `voice_id`
3. Use it as a string in the `voice_id` field in the CreateTalkingAvatarClip Block

## Microsoft Azure Voices

1. Select any voice from the voice gallery: [https://speech.microsoft.com/portal/voicegallery](https://speech.microsoft.com/portal/voicegallery)
2. Click on the "Sample code" tab on the right
3. Copy the voice name, for example: `config.SpeechSynthesisVoiceName = "en-GB-AbbiNeural"`
4. Use this string `en-GB-AbbiNeural` in the `voice_id` field in the CreateTalkingAvatarClip Block

## Amazon Polly Voices

1. Select any voice from the voice list: [https://docs.aws.amazon.com/polly/latest/dg/available-voices.html](https://docs.aws.amazon.com/polly/latest/dg/available-voices.html)
2. Copy the voice name / ID
3. Use it as string in the `voice_id` field in the CreateTalkingAvatarClip Block
