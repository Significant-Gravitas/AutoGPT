# Elevenlabs Speech To Text
<!-- MANUAL: file_description -->
Blocks that run on ElevenLabs' speech models.
<!-- END MANUAL -->

## Transcribe Speech

### What it is
Transcribes speech in an audio or video file and labels who said what. Returns the full transcript, each speaker's segments with timestamps, and the speakers heard

### How it works
<!-- MANUAL: how_it_works -->
The block uploads the file to ElevenLabs Scribe v3 with speaker labelling on, then joins each run of consecutive words by one speaker into a segment. Speakers are named `speaker_0`, `speaker_1` and so on, in the order Scribe tells them apart; the names say which voice, not whose.

On the platform's ElevenLabs key a run costs $0.30 per hour of audio, measured by the length Scribe reports, and at least $0.01.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| audio | Audio or video file to transcribe (URL, data URI or workspace file) | str (file) | Yes |
| language | Spoken language as an ISO 639 code, such as 'en' or 'nl'; leave empty to detect it | str | No |
| max_speakers | Most speakers in the recording, up to 32; leave empty to let the model decide | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| transcript | Everything that was said | str |
| segments | Each stretch of speech by one speaker, with start and end in seconds | List[SpeechSegment] |
| speakers | The speakers heard, in order of first appearance | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
Turn a recorded meeting, interview or podcast into notes that say who said what, then pass the segments to an AI block to summarise each speaker's points.
<!-- END MANUAL -->

---
