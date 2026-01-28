# Voice-to-Text Chat Input Feature

## Overview

Add a microphone button to the ChatInput component that allows users to record voice and transcribe it to text using OpenAI's Whisper API, similar to ChatGPT.

## Requirements

- Toggle recording with click (click to start, click to stop)
- Space key triggers recording toggle when input is focused
- Maximum recording duration: 2 minutes
- Use OpenAI Whisper API for transcription
- Frontend API route handles the Whisper call

## Architecture

### Components to Create/Modify

1. **New API route** - `/api/transcribe/route.ts`
   - Accepts audio blob (webm) via POST
   - Calls OpenAI Whisper API (`whisper-1` model)
   - Returns transcribed text

2. **New hook** - `useVoiceRecording.ts`
   - Manages MediaRecorder state
   - Handles start/stop recording
   - Enforces 2-minute max duration
   - Sends audio to transcribe API

3. **Modified component** - `ChatInput.tsx`
   - Add mic button (left of send button)
   - Visual feedback during recording
   - Space key handler for toggle

### Environment Variable

- `OPENAI_API_KEY` (server-side only)

## Hook Design: `useVoiceRecording.ts`

### State
- `isRecording` - boolean for recording state
- `isTranscribing` - boolean for API call in progress
- `error` - string for error messages

### Functions
- `startRecording()` - Request mic permission, start MediaRecorder
- `stopRecording()` - Stop recording, send to API, return text
- `toggleRecording()` - Convenience function for button/space key

### Constraints
- Auto-stop at 2 minutes with timer
- Show elapsed time during recording
- Cleanup on unmount

## UI/UX Design

### Mic Button Placement
- Position: Left of the send button, inside the input wrapper
- Icon: Microphone (Phosphor icons)

### Visual States

| State | Mic Button | Input Area |
|-------|-----------|------------|
| Idle | Gray mic icon | Normal |
| Recording | Red pulsing mic + elapsed time | Red border glow |
| Transcribing | Spinner | Disabled, "Transcribing..." placeholder |
| Error | Mic with warning | Inline error |

### Button Behavior
- Visible when input is empty or has text
- Hidden during streaming (when stop button shows)

### Accessibility
- `aria-label` updates based on state
- Screen reader announcements
- Space key to toggle when focused

## API Route: `POST /api/transcribe`

### Request
- Content-Type: `multipart/form-data`
- Body: `audio` file (webm blob)

### Response
```json
{ "text": "transcribed text here" }
```

### Error Responses
- `400` - No audio file provided
- `401` - Missing API key configuration
- `413` - File too large (> 25MB)
- `500` - Whisper API error

## File Changes Summary

| File | Action |
|------|--------|
| `src/app/api/transcribe/route.ts` | Create |
| `src/components/contextual/Chat/components/ChatInput/useVoiceRecording.ts` | Create |
| `src/components/contextual/Chat/components/ChatInput/ChatInput.tsx` | Modify |
| `src/components/contextual/Chat/components/ChatInput/useChatInput.ts` | Minor modify (space key coordination) |
| `.env.default` | Add OPENAI_API_KEY placeholder |
