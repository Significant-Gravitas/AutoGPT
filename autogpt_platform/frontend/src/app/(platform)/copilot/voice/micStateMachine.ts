/**
 * The voice-mode loop as a pure reducer.
 *
 * The mic is open in exactly two states, and never while AutoPilot speaks —
 * that is what removes echo without acoustic cancellation, and it is the
 * invariant the rest of the feature leans on.
 */

export type VoiceState =
  | "off"
  | "listening"
  | "hearing"
  | "transcribing"
  | "thinking"
  | "speaking";

export type VoiceEvent =
  | { type: "ENABLE" }
  | { type: "DISABLE" }
  | { type: "SPEECH_START" }
  | { type: "SPEECH_END" }
  /** VAD fired but the utterance was too short to be speech. */
  | { type: "SPEECH_MISFIRE" }
  | { type: "TRANSCRIPT_SENT" }
  /** Transcript was empty, filler, or the send failed. */
  | { type: "TRANSCRIPT_DROPPED" }
  | { type: "REPLY_SPEAKING" }
  /** The reply finished with nothing left to say. */
  | { type: "REPLY_DONE" }
  | { type: "ERROR" };

export function voiceReduce(state: VoiceState, event: VoiceEvent): VoiceState {
  if (event.type === "DISABLE") return "off";
  if (state === "off") return event.type === "ENABLE" ? "listening" : "off";

  switch (event.type) {
    case "SPEECH_START":
      return state === "listening" ? "hearing" : state;
    case "SPEECH_END":
      return state === "hearing" ? "transcribing" : state;
    case "SPEECH_MISFIRE":
      return state === "hearing" ? "listening" : state;
    case "TRANSCRIPT_SENT":
      return state === "transcribing" ? "thinking" : state;
    case "TRANSCRIPT_DROPPED":
      return state === "transcribing" ? "listening" : state;
    case "REPLY_SPEAKING":
      return state === "thinking" || state === "speaking" ? "speaking" : state;
    case "REPLY_DONE":
      return state === "thinking" || state === "speaking" ? "listening" : state;
    case "ERROR":
      return "listening";
    default:
      return state;
  }
}

export function isMicOpen(state: VoiceState): boolean {
  return state === "listening" || state === "hearing";
}

export function describeVoiceState(state: VoiceState): string {
  switch (state) {
    case "listening":
      return "Listening";
    case "hearing":
      return "Listening";
    case "transcribing":
      return "Transcribing";
    case "thinking":
      return "Thinking";
    case "speaking":
      return "Speaking";
    case "off":
      return "Voice mode off";
  }
}
