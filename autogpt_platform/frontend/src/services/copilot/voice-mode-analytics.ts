// Voice-mode funnel. The question this has to answer is whether anyone uses
// it twice: turns per session, and every way the loop can end badly.
// Capture is best-effort — a blocked analytics host must never interrupt
// someone mid-sentence.
//
// Lives in services/ alongside the brain-dump funnel because both measure
// the same microphone path from different ends.

import posthog from "posthog-js";

type VoiceModeEvent =
  // Enabled from the composer. `entry` says whether a chat already existed,
  // since starting one costs a session-creation round trip first.
  | "voice_mode_started"
  // Left deliberately: the toggle, or Stop during a reply.
  | "voice_mode_stopped"
  // Closed by the silence timeout rather than by the user. A high share
  // here means the timeout is too short, which is exactly the complaint
  // that moved the VAD from 700 ms to 1540 ms.
  | "voice_mode_timed_out"
  // A completed turn: heard, transcribed, sent. `turn_index` counts within
  // the session, so a histogram shows whether anyone gets past one.
  | "voice_turn_sent"
  // Heard something and threw it away — VAD misfire, or filler/hallucination
  // like Whisper's "Thank you." on silence. Splits by `reason`.
  | "voice_turn_dropped"
  // Speech end to transcript in hand. The spike measured 1.05 s on a
  // streaming session; this route uploads the whole clip, so it is the
  // number that decides whether streaming STT is worth building.
  | "voice_transcribe_latency_ms"
  // Speech end to first audio out — what the user experiences as "did it
  // hear me". Includes the canned acknowledgement.
  | "voice_first_sound_latency_ms"
  // The mic reopened after a reply finished playing: a full loop closed.
  | "voice_turn_completed"
  | "voice_mode_permission_denied"
  // Synthesis or the VAD failed. `stage` says which.
  | "voice_mode_error";

export function trackVoiceMode(
  event: VoiceModeEvent,
  properties?: Record<string, unknown>,
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a broken conversation.
  }
}
