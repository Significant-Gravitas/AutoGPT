import type { VoiceCallCaps } from "@/app/api/__generated__/models/voiceCallCaps";

// The voice relay's control socket (backend/voice_relay/session.py). The
// relay owns every control of a call; the browser only shows them.

export type CallLeg = "websocket" | "livekit";

export type ClientMessage =
  | { type: "start"; token: string; leg: CallLeg }
  | { type: "resume"; ticket: string }
  | { type: "hangup" }
  | { type: "ping"; id: number | string };

export type RelayMessage =
  | {
      type: "ready";
      call_id: string;
      caps: VoiceCallCaps;
      resume: string;
      leg: CallLeg;
    }
  | { type: "leg"; leg: CallLeg }
  | { type: "user_text"; text: string }
  | { type: "agent_text"; text: string }
  | { type: "interrupted" }
  | { type: "turn_complete"; turn: number; turns_left: number }
  | { type: "wrap_up"; reason: string }
  | { type: "resumed"; gap_s: number }
  | { type: "ended"; reason: string; elapsed_s: number }
  | { type: "error"; error: string; call_id?: string | null }
  | { type: "pong"; id: number | string | null };

export type RelayMessageType = RelayMessage["type"];

// Why the relay closes a socket, besides a normal end (1000) and a server
// going away for a restart (1012).
export const CLOSE_POLICY = 1008;
export const CLOSE_TRY_AGAIN = 1013;
export const CLOSE_NO_CALL = 4404;
// The relay closes a socket a re-attach replaced with this.
export const CLOSE_REPLACED = 4000;

// The relay's own refusals, sent as `error` before it closes the socket.
export const CALL_IN_PROGRESS = "call_in_progress";

// The person's audio goes up as 16 kHz PCM16 in 40 ms frames, and Otto's
// comes down as 24 kHz PCM16.
export const MIC_RATE = 16_000;
export const MIC_FRAME_MS = 40;
export const PLAYBACK_RATE = 24_000;
