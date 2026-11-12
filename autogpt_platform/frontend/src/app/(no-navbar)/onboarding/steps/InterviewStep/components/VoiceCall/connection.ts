import {
  endOnboardingVoiceCall,
  getOnboardingVoiceCallStatus,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { VoiceCallCaps } from "@/app/api/__generated__/models/voiceCallCaps";
import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import type { CallRoomPass } from "@/services/onboarding/voice-call-ticket";
import { refusalOf } from "../../helpers";
import type { CallAudio, MicrophoneSettings } from "./audioGraph";
import type { CallRoom } from "./callRoom";
import type { CallMarks } from "./callTimings";
import { END_POLL_MS, END_WAIT_MS } from "./helpers";
import type { OttoListening } from "./ottoStats";

// The plumbing under useVoiceCall: what it keeps between renders without
// rendering on it, and the reads and requests a call's lifecycle makes.

// The call this page holds a socket for.
export interface OpenCall {
  callId: string;
  relayUrl: string;
  caps: VoiceCallCaps;
  // Set by `ready`, or from the kept ticket on a rejoin.
  ticket: string | null;
  startedAt: number | null;
  turnsLeft: number;
  turns: number;
  // The call's room, which carries its audio, and this browser's token for
  // it; the socket carries the rest.
  room: CallRoomPass;
  // How far a call this page started has got; null on a rejoin.
  marks: CallMarks | null;
  // How Otto's speech has fared in the call's room on this page.
  listening: OttoListening;
  // What the browser applied to this page's microphone for the call; null
  // while the browser still asks the person for it.
  microphone: MicrophoneSettings | null;
  // The relay gave the call up while the browser still asked for the
  // microphone; the grant starts a fresh one in its place.
  isLapsed: boolean;
}

// What the hook keeps between renders without rendering on it: the socket,
// the audio, and counters that stop a superseded start or poll.
export interface Connection {
  call: OpenCall | null;
  // The call whose ending is shown or awaited.
  shownCallId: string | null;
  // The server named that call as the one that took the interview.
  taken: boolean;
  socket: WebSocket | null;
  audio: CallAudio | null;
  // The call's LiveKit room, while the browser is in it.
  room: CallRoom | null;
  isReady: boolean;
  relayError: { error: string; callId: string | null } | null;
  startRetries: number;
  drop: number;
  dropDeadline: number | null;
  attempt: number;
  poll: number;
  isNewTurn: boolean;
  // The person pressed Hang up and the relay has not yet ended the call.
  isHangingUp: boolean;
  isLeaving: boolean;
  hasChecked: boolean;
  // When the live call's socket last dropped (epoch ms).
  droppedAt: number | null;
  // A call lost on the way is reported ended once its end is known.
  lostCall: {
    turns: number;
    startedAt: number | null;
    droppedAt: number;
    // What ended it, once a read of the call says.
    reason: string | null;
    listening: OttoListening;
  } | null;
  timers: Set<ReturnType<typeof setTimeout>>;
}

export function newConnection(): Connection {
  return {
    call: null,
    shownCallId: null,
    taken: false,
    socket: null,
    audio: null,
    room: null,
    isReady: false,
    relayError: null,
    startRetries: 0,
    drop: 0,
    dropDeadline: null,
    attempt: 0,
    poll: 0,
    isNewTurn: true,
    isHangingUp: false,
    isLeaving: false,
    hasChecked: false,
    droppedAt: null,
    lostCall: null,
    timers: new Set(),
  };
}

// The page is going: the socket closes, and the room is left, without a word
// to the relay, which holds the call for a re-attach and then ends it as the
// client gone.
export function release(conn: Connection) {
  conn.attempt += 1;
  conn.poll += 1;
  // A remount reads the latest call again.
  conn.hasChecked = false;
  conn.call = null;
  clearTimers(conn);
  closeSocket(conn);
  leaveRoom(conn);
  conn.audio?.close();
  conn.audio = null;
}

export function leaveRoom(conn: Connection) {
  const room = conn.room;
  conn.room = null;
  room?.leave();
}

export function later(conn: Connection, run: () => void, ms: number) {
  const timer = setTimeout(() => {
    conn.timers.delete(timer);
    run();
  }, ms);
  conn.timers.add(timer);
}

export function clearTimers(conn: Connection) {
  conn.timers.forEach((timer) => clearTimeout(timer));
  conn.timers.clear();
}

export function closeSocket(conn: Connection) {
  const socket = conn.socket;
  conn.socket = null;
  if (
    socket &&
    socket.readyState !== WebSocket.CLOSING &&
    socket.readyState !== WebSocket.CLOSED
  ) {
    socket.close();
  }
}

export function closeContext(context: AudioContext) {
  void context.close().catch(() => undefined);
}

export function stopTracks(stream: MediaStream) {
  stream.getTracks().forEach((track) => track.stop());
}

export function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

export async function readStatus(
  callId: string,
): Promise<VoiceCallStatusResponse | "missing" | null> {
  try {
    const response = await getOnboardingVoiceCallStatus(callId);
    return response.status === 200 ? response.data : null;
  } catch (error) {
    return refusalOf(error).status === 404 ? "missing" : null;
  }
}

// Asks the relay to end a call, and waits until it has left the live state,
// which frees the person to start again.
export async function endCall(callId: string) {
  try {
    await endOnboardingVoiceCall(callId);
  } catch {
    // Over already, or unreadable for now: the start that follows answers.
  }
  const deadline = Date.now() + END_WAIT_MS;
  while (Date.now() < deadline) {
    const call = await readStatus(callId);
    if (
      call === "missing" ||
      (call && call.status !== "connecting" && call.status !== "live")
    ) {
      return;
    }
    await sleep(END_POLL_MS);
  }
}
