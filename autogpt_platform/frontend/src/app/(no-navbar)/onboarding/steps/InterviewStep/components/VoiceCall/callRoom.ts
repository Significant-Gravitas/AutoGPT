import type { CallRoomPass } from "@/services/onboarding/voice-call-ticket";
import type { Room as LiveKitRoom, RemoteTrack } from "livekit-client";
import { releasePlayer } from "./audioGraph";
import { RoomJoinTimeout } from "./helpers";

// A call's LiveKit room, which carries its audio: the browser publishes the
// microphone the call opened, and plays Otto, whom the relay joins as,
// through the player the call's click started. LiveKit reconnects
// on its own when the network changes; the control socket carries the rest.
// The relay closes the room once the call is done with it, removing the
// browser: the socket says what became of the call.

// The relay's identity in every call's room.
const OTTO = "otto";

export interface CallRoomEvents {
  // LiveKit lost the call's media and is getting it back, then has it.
  onReconnecting: () => void;
  onReconnected: () => void;
  // Otto's speech came into the room, and plays through the player.
  onOttoTrack: (track: MediaStreamTrack) => void;
  // The room is gone for good: LiveKit could not get it back, or the call
  // was rejoined from another tab, which takes the room over.
  onLost: (why: "lost" | "replaced") => void;
  // The relay closed the room: nothing is lost, and the socket says what
  // the call became.
  onClosed: () => void;
}

export interface CallRoom {
  // Stops sending the microphone to the room.
  mute(): void;
  // Leaves once Otto has, so its last words play out; at most *ms* later.
  leaveAfterOtto(ms: number): void;
  leave(): void;
  // WebRTC's stats for Otto's latest speech in the room; nothing before he
  // spoke, once the room is left, or when the browser gives none.
  ottoStats(): Promise<RTCStatsReport | undefined>;
}

interface JoinArgs {
  pass: CallRoomPass;
  microphone: MediaStream;
  player: HTMLAudioElement;
  timeoutMs: number;
  events: CallRoomEvents;
}

// The SDK is downloaded only for calls that join a room.
export function loadLiveKit() {
  return import("livekit-client");
}

// Joins a call's room and publishes the microphone in it, all within
// *timeoutMs*; or throws what LiveKit refused or failed with, or
// RoomJoinTimeout, having left the room and let the player go. The
// microphone stays the call's: leaving the room never stops it.
export async function joinCallRoom({
  pass,
  microphone,
  player,
  timeoutMs,
  events,
}: JoinArgs): Promise<CallRoom> {
  const [track] = microphone.getAudioTracks();
  let room: LiveKitRoom | null = null;
  let ottoTrack: RemoteTrack | null = null;
  // Until the join is done, its own failure says what went wrong.
  let isJoined = false;
  let isLeaving = false;
  let leaveTimer: ReturnType<typeof setTimeout> | null = null;

  function leave() {
    if (leaveTimer !== null) clearTimeout(leaveTimer);
    leaveTimer = null;
    if (isLeaving) return;
    isLeaving = true;
    releasePlayer(player);
    void room?.disconnect(false).catch(() => undefined);
  }

  const callRoom: CallRoom = {
    mute() {
      if (isLeaving || !room || !track) return;
      void room.localParticipant
        .unpublishTrack(track, false)
        .catch(() => undefined);
    },
    leaveAfterOtto(ms) {
      if (isLeaving) return;
      if (!room?.remoteParticipants.has(OTTO)) {
        leave();
        return;
      }
      leaveTimer = setTimeout(leave, ms);
    },
    leave,
    async ottoStats() {
      if (isLeaving || !ottoTrack) return undefined;
      return ottoTrack.getRTCStatsReport().catch(() => undefined);
    },
  };

  async function join() {
    const { DisconnectReason, Room, RoomEvent, Track } = await loadLiveKit();
    if (isLeaving) return callRoom;
    const joined = new Room();
    room = joined;
    joined.on(
      RoomEvent.TrackSubscribed,
      (remote, _publication, participant) => {
        if (participant.identity !== OTTO || remote.kind !== Track.Kind.Audio) {
          return;
        }
        ottoTrack = remote;
        remote.attach(player);
        events.onOttoTrack(remote.mediaStreamTrack);
      },
    );
    joined.on(RoomEvent.Reconnecting, () => {
      if (!isLeaving) events.onReconnecting();
    });
    joined.on(RoomEvent.Reconnected, () => {
      if (!isLeaving) events.onReconnected();
    });
    joined.on(RoomEvent.Disconnected, (reason) => {
      if (isLeaving || !isJoined) return;
      // The call is over, and the room closed while Otto's last words played.
      if (leaveTimer !== null) {
        leave();
        return;
      }
      if (
        reason === DisconnectReason.PARTICIPANT_REMOVED ||
        reason === DisconnectReason.ROOM_DELETED
      ) {
        leave();
        events.onClosed();
        return;
      }
      isLeaving = true;
      releasePlayer(player);
      events.onLost(
        reason === DisconnectReason.DUPLICATE_IDENTITY ? "replaced" : "lost",
      );
    });
    joined.on(RoomEvent.ParticipantDisconnected, (participant) => {
      if (participant.identity === OTTO && leaveTimer !== null) leave();
    });
    if (!track) throw new Error("The call has no microphone to publish");
    await joined.connect(pass.url, pass.token);
    // The call gave up on this join while it connected.
    if (isLeaving) return callRoom;
    await joined.localParticipant.publishTrack(track, {
      source: Track.Source.Microphone,
    });
    isJoined = true;
    return callRoom;
  }

  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new RoomJoinTimeout()), timeoutMs);
  });
  const joining = join();
  try {
    return await Promise.race([joining, timeout]);
  } catch (error) {
    // Leaving also stops a connection still under way.
    joining.catch(() => undefined);
    leave();
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
