import type { CallRoomPass } from "@/services/onboarding/voice-call-ticket";
import { type MutableRefObject, useEffect, useRef, useState } from "react";
import { trackInterview } from "../../../../tracking";
import { type CallRoomEvents, joinCallRoom } from "./callRoom";
import { type Connection, leaveRoom, type OpenCall } from "./connection";
import {
  ROOM_JOIN_TIMEOUT_MS,
  ROOM_PLAYOUT_MS,
  secondsBetween,
} from "./helpers";

interface Args {
  connRef: MutableRefObject<Connection>;
  // Another tab rejoined the call, and took its room over.
  onReplaced: (call: OpenCall) => void;
  // LiveKit could not get the call's room back.
  onLost: () => void;
  // LiveKit got the call's audio back after the network changed.
  onReconnected: () => void;
}

// The call's LiveKit room on the LiveKit leg: joining it, what it tells the
// call, and leaving it.
export function useCallRoom(args: Args) {
  const argsRef = useRef(args);
  useEffect(() => {
    argsRef.current = args;
  });
  // LiveKit is getting the call's audio back after the network changed.
  const [isAudioReconnecting, setIsAudioReconnecting] = useState(false);

  // Joins the call's room and keeps it for the call. False, with nothing of
  // the room left, when the call went meanwhile or the join failed, which
  // *onFailure* hears.
  async function enterRoom(
    call: OpenCall,
    pass: CallRoomPass,
    microphone: MediaStream,
    player: HTMLAudioElement,
    onFailure?: (error: unknown) => void,
  ) {
    const conn = argsRef.current.connRef.current;
    try {
      const room = await joinCallRoom({
        pass,
        microphone,
        player,
        timeoutMs: ROOM_JOIN_TIMEOUT_MS,
        events: roomEventsFor(call),
      });
      if (conn.call !== call) {
        room.leave();
        return false;
      }
      conn.room = room;
      return true;
    } catch (error) {
      if (conn.call === call) onFailure?.(error);
      return false;
    }
  }

  // What the call's room tells the call.
  function roomEventsFor(call: OpenCall): CallRoomEvents {
    const conn = argsRef.current.connRef.current;
    let reconnectingSince: number | null = null;
    return {
      onReconnecting() {
        if (conn.call !== call) return;
        reconnectingSince ??= Date.now();
        setIsAudioReconnecting(true);
      },
      onReconnected() {
        if (conn.call !== call) return;
        setIsAudioReconnecting(false);
        if (reconnectingSince !== null) {
          trackInterview("voice_interview_reconnected", {
            gap_s: secondsBetween(reconnectingSince, Date.now()),
            leg: "livekit",
          });
          reconnectingSince = null;
        }
        argsRef.current.onReconnected();
      },
      onOttoTrack(track) {
        if (conn.call === call) conn.audio?.listen(track);
      },
      onLost(why) {
        if (conn.call !== call) return;
        conn.room = null;
        setIsAudioReconnecting(false);
        if (why === "replaced") {
          argsRef.current.onReplaced(call);
          return;
        }
        argsRef.current.onLost();
      },
    };
  }

  // The call's audio takes the socket instead of the room.
  function takeSocketAudio(call: OpenCall) {
    call.leg = "websocket";
    call.room = null;
    leave();
  }

  // The browser leaves the call's room now.
  function leave() {
    leaveRoom(argsRef.current.connRef.current);
    setIsAudioReconnecting(false);
  }

  // On the LiveKit leg Otto's last words are still playing in the room: the
  // microphone leaves it now, and the browser once Otto has.
  function finishRoom() {
    const conn = argsRef.current.connRef.current;
    const room = conn.room;
    conn.room = null;
    room?.mute();
    room?.leaveAfterOtto(ROOM_PLAYOUT_MS);
    setIsAudioReconnecting(false);
  }

  function clearReconnecting() {
    setIsAudioReconnecting(false);
  }

  return {
    isAudioReconnecting,
    enterRoom,
    takeSocketAudio,
    leave,
    finishRoom,
    clearReconnecting,
  };
}
