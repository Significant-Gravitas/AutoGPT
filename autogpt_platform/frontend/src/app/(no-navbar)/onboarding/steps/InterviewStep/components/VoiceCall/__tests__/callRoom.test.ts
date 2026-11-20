import {
  DISCONNECT_REASON,
  fakeMediaStream,
  installVoiceFakes,
  type VoiceFakes,
} from "@/tests/integrations/voice-fakes";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { type CallRoomEvents, joinCallRoom } from "../callRoom";
import { RoomJoinTimeout } from "../helpers";

vi.mock(
  "livekit-client",
  async () => (await import("@/tests/integrations/voice-fakes")).fakeLiveKit,
);

const PASS = { url: "wss://agpt-test.livekit.cloud", token: "room-token" };

let fakes: VoiceFakes;

beforeEach(() => {
  fakes = installVoiceFakes();
});

afterEach(() => {
  fakes.uninstall();
});

// What the room told the call, in order.
function listener() {
  const told: string[] = [];
  const events: CallRoomEvents = {
    onReconnecting: () => told.push("reconnecting"),
    onReconnected: () => told.push("reconnected"),
    onOttoTrack: () => told.push("otto"),
    onLost: (why) => told.push(`lost:${why}`),
    onClosed: () => told.push("closed"),
  };
  return { told, events };
}

function join({
  microphone = fakeMediaStream(),
  player = new Audio(),
  timeoutMs = 1_000,
}: {
  microphone?: MediaStream;
  player?: HTMLAudioElement;
  timeoutMs?: number;
} = {}) {
  const { told, events } = listener();
  const joining = joinCallRoom({
    pass: PASS,
    microphone,
    player,
    timeoutMs,
    events,
  });
  return { joining, told, microphone, player };
}

async function joined() {
  const call = join();
  const room = await call.joining;
  return { ...call, room, fake: fakes.room()! };
}

describe("the call's LiveKit room", () => {
  it("joins the room the start named, publishing the call's microphone as a microphone", async () => {
    const { microphone, fake } = await joined();

    expect(fake.url).toBe(PASS.url);
    expect(fake.token).toBe(PASS.token);
    expect(fake.published).toEqual([microphone.getAudioTracks()[0]]);
    expect(fake.publishOptions).toEqual([{ source: "microphone" }]);
  });

  it("plays only Otto's speech through the player", async () => {
    const { fake, player, told } = await joined();

    fake.trackFrom("user:someone-else");
    fake.trackFrom("otto", "video");
    expect(told).toEqual([]);
    const otto = fake.ottoJoins();

    expect(otto.attachedTo).toEqual([player]);
    expect(told).toEqual(["otto"]);
  });

  it("throws what LiveKit refused the join with, having left the room and let the player go", async () => {
    const refusal = { name: "ConnectionError", reasonName: "NotAllowed" };
    fakes.refuseRoomJoins(refusal);
    const { joining, player } = join();

    await expect(joining).rejects.toBe(refusal);
    expect(fakes.room()?.isLeft).toBe(true);
    expect(player.srcObject).toBeNull();
  });

  it("gives up a join that does not answer in time", async () => {
    fakes.hangRoomJoins();
    const { joining } = join({ timeoutMs: 20 });

    await expect(joining).rejects.toBeInstanceOf(RoomJoinTimeout);
    expect(fakes.room()?.isLeft).toBe(true);
  });

  it("publishes nothing into a room whose join it gave up on", async () => {
    const letIn = fakes.holdRoomJoins();
    const { joining } = join({ timeoutMs: 20 });
    await expect(joining).rejects.toBeInstanceOf(RoomJoinTimeout);

    letIn();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(fakes.room()?.published).toEqual([]);
    expect(fakes.room()?.isLeft).toBe(true);
  });

  it("joins no room with no microphone to publish", async () => {
    const silent = { getAudioTracks: () => [] } as unknown as MediaStream;
    const { joining } = join({ microphone: silent });

    await expect(joining).rejects.toThrow("no microphone");
    expect(fakes.room()?.published).toEqual([]);
  });

  it("tells the call while LiveKit gets the audio back", async () => {
    const { fake, told } = await joined();

    fake.reconnecting();
    fake.reconnected();

    expect(told).toEqual(["reconnecting", "reconnected"]);
  });

  it("tells the call its room is lost, and lets the player go", async () => {
    const { fake, told, player } = await joined();

    fake.lose(DISCONNECT_REASON.SIGNAL_CLOSE);

    expect(told).toEqual(["lost:lost"]);
    expect(player.srcObject).toBeNull();
  });

  it.each([
    ["removing the browser from it", DISCONNECT_REASON.PARTICIPANT_REMOVED],
    ["deleting it", DISCONNECT_REASON.ROOM_DELETED],
  ])(
    "leaves quietly when the relay closes the room by %s",
    async (_case, reason) => {
      const { room, fake, told, player } = await joined();
      const pause = vi.spyOn(player, "pause");

      fake.lose(reason);

      expect(told).toEqual(["closed"]);
      expect(fake.isLeft).toBe(true);
      expect(pause).toHaveBeenCalled();
      room.mute();
      expect(fake.unpublished).toEqual([]);
    },
  );

  it("tells the call another tab took its room over", async () => {
    const { fake, told } = await joined();

    fake.lose(DISCONNECT_REASON.DUPLICATE_IDENTITY);

    expect(told).toEqual(["lost:replaced"]);
  });

  it("takes the microphone out of the room", async () => {
    const { room, fake, microphone } = await joined();

    room.mute();

    expect(fake.unpublished).toEqual([microphone.getAudioTracks()[0]]);
  });

  it("lets go quietly of a microphone the room can no longer take back", async () => {
    const { room, fake } = await joined();
    const unpublish = vi.fn(async () => {
      throw new Error("The room is gone");
    });
    fake.localParticipant.unpublishTrack = unpublish;

    room.mute();

    await vi.waitFor(() => expect(unpublish).toHaveBeenCalledOnce());
    await expect(unpublish.mock.results[0].value).rejects.toThrow();
  });

  it("hears nothing more from a room it has left", async () => {
    const { room, fake, told } = await joined();

    room.leave();
    fake.reconnecting();
    fake.reconnected();
    fake.lose();
    room.mute();
    room.leaveAfterOtto(10_000);

    expect(fake.isLeft).toBe(true);
    expect(told).toEqual([]);
    expect(fake.unpublished).toEqual([]);
  });

  it("leaves at once when the call ends with Otto already gone", async () => {
    const { room, fake } = await joined();

    room.leaveAfterOtto(10_000);

    expect(fake.isLeft).toBe(true);
  });

  it("leaves without a word when the room closes while Otto's last words play", async () => {
    const { room, fake, told } = await joined();
    fake.ottoJoins();

    room.leaveAfterOtto(10_000);
    expect(fake.isLeft).toBe(false);
    fake.lose(DISCONNECT_REASON.SIGNAL_CLOSE);

    expect(fake.isLeft).toBe(true);
    expect(told).toEqual(["otto"]);
  });
});
