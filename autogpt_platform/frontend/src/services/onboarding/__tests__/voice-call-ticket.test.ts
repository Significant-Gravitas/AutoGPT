import { afterEach, describe, expect, it } from "vitest";
import {
  type CallTicket,
  forgetCallTicket,
  forgetCallTickets,
  isCallId,
  isCallRoomPass,
  isVoiceCallCaps,
  readCallTicket,
  saveCallTicket,
} from "../voice-call-ticket";

const USER = "ticket-user";
const KEY = `autogpt:onboarding-voice-call:${USER}`;
const CALL_ID = "c".repeat(32);
const ROOM = { url: "wss://agpt.livekit.cloud", token: "room-token" };
const CAPS = {
  max_call_s: 300,
  soft_wrap_s: 30,
  max_agent_turns: 16,
  soft_wrap_turns: 2,
  idle_s: 20,
};

function ticket(overrides: Partial<CallTicket> = {}): CallTicket {
  return {
    callId: CALL_ID,
    ticket: "resume-ticket",
    caps: CAPS,
    startedAt: Date.now(),
    turnsLeft: 9,
    roomToken: ROOM.token,
    ...overrides,
  };
}

function keep(value: unknown) {
  sessionStorage.setItem(KEY, JSON.stringify(value));
}

afterEach(() => sessionStorage.clear());

describe("the call's ticket, kept for a reload", () => {
  it("gives the ticket back for its own call", () => {
    saveCallTicket(USER, ticket());

    expect(readCallTicket(USER, CALL_ID)).toEqual(
      ticket({ startedAt: expect.any(Number) }),
    );
  });

  it("keeps no address to rejoin on: the server names the relay's socket and the room's project", () => {
    saveCallTicket(USER, ticket());

    expect(
      Object.keys(JSON.parse(sessionStorage.getItem(KEY) ?? "{}")),
    ).toEqual([
      "callId",
      "ticket",
      "caps",
      "startedAt",
      "turnsLeft",
      "roomToken",
    ]);
  });

  it("keeps this browser's token for the call's room, so a reload joins it again", () => {
    saveCallTicket(USER, ticket());

    expect(readCallTicket(USER, CALL_ID)?.roomToken).toBe(ROOM.token);
  });

  it("gives back no room address stored beside the ticket", () => {
    keep({
      ...ticket(),
      room: ROOM,
      roomUrl: ROOM.url,
    });

    const kept = readCallTicket(USER, CALL_ID);

    expect(kept).toEqual(ticket({ startedAt: expect.any(Number) }));
    expect(JSON.stringify(kept)).not.toContain(ROOM.url);
  });

  it("gives nothing for another call or another person, and keeps the ticket", () => {
    saveCallTicket(USER, ticket());

    expect(readCallTicket(USER, "d".repeat(32))).toBeNull();
    expect(readCallTicket("someone-else", CALL_ID)).toBeNull();
    expect(sessionStorage.getItem(KEY)).not.toBeNull();
  });

  it("forgets a ticket that has outlived its call", () => {
    saveCallTicket(USER, ticket({ startedAt: Date.now() - 361_000 }));

    expect(readCallTicket(USER, CALL_ID)).toBeNull();
    expect(sessionStorage.getItem(KEY)).toBeNull();
  });

  it.each([
    ["text that is not JSON", "{oops"],
    ["caps of the wrong shape", { ...ticket(), caps: { max_call_s: "300" } }],
    ["caps missing", { ...ticket(), caps: undefined }],
    ["a call id that is not one", { ...ticket(), callId: "../../calls" }],
    ["an empty ticket", { ...ticket(), ticket: "" }],
    ["a start time that is not a time", { ...ticket(), startedAt: "now" }],
    // Its audio is in a room this tab could not join again.
    ["no room token", { ...ticket(), roomToken: undefined }],
    ["a room token kept as none", { ...ticket(), roomToken: null }],
    ["an empty room token", { ...ticket(), roomToken: "" }],
    ["a room token that is not text", { ...ticket(), roomToken: 42 }],
    [
      "a room token too long to be one",
      { ...ticket(), roomToken: "t".repeat(4_097) },
    ],
  ])("forgets a ticket with %s, without throwing", (_case, value) => {
    if (typeof value === "string") sessionStorage.setItem(KEY, value);
    else keep(value);

    expect(readCallTicket(USER, CALL_ID)).toBeNull();
    expect(sessionStorage.getItem(KEY)).toBeNull();
  });

  it("forgets the ticket once the call is over", () => {
    saveCallTicket(USER, ticket());
    forgetCallTicket(USER);

    expect(readCallTicket(USER, CALL_ID)).toBeNull();
  });

  it("forgets every ticket on signing out, and nothing else", () => {
    saveCallTicket(USER, ticket());
    saveCallTicket("another-user", ticket());
    sessionStorage.setItem("autogpt:onboarding-intro-path", "A");

    forgetCallTickets();

    expect(readCallTicket(USER, CALL_ID)).toBeNull();
    expect(readCallTicket("another-user", CALL_ID)).toBeNull();
    expect(sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe("A");
  });

  it("forgets every ticket but the signed-in account's when an account takes over the tab", () => {
    saveCallTicket(USER, ticket());
    saveCallTicket("another-user", ticket());
    sessionStorage.setItem("autogpt:onboarding-intro-path", "A");

    forgetCallTickets(USER);

    expect(readCallTicket(USER, CALL_ID)).not.toBeNull();
    expect(readCallTicket("another-user", CALL_ID)).toBeNull();
    expect(sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe("A");
  });
});

describe("the shapes a call's values come in", () => {
  it("knows a call id only in its 32 hex characters", () => {
    expect(isCallId(CALL_ID)).toBe(true);
    expect(isCallId("C".repeat(32))).toBe(false);
    expect(isCallId("c".repeat(31))).toBe(false);
    expect(isCallId("../voice/end")).toBe(false);
    expect(isCallId(42)).toBe(false);
  });

  it("knows a call's room only as a secure socket with a token", () => {
    expect(isCallRoomPass(ROOM)).toBe(true);
    expect(isCallRoomPass({ ...ROOM, url: "not a url" })).toBe(false);
    expect(isCallRoomPass({ ...ROOM, token: "t".repeat(4_097) })).toBe(false);
    expect(isCallRoomPass({ url: ROOM.url })).toBe(false);
    expect(isCallRoomPass("wss://agpt.livekit.cloud")).toBe(false);
  });

  it("takes a plain room socket only on a page served over plain HTTP", () => {
    const plain = { ...ROOM, url: "ws://localhost:7880" };
    expect(isCallRoomPass(plain)).toBe(window.location.protocol === "http:");
  });

  it("knows a call's caps only with every limit a number", () => {
    expect(isVoiceCallCaps(CAPS)).toBe(true);
    expect(isVoiceCallCaps({ ...CAPS, idle_s: null })).toBe(false);
    expect(isVoiceCallCaps({ ...CAPS, max_call_s: -1 })).toBe(false);
    expect(isVoiceCallCaps(null)).toBe(false);
  });
});
