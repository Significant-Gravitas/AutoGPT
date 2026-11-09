import { afterEach, describe, expect, it } from "vitest";
import {
  type CallTicket,
  forgetCallTicket,
  forgetCallTickets,
  isCallId,
  isVoiceCallCaps,
  readCallTicket,
  saveCallTicket,
} from "../voice-call-ticket";

const USER = "ticket-user";
const KEY = `autogpt:onboarding-voice-call:${USER}`;
const CALL_ID = "c".repeat(32);
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

  it("keeps no relay socket: the server names it when the call is rejoined", () => {
    saveCallTicket(USER, ticket());

    expect(
      Object.keys(JSON.parse(sessionStorage.getItem(KEY) ?? "{}")),
    ).toEqual(["callId", "ticket", "caps", "startedAt", "turnsLeft"]);
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
});

describe("the shapes a call's values come in", () => {
  it("knows a call id only in its 32 hex characters", () => {
    expect(isCallId(CALL_ID)).toBe(true);
    expect(isCallId("C".repeat(32))).toBe(false);
    expect(isCallId("c".repeat(31))).toBe(false);
    expect(isCallId("../voice/end")).toBe(false);
    expect(isCallId(42)).toBe(false);
  });

  it("knows a call's caps only with every limit a number", () => {
    expect(isVoiceCallCaps(CAPS)).toBe(true);
    expect(isVoiceCallCaps({ ...CAPS, idle_s: null })).toBe(false);
    expect(isVoiceCallCaps({ ...CAPS, max_call_s: -1 })).toBe(false);
    expect(isVoiceCallCaps(null)).toBe(false);
  });
});
