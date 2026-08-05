import { beforeEach, describe, expect, it } from "vitest";
import {
  clearWelcomePending,
  peekCapabilityCardsSeen,
  peekGreetingDone,
  peekIntroPath,
  peekWelcomePending,
  setCapabilityCardsSeen,
  setGreetingDone,
  setIntroAwaitingFollowup,
  setIntroPath,
  setMicGlow,
  setPendingLaterDump,
  setWelcomePending,
  takeIntroAwaitingFollowup,
  takeIntroPath,
  takeMicGlow,
  takePendingLaterDump,
} from "../brain-dump-handoff";

const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";
const MIC_GLOW_KEY = "autogpt:onboarding-mic-glow";
const WELCOME_PENDING_KEY = "autogpt:onboarding-welcome-pending";
const GREETING_DONE_KEY = "autogpt:copilot-greeting-done";
const CAPABILITY_CARDS_KEY = "autogpt:copilot-capability-cards-seen";

beforeEach(() => {
  window.sessionStorage.clear();
  window.localStorage.clear();
});

describe("intro path handoff", () => {
  it("round-trips the path the wizard recorded", () => {
    setIntroPath("A");

    expect(peekIntroPath()).toBe("A");
  });

  it("peeking leaves the path in place so a re-render still sees it", () => {
    setIntroPath("B");

    expect(peekIntroPath()).toBe("B");
    expect(peekIntroPath()).toBe("B");
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
  });

  it("taking the path clears it so a refresh cannot replay the intro", () => {
    setIntroPath("A");

    expect(takeIntroPath()).toBe("A");
    expect(takeIntroPath()).toBeNull();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
  });

  it("returns null for a stored value that is not a known path", () => {
    window.sessionStorage.setItem(INTRO_PATH_KEY, "C");

    expect(peekIntroPath()).toBeNull();
    expect(takeIntroPath()).toBeNull();
  });

  it("taking an absent path is a no-op rather than a write", () => {
    expect(takeIntroPath()).toBeNull();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
  });
});

describe("one-shot session flags", () => {
  it("mic glow is consumed by the first take", () => {
    setMicGlow();

    expect(takeMicGlow()).toBe(true);
    expect(takeMicGlow()).toBe(false);
    expect(window.sessionStorage.getItem(MIC_GLOW_KEY)).toBeNull();
  });

  it("mic glow is false when it was never set", () => {
    expect(takeMicGlow()).toBe(false);
  });

  it("intro-awaiting-followup is consumed by the first take", () => {
    setIntroAwaitingFollowup();

    expect(takeIntroAwaitingFollowup()).toBe(true);
    expect(takeIntroAwaitingFollowup()).toBe(false);
  });

  it("pending-later-dump is consumed by the first take", () => {
    setPendingLaterDump();

    expect(takePendingLaterDump()).toBe(true);
    expect(takePendingLaterDump()).toBe(false);
  });

  it("keeps the three flags independent of each other", () => {
    setMicGlow();

    expect(takeIntroAwaitingFollowup()).toBe(false);
    expect(takePendingLaterDump()).toBe(false);
    expect(takeMicGlow()).toBe(true);
  });
});

describe("welcome pending", () => {
  it("peeking does not consume it — the overlay survives a refresh", () => {
    setWelcomePending();

    expect(peekWelcomePending()).toBe(true);
    expect(peekWelcomePending()).toBe(true);
  });

  it("is only cleared deliberately", () => {
    setWelcomePending();
    clearWelcomePending();

    expect(peekWelcomePending()).toBe(false);
    expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull();
  });

  it("lives in sessionStorage, so it does not survive a new browser session", () => {
    setWelcomePending();

    expect(window.localStorage.getItem(WELCOME_PENDING_KEY)).toBeNull();
    window.sessionStorage.clear();
    expect(peekWelcomePending()).toBe(false);
  });

  it("clearing when nothing is pending is harmless", () => {
    clearWelcomePending();

    expect(peekWelcomePending()).toBe(false);
  });
});

describe("greeting done", () => {
  it("is stored against the user id, not as a boolean", () => {
    setGreetingDone("user-1");

    expect(window.localStorage.getItem(GREETING_DONE_KEY)).toBe("user-1");
    expect(peekGreetingDone("user-1")).toBe(true);
  });

  it("does not leak to a different account on the same browser", () => {
    setGreetingDone("user-1");

    expect(peekGreetingDone("user-2")).toBe(false);
  });

  it("ignores a missing user id on both read and write", () => {
    setGreetingDone(null);
    setGreetingDone(undefined);

    expect(window.localStorage.getItem(GREETING_DONE_KEY)).toBeNull();
    expect(peekGreetingDone(null)).toBe(false);
    expect(peekGreetingDone(undefined)).toBe(false);
  });

  it("survives a new session because it is kept in localStorage", () => {
    setGreetingDone("user-1");
    window.sessionStorage.clear();

    expect(peekGreetingDone("user-1")).toBe(true);
  });
});

describe("capability cards seen", () => {
  it("is stored against the user id", () => {
    setCapabilityCardsSeen("user-1");

    expect(window.localStorage.getItem(CAPABILITY_CARDS_KEY)).toBe("user-1");
    expect(peekCapabilityCardsSeen("user-1")).toBe(true);
    expect(peekCapabilityCardsSeen("user-2")).toBe(false);
  });

  it("ignores a missing user id on both read and write", () => {
    setCapabilityCardsSeen(null);

    expect(window.localStorage.getItem(CAPABILITY_CARDS_KEY)).toBeNull();
    expect(peekCapabilityCardsSeen(null)).toBe(false);
    expect(peekCapabilityCardsSeen(undefined)).toBe(false);
  });

  it("is tracked separately from the greeting flag", () => {
    setCapabilityCardsSeen("user-1");

    expect(peekGreetingDone("user-1")).toBe(false);
  });
});
