import { describe, expect, it } from "vitest";

import {
  backoffDelay,
  HEARTBEAT_MISS_MS,
  initialLifecycle,
  transition,
  type Lifecycle,
} from "../lifecycle";

const visible = { visible: true, random: () => 1 };
const streaming: Lifecycle = { kind: "streaming", attempt: 0, verified: null };

describe("stream lifecycle", () => {
  it("reconnects a lost connection with backoff, polling the session once it keeps failing", () => {
    const first = transition(
      streaming,
      { type: "connection-lost", failures: 1 },
      visible,
    );
    expect(first.state.kind).toBe("reconnecting");
    expect(first.effect).toEqual({
      type: "reconnect",
      delayMs: 500,
      pollFirst: false,
    });

    const fifth = transition(
      first.state,
      { type: "connection-lost", failures: 5 },
      visible,
    );
    expect(fifth.effect).toEqual({
      type: "reconnect",
      delayMs: 8000,
      pollFirst: true,
    });
  });

  it("never gives up while the tab is visible, and waits while it is hidden", () => {
    const late = transition(
      streaming,
      { type: "connection-lost", failures: 40 },
      visible,
    );
    expect(late.effect).toMatchObject({ type: "reconnect", delayMs: 15_000 });

    const hidden = transition(
      streaming,
      { type: "connection-lost", failures: 1 },
      { visible: false },
    );
    expect(hidden.effect).toEqual({ type: "wait-for-visible" });
    const woken = transition(
      hidden.state,
      { type: "tab-visible", hiddenMs: 60_000, sinceLastFrameMs: 60_000 },
      visible,
    );
    expect(woken.effect).toEqual({
      type: "reconnect",
      delayMs: 0,
      pollFirst: false,
    });
  });

  it("reconnects at once on missed heartbeats", () => {
    const { state, effect } = transition(
      streaming,
      { type: "heartbeat-missed" },
      visible,
    );
    expect(state.kind).toBe("reconnecting");
    expect(effect).toEqual({ type: "reconnect", delayMs: 0, pollFirst: false });
  });

  it("leaves a stream that heartbeats alone when the tab comes back", () => {
    const { state, effect } = transition(
      streaming,
      {
        type: "tab-visible",
        hiddenMs: 120_000,
        sinceLastFrameMs: HEARTBEAT_MISS_MS - 1,
      },
      visible,
    );
    expect(state).toBe(streaming);
    expect(effect).toBeNull();
  });

  it("checks the session for a new turn after a long hide while idle", () => {
    expect(
      transition(
        initialLifecycle,
        { type: "tab-visible", hiddenMs: 31_000, sinceLastFrameMs: Infinity },
        visible,
      ).effect,
    ).toEqual({ type: "probe-session", reason: "wake" });
    expect(
      transition(
        initialLifecycle,
        { type: "tab-visible", hiddenMs: 5_000, sinceLastFrameMs: Infinity },
        visible,
      ).effect,
    ).toBeNull();
  });

  it("settles on finish and probes for a turn the finished one woke", () => {
    const { state, effect } = transition(
      streaming,
      { type: "finished", verified: true },
      visible,
    );
    expect(state).toEqual({ kind: "settled", attempt: 0, verified: true });
    expect(effect).toEqual({ type: "probe-session", reason: "finish" });
  });

  it("jitters the backoff between half and the whole step", () => {
    expect(backoffDelay(3, () => 0)).toBe(1000);
    expect(backoffDelay(3, () => 1)).toBe(2000);
  });
});
