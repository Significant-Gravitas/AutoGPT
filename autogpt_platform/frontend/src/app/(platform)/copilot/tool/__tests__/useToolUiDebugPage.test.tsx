import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useToolUiDebugPage } from "../useToolUiDebugPage";

vi.mock("../sampleScript", () => ({
  buildSampleEvents: () => [
    { delay: 10, kind: "status", message: "Planning" },
    { delay: 10, kind: "assistant-start", id: "assistant-1" },
    { delay: 10, kind: "await-user" },
    { delay: 10, kind: "text-start", messageId: "assistant-1" },
    {
      delay: 10,
      kind: "text-delta",
      messageId: "assistant-1",
      delta: "Done",
    },
  ],
}));

describe("useToolUiDebugPage", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("adds manual messages, changes variants, and resets", () => {
    const { result } = renderHook(() => useToolUiDebugPage());

    act(() => {
      result.current.sendUserMessage("Manual message");
      result.current.setVariant("old");
    });

    expect(result.current.messages[0]).toMatchObject({
      id: "debug-user-1",
      role: "user",
      parts: [{ type: "text", text: "Manual message" }],
    });
    expect(result.current.variant).toBe("old");

    act(() => result.current.reset());

    expect(result.current.messages).toEqual([]);
    expect(result.current.status).toBe("ready");
    expect(result.current.statusMessage).toBeNull();
  });

  it("plays events and resumes after a user answer", async () => {
    const { result } = renderHook(() => useToolUiDebugPage());
    let playback: Promise<void>;

    act(() => {
      playback = result.current.play();
    });
    expect(result.current.isPlaying).toBe(true);

    await act(() => vi.advanceTimersByTimeAsync(10));
    expect(result.current.statusMessage).toBe("Planning");

    await act(() => vi.advanceTimersByTimeAsync(20));
    expect(result.current.awaitingUser).toBe(true);
    expect(result.current.status).toBe("ready");

    await act(async () => result.current.sendUserMessage("Continue"));
    expect(result.current.awaitingUser).toBe(false);
    expect(result.current.status).toBe("streaming");

    await act(() => vi.advanceTimersByTimeAsync(20));
    await act(async () => playback);

    expect(result.current.status).toBe("ready");
    expect(result.current.messages).toEqual([
      {
        id: "assistant-1",
        role: "assistant",
        parts: [{ type: "text", text: "Done" }],
      },
      {
        id: "debug-user-1",
        role: "user",
        parts: [{ type: "text", text: "Continue" }],
      },
    ]);
  });

  it("cancels a paused run when reset", async () => {
    const { result } = renderHook(() => useToolUiDebugPage());
    let playback: Promise<void>;

    act(() => {
      playback = result.current.play();
    });
    await act(() => vi.advanceTimersByTimeAsync(30));

    act(() => result.current.reset());
    await act(async () => playback);

    expect(result.current.awaitingUser).toBe(false);
    expect(result.current.status).toBe("ready");
    expect(result.current.messages).toEqual([]);
  });
});
