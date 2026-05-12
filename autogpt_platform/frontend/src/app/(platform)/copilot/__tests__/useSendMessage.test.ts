import { act, cleanup, renderHook } from "@testing-library/react";
import { useRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useSendMessage } from "../useSendMessage";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: vi.fn(),
}));

function resetStore() {
  useCopilotStreamStore.setState({
    sessions: {},
    messageSnapshots: {},
    pendingFirstSend: null,
    pendingFileParts: [],
  });
}

function useTestHarness(args: {
  sessionId: string | null;
  sendMessage: ReturnType<typeof vi.fn>;
  createSession: ReturnType<typeof vi.fn>;
}) {
  const isUserStoppingRef = useRef(false);
  return useSendMessage({
    sessionId: args.sessionId,
    sendMessage: args.sendMessage as never,
    createSession: args.createSession as unknown as () => Promise<
      string | undefined
    >,
    isUserStoppingRef,
  });
}

describe("useSendMessage", () => {
  beforeEach(() => {
    resetStore();
    mockToast.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    cleanup();
  });

  describe("first-send flush on sessionId change", () => {
    it("dispatches the pending message and clears the slot when sessionId arrives", async () => {
      const sendMessage = vi.fn();
      const createSession = vi.fn();

      useCopilotStreamStore.getState().setPendingFirstSend({
        text: "hello",
        files: [],
      });

      const { rerender } = renderHook(
        ({ sessionId }) =>
          useTestHarness({ sessionId, sendMessage, createSession }),
        { initialProps: { sessionId: null as string | null } },
      );

      expect(sendMessage).not.toHaveBeenCalled();

      await act(async () => {
        rerender({ sessionId: "new-session-id" });
      });

      expect(sendMessage).toHaveBeenCalledWith({ text: "hello" });
      expect(useCopilotStreamStore.getState().pendingFirstSend).toBeNull();
    });

    it("does nothing when sessionId arrives without a pending send", async () => {
      const sendMessage = vi.fn();
      const createSession = vi.fn();

      const { rerender } = renderHook(
        ({ sessionId }) =>
          useTestHarness({ sessionId, sendMessage, createSession }),
        { initialProps: { sessionId: null as string | null } },
      );

      await act(async () => {
        rerender({ sessionId: "new-session-id" });
      });

      expect(sendMessage).not.toHaveBeenCalled();
    });
  });

  describe("stalled-flush detection", () => {
    it("toasts and clears the slot if the flush never runs within the timeout", async () => {
      const sendMessage = vi.fn();
      const createSession = vi.fn().mockResolvedValue("new-session-id");

      const { result } = renderHook(() =>
        useTestHarness({
          sessionId: null,
          sendMessage,
          createSession,
        }),
      );

      await act(async () => {
        await result.current.onSend("hello");
      });

      expect(useCopilotStreamStore.getState().pendingFirstSend).not.toBeNull();

      await act(async () => {
        vi.advanceTimersByTime(5000);
      });

      expect(useCopilotStreamStore.getState().pendingFirstSend).toBeNull();
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "destructive" }),
      );
    });

    it("does not toast when the flush effect consumes the slot", async () => {
      const sendMessage = vi.fn();
      const createSession = vi.fn().mockResolvedValue("new-session-id");

      const { result } = renderHook(() =>
        useTestHarness({
          sessionId: null,
          sendMessage,
          createSession,
        }),
      );

      await act(async () => {
        await result.current.onSend("hello");
      });

      // Simulate the post-create remount picking up the pending send.
      act(() => {
        useCopilotStreamStore.getState().takePendingFirstSend();
      });

      await act(async () => {
        vi.advanceTimersByTime(5000);
      });

      expect(mockToast).not.toHaveBeenCalled();
    });

    it("cancels the watchdog on unmount so the toast can't fire on an unrelated page", async () => {
      const sendMessage = vi.fn();
      const createSession = vi.fn().mockResolvedValue("new-session-id");

      const { result, unmount } = renderHook(() =>
        useTestHarness({
          sessionId: null,
          sendMessage,
          createSession,
        }),
      );

      await act(async () => {
        await result.current.onSend("hello");
      });

      // User navigates away before the watchdog fires.
      unmount();

      await act(async () => {
        vi.advanceTimersByTime(5000);
      });

      expect(mockToast).not.toHaveBeenCalled();
    });

    it("does not toast when a subsequent send overwrites the slot", async () => {
      const sendMessage = vi.fn();
      const createSession = vi.fn().mockResolvedValue("new-session-id");

      const { result } = renderHook(() =>
        useTestHarness({
          sessionId: null,
          sendMessage,
          createSession,
        }),
      );

      await act(async () => {
        await result.current.onSend("first");
      });

      // Simulate another new chat starting: a fresh setPendingFirstSend
      // swaps the slot identity, so the first timer should silently no-op.
      act(() => {
        useCopilotStreamStore.getState().setPendingFirstSend({
          text: "second",
          files: [],
        });
      });

      await act(async () => {
        vi.advanceTimersByTime(5000);
      });

      expect(mockToast).not.toHaveBeenCalled();
      // The second slot is still there — its own timer hasn't elapsed.
      expect(useCopilotStreamStore.getState().pendingFirstSend?.text).toBe(
        "second",
      );
    });
  });
});
