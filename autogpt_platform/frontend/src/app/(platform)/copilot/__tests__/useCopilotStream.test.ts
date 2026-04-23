import { act, renderHook } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCopilotStream } from "../useCopilotStream";

// Capture the args passed to ``useChat`` so tests can invoke onFinish/onError
// directly — that's the only way to drive handleReconnect without a real SSE.
let lastUseChatArgs: {
  onFinish?: (args: { isDisconnect?: boolean; isAbort?: boolean }) => void;
  onError?: (err: Error) => void;
} | null = null;

const resumeStreamMock = vi.fn();
const sdkStopMock = vi.fn();
const sdkSendMessageMock = vi.fn();
const setMessagesMock = vi.fn();

function resetSdkMocks() {
  lastUseChatArgs = null;
  resumeStreamMock.mockReset();
  sdkStopMock.mockReset();
  sdkSendMessageMock.mockReset();
  setMessagesMock.mockReset();
}

vi.mock("@ai-sdk/react", () => ({
  useChat: (args: unknown) => {
    lastUseChatArgs = args as typeof lastUseChatArgs;
    return {
      messages: [] as UIMessage[],
      sendMessage: sdkSendMessageMock,
      stop: sdkStopMock,
      status: "ready" as const,
      error: undefined,
      setMessages: setMessagesMock,
      resumeStream: resumeStreamMock,
    };
  },
}));

vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  return {
    ...actual,
    DefaultChatTransport: class {
      constructor(public opts: unknown) {}
    },
  };
});

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  getGetV2GetCopilotUsageQueryKey: () => ["copilot-usage"],
  getGetV2GetSessionQueryKey: (id: string) => ["session", id],
  postV2CancelSessionTask: vi.fn(),
  deleteV2DisconnectSessionStream: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: () => "http://localhost",
  },
}));

vi.mock("../helpers", async () => {
  const actual =
    await vi.importActual<typeof import("../helpers")>("../helpers");
  return {
    ...actual,
    getCopilotAuthHeaders: vi.fn().mockResolvedValue({}),
    disconnectSessionStream: vi.fn(),
  };
});

vi.mock("../useHydrateOnStreamEnd", () => ({
  useHydrateOnStreamEnd: () => undefined,
}));

function renderStream() {
  return renderHook(() =>
    useCopilotStream({
      sessionId: "sess-1",
      hydratedMessages: [],
      hasActiveStream: false,
      refetchSession: vi.fn().mockResolvedValue({ data: undefined }),
      copilotMode: undefined,
      copilotModel: undefined,
    }),
  );
}

describe("useCopilotStream — reconnect debounce", () => {
  beforeEach(() => {
    resetSdkMocks();
    vi.useFakeTimers();
    // Pin Date.now so sinceLastResume math is deterministic. The hook reads
    // Date.now() both when stashing lastReconnectResumeAtRef and when
    // deciding whether to debounce.
    vi.setSystemTime(new Date(2025, 0, 1, 12, 0, 0));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("coalesces a burst of disconnect events into one resumeStream call", async () => {
    renderStream();

    // First disconnect — schedules a reconnect at the exponential backoff
    // delay (1000ms for attempt #1).
    await act(async () => {
      await lastUseChatArgs!.onFinish!({ isDisconnect: true });
    });

    // Fire the scheduled timer → resumeStream runs once and stamps
    // lastReconnectResumeAtRef.current = Date.now().
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(resumeStreamMock).toHaveBeenCalledTimes(1);

    // A second disconnect arrives immediately after (still inside the
    // 1500ms debounce window) — the debounce path must fire and queue a
    // coalesced timer, NOT a fresh resume.
    await act(async () => {
      await lastUseChatArgs!.onFinish!({ isDisconnect: true });
    });
    expect(resumeStreamMock).toHaveBeenCalledTimes(1);

    // The coalesced timer fires at the window boundary and reschedules a
    // real reconnect. Advance past the window AND past the second
    // reconnect's backoff (attempt #2 = 2000ms) so resume runs.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_500);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    expect(resumeStreamMock).toHaveBeenCalledTimes(2);
  });

  it("does not debounce a reconnect that arrives after the window closes", async () => {
    renderStream();

    // First reconnect cycle.
    await act(async () => {
      await lastUseChatArgs!.onFinish!({ isDisconnect: true });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(resumeStreamMock).toHaveBeenCalledTimes(1);

    // Wait past the debounce window before the next disconnect.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });

    // Now a fresh disconnect should go through the normal path (NOT the
    // debounce branch) and schedule a backoff of 2000ms (attempt #2).
    await act(async () => {
      await lastUseChatArgs!.onFinish!({ isDisconnect: true });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000);
    });
    expect(resumeStreamMock).toHaveBeenCalledTimes(2);
  });
});
