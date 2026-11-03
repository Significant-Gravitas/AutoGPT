import React, { useState } from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatContainer, type ChatContainerProps } from "../ChatContainer";

// Counts the composer's own renders: `useChatInput` runs once per ChatInput
// render, and the real ChatInput (with its memo boundary) stays in the tree.
const composerRenders = vi.hoisted(() => ({ count: 0 }));
vi.mock("../../ChatInput/useChatInput", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../../ChatInput/useChatInput")>();
  return {
    ...actual,
    useChatInput(...args: Parameters<typeof actual.useChatInput>) {
      composerRenders.count += 1;
      return actual.useChatInput(...args);
    },
  };
});

const enabledFlags = vi.hoisted(() => new Set<string>());
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => enabledFlags.has(flag),
  };
});

vi.mock("../../ChatMessagesContainer/ChatMessagesContainer", () => ({
  ChatMessagesContainer: ({ onRetry }: { onRetry?: () => void }) => (
    <button type="button" onClick={onRetry}>
      Retry message
    </button>
  ),
}));

vi.mock("../../UsageLimits/useIsUsageLimitReached", () => ({
  useIsUsageLimitReached: () => false,
}));

class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

function textMessage(
  id: string,
  role: "user" | "assistant",
  text: string,
): UIMessage {
  return { id, role, parts: [{ type: "text", text }] } as UIMessage;
}

function streamedMessages(reply: string) {
  return [
    textMessage("u1", "user", "Plan my week"),
    textMessage("a1", "assistant", reply),
  ];
}

const baseProps: ChatContainerProps = {
  messages: streamedMessages("Here"),
  status: "streaming",
  error: undefined,
  sessionId: "session-123",
  isLoadingSession: false,
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
  onStop: vi.fn(),
  onEnqueue: vi.fn(),
  droppedFiles: [],
  onDroppedFilesConsumed: vi.fn(),
};

// Updates go through state, not `rerender`, so the test providers above the
// container keep their values and only the container's props change.
let setProps: (update: Partial<ChatContainerProps>) => void = () => {};

function Harness(initial: Partial<ChatContainerProps>) {
  const [props, setState] = useState<ChatContainerProps>({
    ...baseProps,
    ...initial,
  });
  setProps = (update) => setState((prev) => ({ ...prev, ...update }));
  return <ChatContainer {...props} />;
}

function renderContainer(initial: Partial<ChatContainerProps> = {}) {
  render(<Harness {...initial} />);
  return composerRenders.count;
}

function update(props: Partial<ChatContainerProps>) {
  act(() => setProps(props));
}

describe("ChatContainer composer renders", () => {
  beforeEach(() => {
    composerRenders.count = 0;
    enabledFlags.clear();
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("does not render the composer for a streamed flush", () => {
    const before = renderContainer();

    update({ messages: streamedMessages("Here is") });
    update({ messages: streamedMessages("Here is the plan") });

    expect(composerRenders.count).toBe(before);
  });

  it("does not render the composer when the host passes new callbacks", () => {
    const before = renderContainer();

    // CopilotPage passes `() => setDroppedFiles([])` inline.
    update({
      messages: streamedMessages("Here is"),
      onStop: vi.fn(),
      onEnqueue: vi.fn(),
      onDroppedFilesConsumed: vi.fn(),
    });

    expect(composerRenders.count).toBe(before);
  });

  it("does not render the composer for a flush with voice and auto mode on", () => {
    enabledFlags.add("copilot-voice-mode");
    enabledFlags.add("copilot-auto-mode");
    const before = renderContainer();

    update({ messages: streamedMessages("Here is") });

    expect(composerRenders.count).toBe(before);
  });

  it("still renders the composer when the turn ends", () => {
    const before = renderContainer();
    expect(screen.getByRole("button", { name: /stop/i })).toBeDefined();

    update({ status: "ready" });

    expect(composerRenders.count).toBeGreaterThan(before);
    expect(screen.queryByRole("button", { name: /stop/i })).toBeNull();
  });

  it("calls the latest onStop after the host replaced it", async () => {
    const firstStop = vi.fn();
    const latestStop = vi.fn();
    renderContainer({ onStop: firstStop });

    update({ onStop: latestStop });
    // Stop arms a second after it appears.
    const stop = screen.getByRole("button", { name: /stop/i });
    await waitFor(() => expect(stop).toHaveProperty("disabled", false), {
      timeout: 2_000,
    });
    fireEvent.click(stop);

    expect(latestStop).toHaveBeenCalledTimes(1);
    expect(firstStop).not.toHaveBeenCalled();
  });

  it("retries the latest user message, not the one from the first render", async () => {
    const onSend = vi.fn();
    renderContainer({ onSend, status: "ready" });

    update({
      messages: [
        ...streamedMessages("Here is the plan"),
        textMessage("u2", "user", "Now the budget"),
      ],
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Retry message" }));
    });

    expect(onSend).toHaveBeenCalledWith(
      "Now the budget",
      undefined,
      undefined,
      undefined,
    );
  });
});
