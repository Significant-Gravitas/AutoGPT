import type { ComponentProps, ReactNode } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatMessagesContainer } from "../ChatMessagesContainer";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

// The transcript's clock is the subject here, so the timer hooks and the
// ThinkingIndicator stay real; everything that only adds noise is stubbed.
const chainRenders = { count: 0 };

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({
    scrollRef: {
      current: { scrollHeight: 100, scrollTop: 0, clientHeight: 500 },
    },
  }),
}));

vi.mock("@/components/organisms/Conversation/Conversation", () => ({
  Conversation: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationContent: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationScrollButton: () => null,
}));

vi.mock("@/components/organisms/Message/Message", () => ({
  Message: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  MessageActions: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  MessageContent: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("../components/AssistantMessageActions", () => ({
  AssistantMessageActions: () => null,
}));
vi.mock("../components/ChainMessageParts", () => ({
  ChainMessageParts: () => {
    chainRenders.count++;
    return <div data-testid="chain-message-parts" />;
  },
}));
vi.mock("../components/CopyButton", () => ({
  CopyButton: () => null,
}));
vi.mock("../components/QueueBadge", () => ({
  QueueBadge: () => null,
}));
vi.mock("../components/MessageAttachments", () => ({
  MessageAttachments: () => null,
}));
vi.mock("../components/MessagePartRenderer", () => ({
  MessagePartRenderer: ({ part }: { part: { type: string; text?: string } }) =>
    part.type === "text" ? <span>{part.text}</span> : null,
}));
vi.mock("../../ToolChain/SwapText", () => ({
  SwapText: ({ text }: { text: string }) => <span>{text}</span>,
}));
vi.mock("../../DrawnLineLoader/DrawnLineLoader", () => ({
  DrawnLineLoader: () => null,
}));
vi.mock("../../JobStatsBar/TurnStatsBar", () => ({
  TurnStatsBar: ({ elapsedSeconds }: { elapsedSeconds?: number }) => (
    <div data-testid="turn-stats-bar">{String(elapsedSeconds)}</div>
  ),
}));
vi.mock("../../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));
vi.mock("../../ChatMinimap/ChatMinimap", () => ({
  ChatMinimap: () => null,
}));

class MockIntersectionObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
  takeRecords() {
    return [];
  }
  root = null;
  rootMargin = "";
  thresholds = [];
}

const NOW = new Date("2026-04-23T10:00:00.000Z");

const userMessage: Message = {
  id: "user-1",
  role: "user",
  parts: [{ type: "text", text: "go" }],
};

const runningTool: Message = {
  id: "assistant-1",
  role: "assistant",
  parts: [
    {
      type: "tool-run_block",
      toolCallId: "call-1",
      state: "input-available",
      input: {},
    },
  ] as Message["parts"],
};

const answer: Message = {
  id: "assistant-1",
  role: "assistant",
  parts: [{ type: "text", text: "Done" }],
};

const baseProps = {
  messages: [userMessage, runningTool],
  status: "streaming",
  error: undefined,
  isLoading: false,
  sessionID: "sess-1",
  queuedMessages: [] as string[],
  hasMoreMessages: false,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  onRetry: vi.fn(),
} satisfies ComponentProps<typeof ChatMessagesContainer>;

function advanceSeconds(seconds: number) {
  for (let i = 0; i < seconds; i++) {
    act(() => {
      vi.advanceTimersByTime(1_000);
    });
  }
}

describe("ChatMessagesContainer — thinking clock", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
    vi.setSystemTime(NOW);
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
    chainRenders.count = 0;
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("does not re-render the transcript every second while a tool runs", () => {
    render(<ChatMessagesContainer {...baseProps} />);
    act(() => {
      vi.advanceTimersByTime(100);
    });
    const settled = chainRenders.count;
    expect(settled).toBeGreaterThan(0);

    advanceSeconds(10);

    expect(chainRenders.count).toBe(settled);
  });

  it("shows the elapsed time in the thinking indicator after 20 s, without re-rendering the transcript", () => {
    const startedAt = new Date(NOW.getTime() - 15_000).toISOString();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[userMessage, { ...runningTool, parts: [] }]}
        status="submitted"
        activeStreamStartedAt={startedAt}
      />,
    );
    act(() => {
      vi.advanceTimersByTime(100);
    });
    const settled = chainRenders.count;
    expect(screen.getByRole("status").textContent).toBe("Thinking…");

    advanceSeconds(4);
    expect(screen.getByRole("status").textContent).toBe("Thinking…");

    advanceSeconds(1);
    expect(screen.getByRole("status").textContent).toBe("Thinking…20s");

    advanceSeconds(3);
    expect(screen.getByRole("status").textContent).toBe("Thinking…23s");
    expect(chainRenders.count).toBe(settled);
  });

  it("hands the finished turn's elapsed time to its stats bar", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    advanceSeconds(7);

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        messages={[userMessage, answer]}
        status="ready"
      />,
    );
    expect(screen.getByTestId("turn-stats-bar").textContent).toBe("7");

    advanceSeconds(5);
    expect(screen.getByTestId("turn-stats-bar").textContent).toBe("7");
  });
});
