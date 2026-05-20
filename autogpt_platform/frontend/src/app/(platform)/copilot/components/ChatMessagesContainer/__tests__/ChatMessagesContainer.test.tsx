import { act } from "@testing-library/react";
import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatMessagesContainer } from "../ChatMessagesContainer";

const mockScrollEl = {
  scrollHeight: 100,
  scrollTop: 0,
  clientHeight: 500,
};

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({ scrollRef: { current: mockScrollEl } }),
  Conversation: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationScrollButton: () => null,
}));

vi.mock("@/components/ai-elements/conversation", () => ({
  Conversation: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="conversation">{children}</div>
  ),
  ConversationContent: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="conversation-content">{children}</div>
  ),
  ConversationScrollButton: () => null,
}));

vi.mock("@/components/ai-elements/message", () => ({
  Message: ({
    children,
    from,
  }: {
    children: React.ReactNode;
    from?: string;
  }) => (
    <div data-testid={`message-${from ?? "unknown"}`} data-from={from}>
      {children}
    </div>
  ),
  MessageActions: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  MessageContent: ({
    children,
    className,
  }: {
    children: React.ReactNode;
    className?: string;
  }) => <div className={className}>{children}</div>,
}));

vi.mock("../components/AssistantMessageActions", () => ({
  AssistantMessageActions: () => null,
}));

vi.mock("../components/QueueBadge", () => ({
  QueueBadge: ({ sessionID }: { sessionID: string | null }) => (
    <span data-testid="queue-badge" data-session-id={sessionID ?? ""}>
      QueueBadge
    </span>
  ),
}));

vi.mock("../components/CopyButton", () => ({ CopyButton: () => null }));
vi.mock("../components/CollapsedToolGroup", () => ({
  CollapsedToolGroup: () => null,
}));
vi.mock("../components/MessageAttachments", () => ({
  MessageAttachments: () => null,
}));
vi.mock("../components/MessagePartRenderer", () => ({
  MessagePartRenderer: () => null,
}));
vi.mock("../components/ReasoningCollapse", () => ({
  ReasoningCollapse: () => null,
}));
vi.mock("../components/ThinkingIndicator", () => ({
  ThinkingIndicator: ({ statusMessage }: { statusMessage?: string | null }) => (
    <div data-testid="thinking-indicator">{statusMessage ?? "thinking"}</div>
  ),
}));
vi.mock("../../JobStatsBar/TurnStatsBar", () => ({
  TurnStatsBar: () => null,
}));
vi.mock("../../JobStatsBar/useElapsedTimer", () => ({
  useElapsedTimer: () => ({ elapsedSeconds: 0 }),
}));
vi.mock("../../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));
// Tests below override this default by re-mocking ../helpers as needed.
vi.mock("../helpers", () => ({
  buildRenderSegments: () => [],
  getTurnMessages: () => [],
  parseSpecialMarkers: (text: string) => {
    if (typeof text === "string" && text.startsWith("[__COPILOT_ERROR_")) {
      return { markerType: "error" };
    }
    if (
      typeof text === "string" &&
      text.startsWith("[__COPILOT_RETRYABLE_ERROR_")
    ) {
      return { markerType: "retryable_error" };
    }
    return { markerType: null };
  },
  splitReasoningAndResponse: (parts: unknown[]) => ({
    reasoning: [],
    response: parts,
  }),
}));

vi.mock("@/components/atoms/LoadingSpinner/LoadingSpinner", () => ({
  LoadingSpinner: () => <div data-testid="loading-spinner" />,
}));

vi.mock("@phosphor-icons/react", () => ({
  Clock: () => <span data-testid="clock-icon" />,
  ArrowDown: () => null,
  ArrowUp: () => null,
}));

// ── helpers ───────────────────────────────────────────────────────────────

type ObserverCallback = (entries: { isIntersecting: boolean }[]) => void;
class MockIntersectionObserver {
  static lastCallback: ObserverCallback | null = null;
  private callback: ObserverCallback;
  constructor(cb: ObserverCallback) {
    this.callback = cb;
    MockIntersectionObserver.lastCallback = cb;
  }
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

const baseProps = {
  messages: [] as any[],
  status: "ready" as const,
  error: undefined,
  isLoading: false,
  sessionID: "sess-123",
  queuedMessages: [] as string[],
  hasMoreMessages: true,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  onRetry: vi.fn(),
};

// ── queued-messages rendering ─────────────────────────────────────────────

describe("ChatMessagesContainer — queuedMessages", () => {
  beforeEach(() => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    mockScrollEl.clientHeight = 500;
    MockIntersectionObserver.lastCallback = null;
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders nothing extra when queuedMessages is empty", () => {
    render(<ChatMessagesContainer {...baseProps} queuedMessages={[]} />);
    expect(screen.queryByText("Queued")).toBeNull();
  });

  it("renders a single queued message with Queued label", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        queuedMessages={["What about section 3?"]}
      />,
    );
    expect(screen.getByText("What about section 3?")).toBeDefined();
    expect(screen.getByText("Queued")).toBeDefined();
  });

  it("renders multiple queued messages as separate bubbles", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        queuedMessages={["First follow-up", "Second follow-up"]}
      />,
    );
    expect(screen.getByText("First follow-up")).toBeDefined();
    expect(screen.getByText("Second follow-up")).toBeDefined();
    const queuedLabels = screen.getAllByText("Queued");
    expect(queuedLabels.length).toBe(2);
  });

  it("renders queued messages even when status is streaming", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        queuedMessages={["queued during stream"]}
      />,
    );
    expect(screen.getByText("queued during stream")).toBeDefined();
    expect(screen.getByText("Queued")).toBeDefined();
  });

  it("renders no queued messages when prop is undefined", () => {
    const { queuedMessages: _, ...propsWithoutQueued } = baseProps;
    render(<ChatMessagesContainer {...propsWithoutQueued} />);
    expect(screen.queryByText("Queued")).toBeNull();
  });
});

// ── loading state ─────────────────────────────────────────────────────────

describe("ChatMessagesContainer — loading", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    mockScrollEl.clientHeight = 500;
    MockIntersectionObserver.lastCallback = null;
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    vi.useRealTimers();
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows loading spinner when isLoading is true", () => {
    render(<ChatMessagesContainer {...baseProps} isLoading />);
    expect(screen.getByTestId("loading-spinner")).toBeDefined();
  });

  it("does not show spinner when not loading", () => {
    render(<ChatMessagesContainer {...baseProps} isLoading={false} />);
    expect(screen.queryByTestId("loading-spinner")).toBeNull();
  });

  it("shows the restore message instead of stale tail content during active-session resume", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        isLoading={false}
        isRestoringActiveSession
        messages={[
          {
            id: "user-1",
            role: "user",
            parts: [{ type: "text", text: "Investigate this" }],
          },
        ]}
      />,
    );

    expect(screen.getByTestId("message-user")).toBeDefined();
    expect(screen.getByText("Retrieving latest messages")).toBeDefined();
  });

  it("shows a reconnecting fallback after 6 seconds of restore", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        isLoading={false}
        isRestoringActiveSession
        activeStreamStartedAt="2026-04-23T15:00:00.000Z"
        messages={[
          {
            id: "user-1",
            role: "user",
            parts: [{ type: "text", text: "Investigate this" }],
          },
        ]}
      />,
    );

    act(() => {
      vi.advanceTimersByTime(6_000);
    });

    expect(screen.getByTestId("thinking-indicator")).toBeDefined();
    expect(screen.getByText("Reconnecting to live stream...")).toBeDefined();
    expect(
      screen.getByText("Still syncing the latest progress."),
    ).toBeDefined();
  });

  it("prefers the backend status message in the restore fallback", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        isLoading={false}
        isRestoringActiveSession
        restoreStatusMessage="Analyzing result..."
        messages={[
          {
            id: "user-1",
            role: "user",
            parts: [{ type: "text", text: "Investigate this" }],
          },
        ]}
      />,
    );

    act(() => {
      vi.advanceTimersByTime(6_000);
    });

    expect(screen.getByText("Analyzing result...")).toBeDefined();
  });
});

// ── error banner dedup ────────────────────────────────────────────────────

describe("ChatMessagesContainer — error banner dedup", () => {
  beforeEach(() => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    mockScrollEl.clientHeight = 500;
    MockIntersectionObserver.lastCallback = null;
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the trailing banner when no persisted error marker is in messages", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        error={new Error("SDK stream error: Prompt is too long")}
        status="error"
        messages={[
          {
            id: "u-1",
            role: "user",
            parts: [{ type: "text", text: "go" }],
          },
        ]}
      />,
    );
    expect(
      screen.getByText("SDK stream error: Prompt is too long"),
    ).toBeDefined();
    expect(screen.getByText(/encountered an error/i)).toBeDefined();
  });

  it("suppresses the trailing banner when the last assistant message carries an error marker", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        error={new Error("SDK stream error: Prompt is too long")}
        status="error"
        messages={[
          {
            id: "u-1",
            role: "user",
            parts: [{ type: "text", text: "go" }],
          },
          {
            id: "a-1",
            role: "assistant",
            parts: [
              {
                type: "text",
                text: "[__COPILOT_ERROR_f7a1__] SDK stream error: Prompt is too long",
              },
            ],
          },
        ]}
      />,
    );
    expect(screen.queryByText(/encountered an error/i)).toBeNull();
  });

  it("suppresses the trailing banner when the marker is retryable", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        error={new Error("Transient error")}
        status="error"
        messages={[
          {
            id: "u-1",
            role: "user",
            parts: [{ type: "text", text: "go" }],
          },
          {
            id: "a-1",
            role: "assistant",
            parts: [
              {
                type: "text",
                text: "[__COPILOT_RETRYABLE_ERROR_a9c2__] Transient error",
              },
            ],
          },
        ]}
      />,
    );
    expect(screen.queryByText(/encountered an error/i)).toBeNull();
  });
});

// ── pagination sentinel ───────────────────────────────────────────────────

describe("ChatMessagesContainer", () => {
  beforeEach(() => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    mockScrollEl.clientHeight = 500;
    MockIntersectionObserver.lastCallback = null;
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders top sentinel for backward pagination", () => {
    render(<ChatMessagesContainer {...baseProps} />);
    expect(
      screen.getByRole("button", { name: /load older messages/i }),
    ).toBeDefined();
  });

  it("hides sentinel when hasMoreMessages is false", () => {
    render(<ChatMessagesContainer {...baseProps} hasMoreMessages={false} />);
    expect(
      screen.queryByRole("button", { name: /load older messages/i }),
    ).toBeNull();
  });

  it("hides sentinel when onLoadMore is not provided", () => {
    render(<ChatMessagesContainer {...baseProps} onLoadMore={undefined} />);
    expect(
      screen.queryByRole("button", { name: /load older messages/i }),
    ).toBeNull();
  });
});

// ── turnStats plumbing ────────────────────────────────────────────────────

describe("ChatMessagesContainer — turnStats", () => {
  beforeEach(() => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    mockScrollEl.clientHeight = 500;
    MockIntersectionObserver.lastCallback = null;
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the local timestamp on a user message (hover reveal)", () => {
    const userId = "user-1";
    const turnStats = new Map([
      [userId, { createdAt: "2026-04-23T08:32:09.000Z" }],
    ]);
    const messages = [
      {
        id: userId,
        role: "user" as const,
        parts: [{ type: "text" as const, text: "hi", state: "done" }],
      },
    ];
    render(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <ChatMessagesContainer
        {...(baseProps as any)}
        messages={messages as any}
        turnStats={turnStats as any}
      />,
    );
    // The timestamp is rendered in the MessageActions area alongside CopyButton;
    // we just assert that SOMETHING containing the year is in the DOM.
    const labels = screen.getAllByText(
      (_, el) =>
        !!el?.className.includes("tabular-nums") &&
        /2026/.test(el?.textContent ?? ""),
    );
    expect(labels.length).toBeGreaterThan(0);
  });

  it("skips the user timestamp when turnStats has no entry for that message id", () => {
    const messages = [
      {
        id: "user-unknown",
        role: "user" as const,
        parts: [{ type: "text" as const, text: "hi", state: "done" }],
      },
    ];
    render(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <ChatMessagesContainer
        {...(baseProps as any)}
        messages={messages as any}
        turnStats={new Map() as any}
      />,
    );
    const labels = screen.queryAllByText((_, el) =>
      /2026/.test(el?.textContent ?? ""),
    );
    expect(labels.length).toBe(0);
  });
});

// ── per-message queue badge ───────────────────────────────────────────────

describe("ChatMessagesContainer — queue badges on user messages", () => {
  beforeEach(() => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    mockScrollEl.clientHeight = 500;
    MockIntersectionObserver.lastCallback = null;
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders a QueueBadge when the row is the latest user in a queued session", () => {
    const userId = "user-q1";
    const turnStats = new Map([
      [
        userId,
        {
          isLatestUserMessage: true,
          rawMessageId: "uuid-q1",
        },
      ],
    ]);
    const messages = [
      {
        id: userId,
        role: "user" as const,
        parts: [{ type: "text" as const, text: "queue me", state: "done" }],
      },
    ];
    render(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <ChatMessagesContainer
        {...(baseProps as any)}
        messages={messages as any}
        turnStats={turnStats as any}
        sessionChatStatus="queued"
      />,
    );
    const badge = screen.getByTestId("queue-badge");
    expect(badge.getAttribute("data-session-id")).toBe("sess-123");
  });

  it("does NOT render a QueueBadge for normal (non-queued) user messages", () => {
    const userId = "user-n1";
    const turnStats = new Map([
      [userId, { createdAt: "2026-04-23T08:32:09.000Z" }],
    ]);
    const messages = [
      {
        id: userId,
        role: "user" as const,
        parts: [{ type: "text" as const, text: "hi", state: "done" }],
      },
    ];
    render(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <ChatMessagesContainer
        {...(baseProps as any)}
        messages={messages as any}
        turnStats={turnStats as any}
      />,
    );
    expect(screen.queryByTestId("queue-badge")).toBeNull();
  });

  it("does NOT render the badge when isLatestUserMessage but session is idle", () => {
    // Guards against regressing the AND-gate: even if a row is the
    // latest user message, the badge should stay hidden unless the
    // OWNING session is in the queued state.
    const userId = "user-q2";
    const turnStats = new Map([
      [userId, { isLatestUserMessage: true, rawMessageId: "uuid-q2" }],
    ]);
    const messages = [
      {
        id: userId,
        role: "user" as const,
        parts: [{ type: "text" as const, text: "live", state: "done" }],
      },
    ];
    render(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      <ChatMessagesContainer
        {...(baseProps as any)}
        messages={messages as any}
        turnStats={turnStats as any}
        sessionChatStatus="idle"
      />,
    );
    expect(screen.queryByTestId("queue-badge")).toBeNull();
  });
});
