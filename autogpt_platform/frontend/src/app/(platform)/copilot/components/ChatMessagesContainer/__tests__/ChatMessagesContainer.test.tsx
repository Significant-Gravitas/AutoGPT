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
vi.mock("../helpers", () => ({
  buildRenderSegments: () => [],
  getTurnMessages: () => [],
  parseSpecialMarkers: () => ({ markerType: null }),
  splitReasoningAndResponse: (parts: unknown[]) => ({
    reasoningParts: [],
    responseParts: parts,
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
