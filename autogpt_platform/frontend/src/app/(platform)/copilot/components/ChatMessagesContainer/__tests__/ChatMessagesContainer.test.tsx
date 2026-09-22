import type { ComponentProps } from "react";
import { act } from "@testing-library/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatMessagesContainer } from "../ChatMessagesContainer";
import { buildKickoffMessage } from "../../../expertKickoff";
import type { TurnStatsMap } from "../../../helpers/convertChatSessionToUiMessages";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

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
    "data-message-id": messageId,
  }: {
    children: React.ReactNode;
    from?: string;
    "data-message-id"?: string;
  }) => (
    <div
      data-testid={`message-${from ?? "unknown"}`}
      data-from={from}
      data-message-id={messageId}
    >
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
vi.mock("../components/ChainMessageParts", () => ({
  ChainMessageParts: ({
    parts,
    isCurrentlyStreaming,
  }: {
    parts: unknown[];
    isCurrentlyStreaming?: boolean;
  }) => (
    <div
      data-testid="chain-message-parts"
      data-parts={JSON.stringify(parts)}
      data-streaming={String(!!isCurrentlyStreaming)}
    />
  ),
}));

vi.mock("../components/QueueBadge", () => ({
  QueueBadge: ({ sessionID }: { sessionID: string | null }) => (
    <span data-testid="queue-badge" data-session-id={sessionID ?? ""}>
      QueueBadge
    </span>
  ),
}));

vi.mock("../components/CopyButton", () => ({
  CopyButton: ({ text }: { text: string }) => (
    <span data-testid="copy-message" data-text={text} />
  ),
}));
vi.mock("../components/MessageAttachments", () => ({
  MessageAttachments: () => null,
}));
vi.mock("../components/MessagePartRenderer", () => ({
  MessagePartRenderer: ({ part }: { part: { type: string; text?: string } }) =>
    part.type === "text" ? <span>{part.text}</span> : null,
}));
vi.mock("../components/ReasoningCollapse", () => ({
  ReasoningCollapse: () => null,
}));
vi.mock("../components/ThinkingIndicator", () => ({
  ThinkingIndicator: ({ statusMessage }: { statusMessage?: string | null }) => (
    <div data-testid="thinking-indicator">{statusMessage ?? "thinking"}</div>
  ),
}));
vi.mock("../../ToolChain/ToolChain", () => ({
  ToolChain: () => <div data-testid="tool-chain" />,
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
  getLatestCompactionPhase: () => null,
  getTurnMessages: () => [],
  isChainableToolPart: () => false,
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
}));

vi.mock("@/components/atoms/LoadingSpinner/LoadingSpinner", () => ({
  LoadingSpinner: () => <div data-testid="loading-spinner" />,
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
  messages: [] as Message[],
  status: "ready",
  error: undefined,
  isLoading: false,
  sessionID: "sess-123",
  queuedMessages: [] as string[],
  hasMoreMessages: true,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  onRetry: vi.fn(),
} satisfies ComponentProps<typeof ChatMessagesContainer>;

describe("ChatMessagesContainer — assistant rendering", () => {
  const messages = [
    {
      id: "assistant-tools",
      role: "assistant" as const,
      parts: [{ type: "text" as const, text: "Done" }],
    },
  ];

  afterEach(() => {
    cleanup();
  });

  it("renders assistant messages through the chain renderer", () => {
    render(<ChatMessagesContainer {...baseProps} messages={messages} />);

    expect(screen.getByTestId("chain-message-parts")).toBeDefined();
  });

  it("shows the thinking indicator inside a submitted assistant turn", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        status="submitted"
      />,
    );

    expect(screen.getByTestId("thinking-indicator")).toBeDefined();
  });

  it("keeps the thinking indicator out of a read-only transcript", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        status="submitted"
        readOnly
      />,
    );

    expect(screen.queryByTestId("thinking-indicator")).toBeNull();
  });
});

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
    const turnStats: TurnStatsMap = new Map([
      [userId, { createdAt: "2026-04-23T08:32:09.000Z" }],
    ]);
    const messages: Message[] = [
      {
        id: userId,
        role: "user" as const,
        parts: [{ type: "text" as const, text: "hi", state: "done" }],
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        turnStats={turnStats}
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
    const messages: Message[] = [
      {
        id: "user-unknown",
        role: "user" as const,
        parts: [{ type: "text" as const, text: "hi", state: "done" }],
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        turnStats={new Map()}
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
    const turnStats: TurnStatsMap = new Map([
      [
        userId,
        {
          isLatestUserMessage: true,
          rawMessageId: "uuid-q1",
        },
      ],
    ]);
    const messages: Message[] = [
      {
        id: userId,
        role: "user" as const,
        parts: [
          { type: "text" as const, text: "queue me", state: "done" as const },
        ],
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        turnStats={turnStats}
        sessionChatStatus="queued"
      />,
    );
    const badge = screen.getByTestId("queue-badge");
    expect(badge.getAttribute("data-session-id")).toBe("sess-123");
  });

  it.each([false, true])(
    "shows and copies only the delegated task (readOnly=%s)",
    (readOnly) => {
      const preamble =
        "[Delegated task from Ari, a teammate on this user's team — not " +
        "the user. They cannot see your thread, so report the outcome in your " +
        "final message. If the task needs something only the user can " +
        "provide, say what is missing instead of guessing.]";
      const message = {
        id: "sess-123-seq-0",
        role: "user" as const,
        parts: [
          { type: "text" as const, text: `${preamble}\n\nDraft the update.` },
        ],
        metadata: readOnly
          ? undefined
          : {
              from_session_id: "3f2c1a9e-7b4d-4e8a-9c1b-2d3e4f5a6b7c",
              from_expert_name: "Ari",
            },
      };
      render(
        <ChatMessagesContainer
          {...baseProps}
          messages={[message]}
          readOnly={readOnly}
        />,
      );

      expect(screen.getByText("Draft the update.")).toBeDefined();
      expect(screen.queryByText(/They cannot see your thread/)).toBeNull();
      expect(screen.getByTestId("copy-message").dataset.text).toBe(
        "Draft the update.",
      );
      expect(screen.queryAllByTestId("sent-from-badge")).toHaveLength(
        readOnly ? 0 : 1,
      );
      expect(message.parts[0].text).toBe(`${preamble}\n\nDraft the update.`);
    },
  );

  it("renders a Sent from badge linking to the session a delegated message came from", () => {
    const messages: Message[] = [
      {
        id: "user-d1",
        role: "user" as const,
        parts: [{ type: "text" as const, text: "draft the ops update" }],
        metadata: {
          from_session_id: "3f2c1a9e-7b4d-4e8a-9c1b-2d3e4f5a6b7c",
          from_expert_id: "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d",
          from_expert_name: "Ari",
        },
      },
    ];
    render(<ChatMessagesContainer {...baseProps} messages={messages} />);
    const card = screen.getByTestId("sent-from-badge");
    expect(card.textContent).toContain("Sent from Ari");
    expect(card.getAttribute("href")).toBe(
      "/copilot?sessionId=3f2c1a9e-7b4d-4e8a-9c1b-2d3e4f5a6b7c",
    );
  });

  const sessionSentFrom = {
    sessionId: "3f2c1a9e-7b4d-4e8a-9c1b-2d3e4f5a6b7c",
    expertId: null,
    expertName: null,
  };

  function userRow(id: string, text: string) {
    return {
      id,
      role: "user" as const,
      parts: [{ type: "text" as const, text }],
    };
  }

  it("falls back to the session's delegation provenance on the opening message only", () => {
    const messages: Message[] = [
      userRow("sess-123-seq-0", "first task"),
      {
        id: "sess-123-seq-1",
        role: "assistant" as const,
        parts: [{ type: "text" as const, text: "done" }],
      },
      userRow("sess-123-seq-2", "typed by the user"),
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        hasMoreMessages={false}
        sessionSentFrom={sessionSentFrom}
      />,
    );
    const cards = screen.getAllByTestId("sent-from-badge");
    expect(cards).toHaveLength(1);
    expect(cards[0].textContent).toContain("Sent from Otto");
  });

  it("does NOT fall back onto a later row when pagination stopped short of the opening row", () => {
    // hasMoreMessages goes false after repeated load errors or at the
    // history cap while older rows still exist; the first retained row is
    // then a later human message, not the one the delegation opened with.
    const messages: Message[] = [
      userRow("sess-123-seq-40", "typed by the user"),
      {
        id: "sess-123-seq-41",
        role: "assistant" as const,
        parts: [{ type: "text" as const, text: "sure" }],
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        hasMoreMessages={false}
        sessionSentFrom={sessionSentFrom}
      />,
    );
    expect(screen.queryByTestId("sent-from-badge")).toBeNull();
  });

  it("keeps the fallback on the opening row while more history is still loadable", () => {
    // The pagination flag flips true during a refetch; the opening row's
    // identity does not depend on it.
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[userRow("sess-123-seq-0", "first task")]}
        hasMoreMessages={true}
        sessionSentFrom={sessionSentFrom}
      />,
    );
    expect(screen.getByTestId("sent-from-badge").textContent).toContain(
      "Sent from Otto",
    );
  });

  it("does NOT fall back onto a row whose id carries no DB sequence", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[userRow("user-streamed", "first task")]}
        hasMoreMessages={false}
        sessionSentFrom={sessionSentFrom}
      />,
    );
    expect(screen.queryByTestId("sent-from-badge")).toBeNull();
  });

  it("does NOT fall back when the session carries no delegation provenance", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[userRow("sess-123-seq-0", "hello")]}
        hasMoreMessages={false}
        sessionSentFrom={null}
      />,
    );
    expect(screen.queryByTestId("sent-from-badge")).toBeNull();
  });

  it("does NOT render a Sent from badge for an ordinary user message", () => {
    const messages: Message[] = [
      {
        id: "user-plain",
        role: "user" as const,
        parts: [{ type: "text" as const, text: "hello" }],
      },
    ];
    render(<ChatMessagesContainer {...baseProps} messages={messages} />);
    expect(screen.queryByTestId("sent-from-badge")).toBeNull();
  });

  it("does NOT render a QueueBadge for normal (non-queued) user messages", () => {
    const userId = "user-n1";
    const turnStats: TurnStatsMap = new Map([
      [userId, { createdAt: "2026-04-23T08:32:09.000Z" }],
    ]);
    const messages: Message[] = [
      {
        id: userId,
        role: "user" as const,
        parts: [{ type: "text" as const, text: "hi", state: "done" }],
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        turnStats={turnStats}
      />,
    );
    expect(screen.queryByTestId("queue-badge")).toBeNull();
  });

  it("does NOT render the badge when isLatestUserMessage but session is idle", () => {
    // Guards against regressing the AND-gate: even if a row is the
    // latest user message, the badge should stay hidden unless the
    // OWNING session is in the queued state.
    const userId = "user-q2";
    const turnStats: TurnStatsMap = new Map([
      [userId, { isLatestUserMessage: true, rawMessageId: "uuid-q2" }],
    ]);
    const messages: Message[] = [
      {
        id: userId,
        role: "user" as const,
        parts: [{ type: "text" as const, text: "live", state: "done" }],
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        turnStats={turnStats}
        sessionChatStatus="idle"
      />,
    );
    expect(screen.queryByTestId("queue-badge")).toBeNull();
  });
});

// ── readOnly viewer behaviour ─────────────────────────────────────────────
//
// The shared-chat viewer (``/share/chat/[token]``) renders the same
// ChatMessagesContainer with ``readOnly`` so the public viewer cannot
// trigger any owner-only interactions (load-more, queue badges,
// feedback actions, error banners).  These tests pin the gate.

describe("ChatMessagesContainer — readOnly mode", () => {
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

  it.each([false, true])(
    "applies streamed tool names before stripping bookkeeping parts (readOnly=%s)",
    (readOnly) => {
      render(
        <ChatMessagesContainer
          {...baseProps}
          readOnly={readOnly}
          messages={[
            {
              id: "assistant-display",
              role: "assistant",
              parts: [
                {
                  type: "tool-run_agent",
                  toolCallId: "call-one",
                  state: "input-available",
                  input: { library_agent_id: "library-id" },
                },
                {
                  type: "data-tool-display",
                  id: "call-one",
                  data: {
                    toolCallId: "call-one",
                    displayName: "Daily briefing",
                  },
                },
              ],
            },
          ]}
        />,
      );
      const renderedParts = JSON.parse(
        screen.getByTestId("chain-message-parts").dataset.parts ?? "[]",
      );
      expect(renderedParts).toEqual([
        {
          type: "tool-run_agent",
          toolCallId: "call-one",
          state: "input-available",
          input: { library_agent_id: "library-id" },
          title: "Daily briefing",
        },
      ]);
    },
  );

  it("renders no Sent from badge in a read-only transcript", () => {
    const messages: Message[] = [
      {
        id: "sess-123-seq-0",
        role: "user" as const,
        parts: [{ type: "text" as const, text: "first task" }],
      },
      {
        id: "sess-123-seq-1",
        role: "assistant" as const,
        parts: [{ type: "text" as const, text: "done" }],
      },
      {
        id: "sess-123-seq-2",
        role: "user" as const,
        parts: [{ type: "text" as const, text: "draft the ops update" }],
        metadata: {
          from_session_id: "3f2c1a9e-7b4d-4e8a-9c1b-2d3e4f5a6b7c",
          from_expert_id: null,
          from_expert_name: "Ari",
        },
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        readOnly
        hasMoreMessages={false}
        sessionSentFrom={{
          sessionId: "3f2c1a9e-7b4d-4e8a-9c1b-2d3e4f5a6b7c",
          expertId: null,
          expertName: null,
        }}
      />,
    );
    expect(screen.queryByTestId("sent-from-badge")).toBeNull();
  });

  it("hides the load-older-messages sentinel even when more history exists", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        hasMoreMessages
        onLoadMore={vi.fn()}
        readOnly
      />,
    );
    expect(
      screen.queryByRole("button", { name: /load older messages/i }),
    ).toBeNull();
  });

  it("hides queued messages even when queuedMessages is non-empty", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        queuedMessages={["should not show"]}
        readOnly
      />,
    );
    expect(screen.queryByText("should not show")).toBeNull();
    expect(screen.queryByText("Queued")).toBeNull();
  });

  it("hides the trailing error banner even on error status", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        error={new Error("upstream blew up")}
        status="error"
        readOnly
        messages={[
          {
            id: "u-1",
            role: "user",
            parts: [{ type: "text", text: "go" }],
          },
        ]}
      />,
    );
    expect(screen.queryByText(/encountered an error/i)).toBeNull();
    expect(screen.queryByText(/upstream blew up/i)).toBeNull();
  });

  it("hides the queue-badge gate even when the session is queued", () => {
    const userId = "user-q-readonly";
    const turnStats: TurnStatsMap = new Map([
      [userId, { isLatestUserMessage: true, rawMessageId: "uuid-q-ro" }],
    ]);
    const messages: Message[] = [
      {
        id: userId,
        role: "user" as const,
        parts: [
          { type: "text" as const, text: "queue me", state: "done" as const },
        ],
      },
    ];
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={messages}
        turnStats={turnStats}
        sessionChatStatus="queued"
        readOnly
      />,
    );
    expect(screen.queryByTestId("queue-badge")).toBeNull();
  });
});

// ── expert kickoff ────────────────────────────────────────────────────────

describe("ChatMessagesContainer — expert kickoff", () => {
  it("hides the kickoff prompt and shows only the reply", () => {
    const kickoff = buildKickoffMessage("3f8b0f7e-9f30-4a3b-a6a1-000000000001");
    render(
      <ChatMessagesContainer
        {...baseProps}
        hasMoreMessages={false}
        messages={[
          {
            id: "m1",
            role: "user",
            parts: [{ type: "text", text: kickoff.text }],
            metadata: kickoff.metadata,
          },
          {
            id: "m2",
            role: "assistant",
            parts: [{ type: "text", text: "Hi, I'm Maria." }],
          },
        ]}
      />,
    );

    expect(screen.queryByTestId("message-user")).toBeNull();
    expect(screen.queryByText(kickoff.text)).toBeNull();
    expect(screen.getAllByTestId("message-assistant").length).toBeGreaterThan(
      0,
    );
  });

  it("keeps a user message that merely repeats the kickoff wording", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        hasMoreMessages={false}
        messages={[
          {
            id: "m1",
            role: "user",
            parts: [
              {
                type: "text",
                text: "You were just hired. Call expert_onboarding once, and nothing else",
              },
            ],
          },
        ]}
      />,
    );

    expect(screen.getAllByTestId("message-user")).toHaveLength(1);
  });
});

// ── pending upload placeholder ────────────────────────────────────────────

describe("ChatMessagesContainer — pendingSend", () => {
  const pendingSend = {
    text: "tell me about this",
    attachments: [
      {
        name: "talk.pdf",
        mediaType: "application/pdf",
        sizeBytes: 2048,
        isUploading: true,
      },
      { name: "icon.png", mediaType: "image/png", isUploading: true },
    ],
  };

  beforeEach(() => {
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the sent text and attachments before the message reaches the transcript", () => {
    render(<ChatMessagesContainer {...baseProps} pendingSend={pendingSend} />);

    expect(screen.getByText("tell me about this")).toBeDefined();
    expect(screen.getByText("talk.pdf")).toBeDefined();
    expect(screen.getByText("icon.png")).toBeDefined();
  });

  it("narrates the upload in the thinking indicator", () => {
    render(<ChatMessagesContainer {...baseProps} pendingSend={pendingSend} />);

    expect(screen.getByTestId("thinking-indicator").textContent).toBe(
      "Uploading 2 files…",
    );
  });

  it("uses the singular for a single file", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        pendingSend={{
          ...pendingSend,
          attachments: pendingSend.attachments.slice(0, 1),
        }}
      />,
    );

    expect(screen.getByTestId("thinking-indicator").textContent).toBe(
      "Uploading 1 file…",
    );
  });

  it("does not show the history spinner while the placeholder is up", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        isLoading
        pendingSend={pendingSend}
      />,
    );

    expect(screen.getByText("tell me about this")).toBeDefined();
    expect(screen.queryByTestId("loading-spinner")).toBeNull();
  });

  it("keeps the placeholder out of a read-only transcript", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        pendingSend={pendingSend}
        readOnly
      />,
    );

    expect(screen.queryByText("tell me about this")).toBeNull();
    expect(screen.queryByTestId("thinking-indicator")).toBeNull();
  });

  it("still shows the history spinner in a read-only transcript", () => {
    // The placeholder is suppressed here, so nothing stands in for the
    // spinner — hiding it would leave the transcript blank while it loads.
    render(
      <ChatMessagesContainer
        {...baseProps}
        isLoading
        pendingSend={pendingSend}
        readOnly
      />,
    );

    expect(screen.getByTestId("loading-spinner")).toBeDefined();
  });
});

// ── mid-turn drain split ──────────────────────────────────────────────────

describe("ChatMessagesContainer — mid-turn follow-up", () => {
  afterEach(cleanup);

  const toolPart = {
    type: "tool-read_file",
    toolCallId: "call-1",
    state: "output-available",
    input: {},
    output: "ok",
  };

  const drainedTurn = [
    {
      id: "user-1",
      role: "user" as const,
      parts: [{ type: "text" as const, text: "plan my week" }],
    },
    {
      id: "assistant-1",
      role: "assistant" as const,
      parts: [
        toolPart,
        {
          type: "data-pending-drained",
          id: "hint-0",
          data: {
            drainedCount: 1,
            messages: [{ id: "pm-1", content: "also check Friday" }],
          },
        },
        { ...toolPart, toolCallId: "call-2" },
      ],
    },
  ] as unknown as UIMessage<unknown, UIDataTypes, UITools>[];

  function renderedRowIds() {
    return Array.from(document.querySelectorAll("[data-message-id]")).map(
      (el) => el.getAttribute("data-message-id"),
    );
  }

  it("renders the drained message between the work before and after it", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={drainedTurn}
      />,
    );

    expect(renderedRowIds()).toEqual([
      "user-1",
      "assistant-1#seg0",
      "midturn-pm-1",
      "assistant-1",
    ]);
    expect(
      Array.from(document.querySelectorAll('[data-testid="message-user"]')),
    ).toHaveLength(2);
  });

  it("settles the chain above the follow-up and streams only the last one", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={drainedTurn}
      />,
    );

    expect(
      screen
        .getAllByTestId("chain-message-parts")
        .map((el) => el.getAttribute("data-streaming")),
    ).toEqual(["false", "true"]);
  });

  it("draws a promoted fallback bubble once its hint carries the text", () => {
    // The backstop GET promoted the chip above the assistant before the SSE
    // hint arrived; the hint then draws the same follow-up at the drain
    // point. Exactly one bubble, between the two assistant segments.
    const withFallback = [
      drainedTurn[0],
      {
        id: "promoted-midturn-pending-chip-local-1",
        role: "user" as const,
        parts: [{ type: "text" as const, text: "also check Friday" }],
      },
      drainedTurn[1],
    ] as unknown as UIMessage<unknown, UIDataTypes, UITools>[];

    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={withFallback}
      />,
    );

    expect(renderedRowIds()).toEqual([
      "user-1",
      "assistant-1#seg0",
      "midturn-pm-1",
      "assistant-1",
    ]);
    // The prompt and one follow-up: the fallback row is not drawn as well.
    expect(screen.getAllByTestId("message-user")).toHaveLength(2);
  });

  it("keeps a text-less hint out of the tool chain it lands in", () => {
    // The hint is stream bookkeeping: left in the parts, it would split the
    // chain around it into two.
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={
          [
            drainedTurn[0],
            {
              ...drainedTurn[1],
              parts: [
                drainedTurn[1].parts[0],
                {
                  type: "data-pending-drained",
                  id: "hint-0",
                  data: { drainedCount: 1 },
                },
                drainedTurn[1].parts[2],
              ],
            },
          ] as unknown as UIMessage<unknown, UIDataTypes, UITools>[]
        }
      />,
    );

    const renderedParts = JSON.parse(
      screen.getByTestId("chain-message-parts").dataset.parts ?? "[]",
    ) as { type: string }[];
    expect(renderedParts.map((p) => p.type)).toEqual([
      "tool-read_file",
      "tool-read_file",
    ]);
  });

  it("shows the thinking indicator while the follow-up is the newest thing in the turn", () => {
    // Answer text is normally inflight, but a drain hint landing after it
    // means the assistant has not produced anything for the follow-up yet.
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={
          [
            drainedTurn[0],
            {
              ...drainedTurn[1],
              parts: [
                { type: "text", text: "Here is your week.", state: "done" },
                drainedTurn[1].parts[1],
              ],
            },
          ] as unknown as UIMessage<unknown, UIDataTypes, UITools>[]
        }
      />,
    );

    expect(renderedRowIds()).toEqual([
      "user-1",
      "assistant-1#seg0",
      "midturn-pm-1",
      "assistant-1",
    ]);
    expect(screen.getByTestId("thinking-indicator")).toBeDefined();
  });

  it("draws one follow-up when the promoted bubble sits behind the stream's placeholder row", () => {
    // The live shape: the prompt, the status-only placeholder `useChat`
    // leaves before the server's message id arrives, the bubble the
    // auto-continue effect promoted, then the assistant whose hint carries
    // the same text.
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={
          [
            drainedTurn[0],
            {
              id: "placeholder-1",
              role: "assistant",
              parts: [{ type: "data-status", data: { message: "Preparing…" } }],
            },
            {
              id: "promoted-auto-continue-pending-chip-local-1",
              role: "user",
              parts: [{ type: "text", text: "also check Friday" }],
            },
            drainedTurn[1],
          ] as unknown as UIMessage<unknown, UIDataTypes, UITools>[]
        }
      />,
    );

    expect(renderedRowIds()).toEqual([
      "user-1",
      "placeholder-1",
      "assistant-1#seg0",
      "midturn-pm-1",
      "assistant-1",
    ]);
    expect(screen.getAllByTestId("message-user")).toHaveLength(2);
  });

  it("leaves the turn whole when the hint carries no text", () => {
    const withoutText = [
      drainedTurn[0],
      {
        ...drainedTurn[1],
        parts: [
          drainedTurn[1].parts[0],
          {
            type: "data-pending-drained",
            id: "hint-0",
            data: { drainedCount: 1 },
          },
          drainedTurn[1].parts[2],
        ],
      },
    ] as unknown as UIMessage<unknown, UIDataTypes, UITools>[];

    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={withoutText}
      />,
    );

    expect(renderedRowIds()).toEqual(["user-1", "assistant-1"]);
  });
});
