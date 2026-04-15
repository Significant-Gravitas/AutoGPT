import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatMessagesContainer } from "../ChatMessagesContainer";

// ── heavy UI dependency mocks ──────────────────────────────────────────────

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

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({
    isAtBottom: true,
    scrollToBottom: vi.fn(),
  }),
}));

vi.mock("../components/ThinkingIndicator", () => ({
  ThinkingIndicator: () => null,
}));

vi.mock("../components/CollapsedToolGroup", () => ({
  CollapsedToolGroup: () => null,
}));

vi.mock("../components/MessagePartRenderer", () => ({
  MessagePartRenderer: () => null,
}));

vi.mock("../components/ReasoningCollapse", () => ({
  ReasoningCollapse: () => null,
}));

vi.mock("../components/AssistantMessageActions", () => ({
  AssistantMessageActions: () => null,
}));

vi.mock("../components/CopyButton", () => ({
  CopyButton: () => null,
}));

vi.mock("../components/MessageAttachments", () => ({
  MessageAttachments: () => null,
}));

vi.mock("../components/FeedbackModal", () => ({
  FeedbackModal: () => null,
}));

vi.mock("../components/TTSButton", () => ({
  TTSButton: () => null,
}));

vi.mock("@/components/atoms/LoadingSpinner/LoadingSpinner", () => ({
  LoadingSpinner: () => <div data-testid="loading-spinner" />,
}));

vi.mock("../JobStatsBar/TurnStatsBar", () => ({
  TurnStatsBar: () => null,
}));

vi.mock("../JobStatsBar/useElapsedTimer", () => ({
  useElapsedTimer: () => ({ elapsedSeconds: 0 }),
}));

vi.mock("../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));

vi.mock("@phosphor-icons/react", () => ({
  Clock: () => <span data-testid="clock-icon" />,
  ArrowDown: () => null,
  ArrowUp: () => null,
}));

// ── helpers ───────────────────────────────────────────────────────────────

const baseProps = {
  messages: [] as any[],
  status: "ready",
  error: undefined,
  isLoading: false,
  sessionID: "sess-123",
  queuedMessages: [] as string[],
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

// ── queued-messages rendering ─────────────────────────────────────────────

describe("ChatMessagesContainer — queuedMessages", () => {
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
  it("shows loading spinner when isLoading is true", () => {
    render(<ChatMessagesContainer {...baseProps} isLoading />);
    expect(screen.getByTestId("loading-spinner")).toBeDefined();
  });

  it("does not show spinner when not loading", () => {
    render(<ChatMessagesContainer {...baseProps} isLoading={false} />);
    expect(screen.queryByTestId("loading-spinner")).toBeNull();
  });
});
