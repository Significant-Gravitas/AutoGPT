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
    <div>{children}</div>
  ),
  ConversationContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationScrollButton: () => null,
}));

vi.mock("@/components/ai-elements/message", () => ({
  Message: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  MessageContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  MessageActions: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
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
  ThinkingIndicator: () => null,
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

const BASE_PROPS = {
  messages: [],
  status: "ready" as const,
  error: undefined,
  isLoading: false,
  sessionID: "sess-1",
  hasMoreMessages: true,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  onRetry: vi.fn(),
};

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

  it("renders top sentinel when forwardPaginated is false (backward pagination)", () => {
    render(<ChatMessagesContainer {...BASE_PROPS} forwardPaginated={false} />);
    expect(
      screen.getByRole("button", { name: /load older messages/i }),
    ).toBeDefined();
  });

  it("renders top sentinel when forwardPaginated is undefined (default, backward)", () => {
    render(<ChatMessagesContainer {...BASE_PROPS} />);
    expect(
      screen.getByRole("button", { name: /load older messages/i }),
    ).toBeDefined();
  });

  it("renders bottom sentinel when forwardPaginated is true (forward pagination)", () => {
    render(<ChatMessagesContainer {...BASE_PROPS} forwardPaginated={true} />);
    expect(
      screen.getByRole("button", { name: /load newer messages/i }),
    ).toBeDefined();
  });

  it("hides sentinel when hasMoreMessages is false", () => {
    render(
      <ChatMessagesContainer
        {...BASE_PROPS}
        hasMoreMessages={false}
        forwardPaginated={true}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /load older messages/i }),
    ).toBeNull();
  });

  it("hides sentinel when onLoadMore is not provided", () => {
    render(
      <ChatMessagesContainer
        {...BASE_PROPS}
        onLoadMore={undefined}
        forwardPaginated={true}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /load older messages/i }),
    ).toBeNull();
  });
});
