import type { ComponentProps, ReactNode } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { TurnStatsMap } from "../../../helpers/convertChatSessionToUiMessages";
import { ChatMessagesContainer } from "../ChatMessagesContainer";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

// Each row draws its parts through one of these two, so counting their
// renders per message id counts which transcript rows rendered.
const rowRenders = new Map<string, number>();

function countRender(messageID: string) {
  rowRenders.set(messageID, (rowRenders.get(messageID) ?? 0) + 1);
}

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
  Message: ({
    children,
    "data-message-id": messageID,
  }: {
    children: ReactNode;
    "data-message-id"?: string;
  }) => <div data-message-id={messageID}>{children}</div>,
  MessageActions: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  MessageContent: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("../components/ChainMessageParts", () => ({
  ChainMessageParts: ({
    messageID,
    parts,
  }: {
    messageID: string;
    parts: { type: string; text?: string }[];
  }) => {
    countRender(messageID);
    return (
      <div data-testid={`chain-${messageID}`}>
        {parts.map((part) => part.text ?? "").join("")}
      </div>
    );
  },
}));
vi.mock("../components/MessagePartRenderer", () => ({
  MessagePartRenderer: ({
    messageID,
    part,
  }: {
    messageID: string;
    part: { type: string; text?: string };
  }) => {
    countRender(messageID);
    return part.type === "text" ? <span>{part.text}</span> : null;
  },
}));
vi.mock("../components/AssistantMessageActions", () => ({
  AssistantMessageActions: ({ message }: { message: Message }) => (
    <div data-testid={`actions-${message.id}`} />
  ),
}));
vi.mock("../components/CopyButton", () => ({
  CopyButton: () => null,
}));
vi.mock("../components/QueueBadge", () => ({
  QueueBadge: () => <span data-testid="queue-badge" />,
}));
vi.mock("../components/MessageAttachments", () => ({
  MessageAttachments: () => null,
}));
vi.mock("../../JobStatsBar/TurnStatsBar", () => ({
  TurnStatsBar: () => null,
}));
vi.mock("../../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));
vi.mock("../../ChatMinimap/ChatMinimap", () => ({
  ChatMinimap: () => null,
}));
vi.mock("../../DrawnLineLoader/DrawnLineLoader", () => ({
  DrawnLineLoader: () => null,
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

function buildTranscript(count: number): Message[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `m-${i}`,
    role: i % 2 === 0 ? "user" : "assistant",
    parts: [{ type: "text", text: `message ${i}` }],
  }));
}

function appendToken(messages: Message[], token: string): Message[] {
  const last = messages[messages.length - 1];
  const part = last.parts[0] as { type: "text"; text: string };
  return [
    ...messages.slice(0, -1),
    { ...last, parts: [{ ...part, text: part.text + token }] },
  ];
}

function renderedRows(): string[] {
  return [...rowRenders.keys()];
}

const baseProps = {
  messages: buildTranscript(50),
  status: "streaming",
  error: undefined,
  isLoading: false,
  sessionID: "sess-1",
  sessionChatStatus: "running",
  queuedMessages: [] as string[],
  hasMoreMessages: false,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  onRetry: vi.fn(),
} satisfies ComponentProps<typeof ChatMessagesContainer>;

describe("ChatMessagesContainer — memoised transcript rows", () => {
  beforeEach(() => {
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
    rowRenders.clear();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("re-renders only the streaming row when a token lands", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    expect(rowRenders.size).toBe(50);
    rowRenders.clear();

    const next = appendToken(baseProps.messages, " more");
    rerender(<ChatMessagesContainer {...baseProps} messages={next} />);

    expect(renderedRows()).toEqual(["m-49"]);
    expect(screen.getByTestId("chain-m-49").textContent).toBe(
      "message 49 more",
    );
  });

  it("re-renders no row when the parent re-renders with the same messages", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    rowRenders.clear();

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        sessionChatStatus="idle"
        turnStats={new Map()}
      />,
    );

    expect(renderedRows()).toEqual([]);
  });

  it("re-renders a row whose turn stats changed, and no other", () => {
    const stats = (createdAt: string): TurnStatsMap =>
      new Map([
        ["m-10", { createdAt }],
        ["m-20", { createdAt: "2026-10-01T10:00:00Z" }],
      ]);
    const { rerender } = render(
      <ChatMessagesContainer
        {...baseProps}
        status="ready"
        turnStats={stats("2026-10-01T09:00:00Z")}
      />,
    );
    rowRenders.clear();

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        status="ready"
        turnStats={stats("2026-10-01T09:00:00Z")}
      />,
    );
    expect(renderedRows()).toEqual([]);

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        status="ready"
        turnStats={stats("2026-10-02T09:00:00Z")}
      />,
    );
    expect(renderedRows()).toEqual(["m-10"]);
  });

  it("still updates the last row when the stream ends", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    expect(screen.queryByTestId("actions-m-49")).toBeNull();
    rowRenders.clear();

    rerender(<ChatMessagesContainer {...baseProps} status="ready" />);

    expect(renderedRows()).toEqual(["m-49"]);
    expect(screen.getByTestId("actions-m-49")).toBeDefined();
  });

  it("still shows the queue badge on the latest user row once the session queues", () => {
    const turnStats: TurnStatsMap = new Map([
      ["m-48", { isLatestUserMessage: true }],
    ]);
    const { rerender } = render(
      <ChatMessagesContainer
        {...baseProps}
        status="ready"
        turnStats={turnStats}
      />,
    );
    expect(screen.queryByTestId("queue-badge")).toBeNull();
    rowRenders.clear();

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        status="ready"
        sessionChatStatus="queued"
        turnStats={turnStats}
      />,
    );

    expect(screen.getByTestId("queue-badge")).toBeDefined();
    expect(renderedRows().every((id) => Number(id.slice(2)) % 2 === 0)).toBe(
      true,
    );
  });
});
