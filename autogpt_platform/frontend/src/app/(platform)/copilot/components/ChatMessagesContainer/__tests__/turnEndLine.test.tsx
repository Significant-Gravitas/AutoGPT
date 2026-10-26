import type { ComponentProps, ReactNode } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CANCELLED_MARKER } from "../../../useCopilotStop";
import { ChatMessagesContainer } from "../ChatMessagesContainer";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

// The end of a turn is the subject here, so the rows and their turn-end line
// stay real; everything that only adds noise is stubbed.

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
  ChainMessageParts: () => <div data-testid="chain-message-parts" />,
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
vi.mock("../../PixelGridLoader/PixelGridLoader", () => ({
  PixelGridLoader: () => null,
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
  parts: [{ type: "text", text: "Make me 15 sticker designs" }],
};

function assistant(parts: unknown[]): Message {
  return {
    id: "assistant-1",
    role: "assistant",
    parts: parts as Message["parts"],
  };
}

const workflow = {
  type: "tool-run_agent",
  toolCallId: "call-1",
  input: {},
};
const runningWorkflow = assistant([{ ...workflow, state: "input-available" }]);
const ranWorkflow = assistant([
  { ...workflow, state: "output-available", output: {} },
]);

const baseProps = {
  messages: [userMessage, runningWorkflow],
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
  act(() => {
    vi.advanceTimersByTime(seconds * 1_000);
  });
}

function announcement(container: HTMLElement) {
  return container.querySelector("[aria-live='polite'].sr-only")?.textContent;
}

describe("ChatMessagesContainer — end of a turn", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
    vi.setSystemTime(NOW);
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("ends a turn that only ran a workflow with a done line, and announces it", () => {
    const { container, rerender } = render(
      <ChatMessagesContainer {...baseProps} />,
    );
    expect(screen.queryByTestId("turn-end-line")).toBeNull();
    advanceSeconds(95);

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        messages={[userMessage, ranWorkflow]}
        status="ready"
      />,
    );

    expect(screen.getByTestId("turn-end-line").textContent).toMatch(
      /^Done in 1m 35s·\s*1 workflow run$/,
    );
    expect(announcement(container)).toBe("Reply finished");
  });

  it("says stopped, and announces nothing, when the user stops the turn", () => {
    const { container, rerender } = render(
      <ChatMessagesContainer {...baseProps} />,
    );
    advanceSeconds(8);

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        messages={[
          userMessage,
          assistant([
            { ...workflow, state: "output-error", errorText: "cancelled" },
            { type: "text", text: CANCELLED_MARKER },
          ]),
        ]}
        status="ready"
      />,
    );

    const line = screen.getByTestId("turn-end-line");
    expect(line.textContent).toMatch(/^Stopped after 8s/);
    expect(line.textContent).not.toMatch(/Done/);
    expect(announcement(container)).toBe("");
  });

  it("does not announce a turn loaded from history", () => {
    const { container } = render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[userMessage, ranWorkflow]}
        status="ready"
        turnStats={new Map([["assistant-1", { durationMs: 42_000 }]])}
      />,
    );

    expect(screen.getByTestId("turn-end-line").textContent).toMatch(
      /^Done in 42s/,
    );
    expect(announcement(container)).toBe("");
  });
});
