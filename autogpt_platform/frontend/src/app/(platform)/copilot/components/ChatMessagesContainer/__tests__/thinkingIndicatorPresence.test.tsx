import type { ComponentProps, ReactNode } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatMessagesContainer } from "../ChatMessagesContainer";
import {
  EXIT_MS,
  MIN_VISIBLE_MS,
  STEP_GAP_MS,
  TEXT_IDLE_MS,
} from "../components/ThinkingSlot/helpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

// The indicator's presence is the subject, so ThinkingIndicator and its slot
// stay real; the rest of the transcript is stubbed down to plain text.
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
    className,
  }: {
    children: ReactNode;
    className?: string;
  }) => <div className={className}>{children}</div>,
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
vi.mock("../components/CopyButton", () => ({ CopyButton: () => null }));
vi.mock("../components/QueueBadge", () => ({ QueueBadge: () => null }));
vi.mock("../components/MessageAttachments", () => ({
  MessageAttachments: () => null,
}));
vi.mock("../components/ChainMessageParts", () => ({
  ChainMessageParts: () => null,
}));
vi.mock("../components/MessagePartRenderer", () => ({
  MessagePartRenderer: () => null,
}));
vi.mock("../../ToolChain/SwapText", () => ({
  SwapText: ({ text }: { text: string }) => <span>{text}</span>,
}));
vi.mock("../../DrawnLineLoader/DrawnLineLoader", () => ({
  DrawnLineLoader: () => null,
}));
vi.mock("../../JobStatsBar/TurnStatsBar", () => ({ TurnStatsBar: () => null }));
vi.mock("../../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));
vi.mock("../../ChatMinimap/ChatMinimap", () => ({ ChatMinimap: () => null }));

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

const user: Message = {
  id: "u1",
  role: "user",
  parts: [{ type: "text", text: "go" }],
};

function assistant(parts: unknown[]): Message {
  return { id: "a1", role: "assistant", parts: parts as Message["parts"] };
}

const text = (value: string) => ({ type: "text", text: value });
const runningTool = {
  type: "tool-web_search",
  toolCallId: "call-1",
  state: "input-streaming",
};
const doneTool = {
  ...runningTool,
  state: "output-available",
  input: {},
  output: {},
};

type Props = ComponentProps<typeof ChatMessagesContainer>;
const baseProps = {
  messages: [user],
  status: "submitted",
  error: undefined,
  isLoading: false,
  sessionID: "sess-1",
} satisfies Props;

function indicator() {
  return screen.queryByRole("status");
}

function advance(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

// The fade starts from a render, so its timer is set only after the hold ends.
function holdThenFade() {
  advance(MIN_VISIBLE_MS);
  advance(EXIT_MS);
}

describe("ChatMessagesContainer — thinking indicator presence", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("keeps the same indicator mounted when the first chunk lands", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    const first = indicator();
    expect(first).not.toBeNull();

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={[user, assistant([{ type: "step-start" }])]}
      />,
    );

    expect(indicator()).toBe(first);
  });

  it("holds the indicator for its minimum time, then fades it out", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    advance(100);
    rerender(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={[user, assistant([text("Hello")])]}
      />,
    );
    expect(indicator()).not.toBeNull();

    advance(MIN_VISIBLE_MS - 100);
    const fading = indicator()?.closest(".opacity-0");
    expect(fading).not.toBeNull();

    advance(EXIT_MS);
    expect(indicator()).toBeNull();
  });

  it("comes back when the text stalls, and leaves when it grows again", () => {
    const streaming = (value: string) => (
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={[user, assistant([text(value)])]}
      />
    );
    const { rerender } = render(streaming("Hel"));
    advance(MIN_VISIBLE_MS + EXIT_MS);
    rerender(streaming("Hello"));
    expect(indicator()).toBeNull();

    advance(TEXT_IDLE_MS - 1);
    expect(indicator()).toBeNull();
    advance(1);
    expect(indicator()).not.toBeNull();

    rerender(streaming("Hello there"));
    holdThenFade();
    expect(indicator()).toBeNull();
  });

  it("shows nothing while a tool row is on screen, and returns between steps", () => {
    const { rerender } = render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={[user, assistant([runningTool])]}
      />,
    );
    expect(indicator()).toBeNull();

    rerender(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={[user, assistant([doneTool])]}
      />,
    );
    advance(STEP_GAP_MS - 1);
    expect(indicator()).toBeNull();
    advance(1);
    expect(indicator()).not.toBeNull();
  });

  it("draws nothing for a quick gap between two steps", () => {
    const turn = (parts: unknown[]) => (
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={[user, assistant(parts)]}
      />
    );
    const { rerender } = render(turn([runningTool]));
    rerender(turn([doneTool]));
    advance(STEP_GAP_MS / 2);
    rerender(turn([doneTool, { ...runningTool, toolCallId: "call-2" }]));
    advance(STEP_GAP_MS);

    expect(indicator()).toBeNull();
  });

  it("leaves once the stream ends", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    rerender(
      <ChatMessagesContainer
        {...baseProps}
        status="ready"
        messages={[user, assistant([text("Done.")])]}
      />,
    );
    expect(indicator()).not.toBeNull();
    holdThenFade();

    expect(indicator()).toBeNull();
  });

  it("never shows on a finished transcript", () => {
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="ready"
        messages={[user, assistant([doneTool, text("Done.")])]}
      />,
    );
    advance(TEXT_IDLE_MS * 2);

    expect(indicator()).toBeNull();
  });

  it("hands the wait to a restore at once, without a fade", () => {
    const { rerender } = render(<ChatMessagesContainer {...baseProps} />);
    rerender(<ChatMessagesContainer {...baseProps} isRestoringActiveSession />);

    expect(indicator()).toBeNull();
  });
});
