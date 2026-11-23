import type { ReactNode } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChatMessagesContainer } from "../ChatMessagesContainer";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

// When a stream ends, hydration swaps the turn's client ids for the stored
// rows' `<session>-seq-N` ids. The rows and their parts are real here, so a
// remount shows: the expanded chain collapses and the nodes are new.

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
vi.mock("../../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));
vi.mock("../../ChatMinimap/ChatMinimap", () => ({
  ChatMinimap: () => null,
}));

const SESSION = "session-1";
const PROMPT = "Fetch the example page";
const ANSWER = "The page says Example Domain.";

function fetchPart(): Message["parts"][number] {
  return {
    type: "tool-web_fetch",
    toolCallId: "call-fetch",
    state: "output-available",
    input: { url: "https://example.com" },
    output: { content: "<h1>Example Domain</h1>" },
  } as Message["parts"][number];
}

// As `useChat` holds the turn while it streams: client ids, step markers.
const streamed: Message[] = [
  {
    id: "client-user-1",
    role: "user",
    parts: [{ type: "text", text: PROMPT }],
  },
  {
    id: "ai-sdk-uuid-1",
    role: "assistant",
    parts: [
      { type: "step-start" },
      fetchPart(),
      { type: "step-start" },
      { type: "text", text: ANSWER, state: "done" },
    ],
  },
];

// The same turn as the stored rows hydrate it.
const hydrated: Message[] = [
  {
    id: `${SESSION}-seq-1`,
    role: "user",
    parts: [{ type: "text", text: PROMPT, state: "done" }],
  },
  {
    id: `${SESSION}-seq-4`,
    role: "assistant",
    parts: [fetchPart(), { type: "text", text: ANSWER, state: "done" }],
  },
];

function view(messages: Message[]) {
  return (
    <CopilotChatActionsProvider onSend={vi.fn()}>
      <ChatMessagesContainer
        messages={messages}
        status="ready"
        error={undefined}
        isLoading={false}
        sessionID={SESSION}
        isTurnSettling={false}
      />
    </CopilotChatActionsProvider>
  );
}

afterEach(() => {
  cleanup();
});

describe("the turn a stream just finished, when hydration replaces its ids", () => {
  it("keeps its rows mounted: an expanded chain stays expanded and the text keeps its node", async () => {
    const { rerender, container } = render(view(streamed));
    const toggle = await screen.findByRole("button", { expanded: false });
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    const answer = await screen.findByText(ANSWER);
    const prompt = screen.getByText(PROMPT);
    const assistantRow = container.querySelector(
      '[data-message-id="ai-sdk-uuid-1"]',
    );
    expect(assistantRow).not.toBeNull();

    rerender(view(hydrated));

    const toggleAfter = screen.getByRole("button", { expanded: true });
    expect(toggleAfter).toBe(toggle);
    expect(screen.getByText(ANSWER)).toBe(answer);
    expect(screen.getByText(PROMPT)).toBe(prompt);
    // The row reads the stored id from here on (ratings, the tail anchor).
    expect(assistantRow?.getAttribute("data-message-id")).toBe(
      `${SESSION}-seq-4`,
    );
  });
});
