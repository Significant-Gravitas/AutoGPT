import { useState } from "react";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getGetV2GetSessionMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetV2GetPendingReviewsForChatSessionMockHandler } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { ChatContainer } from "../components/ChatContainer/ChatContainer";

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <textarea aria-label="Chat input" />,
}));

vi.mock(
  "@/app/(platform)/copilot/components/UsageLimits/useIsUsageLimitReached",
  () => ({ useIsUsageLimitReached: () => false }),
);

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => ({
  ...(await importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
}));

type Message = UIMessage<unknown, UIDataTypes, UITools>;

const SESSION_ID = "decision-session";
const CONFIRMATION_ID = "2616d1f1-16bb-43c1-92ad-61014ac9d123";

// What hire_expert returns (ExpertChangeProposedResponse).
const ask: Message = {
  id: `${SESSION_ID}-seq-1`,
  role: "user",
  parts: [{ type: "text", text: "Get someone to run Dewleaf's marketing." }],
};
const hireCard: Message = {
  id: `${SESSION_ID}-seq-2`,
  role: "assistant",
  parts: [
    {
      type: "tool-hire_expert",
      toolCallId: "call-hire",
      state: "output-available",
      input: { template_id: "tpl-marketing", name: "Maya" },
      output: {
        type: "expert_change_proposed",
        message: "Nothing hired yet.",
        session_id: SESSION_ID,
        applied: false,
        confirmation_id: CONFIRMATION_ID,
        preview: {
          kind: "hire",
          name: "Maya",
          role: "Marketing",
          tagline: "Plans and runs your launches.",
        },
      },
    } as ToolUIPart,
    { type: "text", text: "Maya can run the launch." },
  ],
};

// Two hires in one turn, one card that pages between them.
const teamCard: Message = {
  id: `${SESSION_ID}-seq-2`,
  role: "assistant",
  parts: [
    hireCard.parts[0],
    {
      type: "tool-hire_expert",
      toolCallId: "call-hire-theo",
      state: "output-available",
      input: { template_id: "tpl-ops" },
      output: {
        type: "expert_change_proposed",
        message: "Nothing hired yet.",
        session_id: SESSION_ID,
        applied: false,
        confirmation_id: "c-theo",
        preview: { kind: "hire", name: "Theo", role: "Ops" },
      },
    } as ToolUIPart,
  ],
};

function userRow(text: string, metadata?: unknown): Message {
  return {
    id: `${SESSION_ID}-seq-3`,
    role: "user",
    ...(metadata ? { metadata } : {}),
    parts: [{ type: "text", text }],
  };
}

// Stands in for useChat: a sent message lands in the thread as the user's
// own row straight away, before the server's copy replaces it.
function Thread({ initial }: { initial: Message[] }) {
  const [messages, setMessages] = useState(initial);
  function handleSend(text: string) {
    setMessages([...messages, userRow(text)]);
  }
  return (
    <ChatContainer
      messages={messages}
      status="ready"
      error={undefined}
      sessionId={SESSION_ID}
      isLoadingSession={false}
      isCreatingSession={false}
      onCreateSession={vi.fn()}
      onSend={handleSend}
      onStop={vi.fn()}
    />
  );
}

// Stands in for a send while a turn runs: the decision waits in the queue,
// shown as a queued chip, and the send resolves before any row exists.
function QueuedThread({ initial }: { initial: Message[] }) {
  const [queued, setQueued] = useState<string[]>([]);
  async function handleSend(text: string) {
    setQueued([...queued, text]);
  }
  return (
    <ChatContainer
      messages={initial}
      status="streaming"
      error={undefined}
      sessionId={SESSION_ID}
      isLoadingSession={false}
      isCreatingSession={false}
      onCreateSession={vi.fn()}
      onSend={handleSend}
      onStop={vi.fn()}
      queuedMessages={queued}
    />
  );
}

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

function expectNoProtocolText() {
  const page = document.body.textContent ?? "";
  expect(page).not.toContain("confirmation_id");
  expect(page).not.toContain(CONFIRMATION_ID);
  expect(page).not.toContain("Approved: hire");
}

describe("a proposal card's decision in the chat", () => {
  beforeEach(() => {
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
    server.use(getGetV2GetPendingReviewsForChatSessionMockHandler([]));
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("reads as a note of what the user approved, not a line with the id", async () => {
    const user = userEvent.setup();
    render(<Thread initial={[ask, hireCard]} />);

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    expect(await screen.findByText("You approved hiring Maya")).toBeDefined();
    expectNoProtocolText();
  });

  it("says what was declined", async () => {
    const user = userEvent.setup();
    render(<Thread initial={[ask, hireCard]} />);

    await user.click(screen.getByRole("button", { name: "Decline" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    expect(await screen.findByText("You declined hiring Maya")).toBeDefined();
    expect(document.body.textContent).not.toContain(CONFIRMATION_ID);
  });

  it("renders a saved row the server tagged from its metadata", () => {
    const tagged = userRow(
      `Approved: hire Maya (confirmation_id: ${CONFIRMATION_ID}).`,
      {
        kind: "card_decision",
        decisions: [
          {
            verdict: "approved",
            confirmation_id: CONFIRMATION_ID,
            verb: "hire",
            name: "Maya",
          },
        ],
      },
    );
    render(<Thread initial={[ask, hireCard, tagged]} />);

    expect(screen.getByText("You approved hiring Maya")).toBeDefined();
    expectNoProtocolText();
  });

  it("renders the same note for an older chat saved without the tag", () => {
    const legacy = userRow(
      `Approved: hire Maya (confirmation_id: ${CONFIRMATION_ID}).`,
    );
    render(<Thread initial={[ask, hireCard, legacy]} />);

    expect(screen.getByText("You approved hiring Maya")).toBeDefined();
    expectNoProtocolText();
  });

  it("keeps a message the user wrote around a decision line as their message", () => {
    const own = userRow(
      `Approved: hire Maya (confirmation_id: ${CONFIRMATION_ID}).\nAlso, start with Instagram.`,
    );
    render(<Thread initial={[ask, hireCard, own]} />);

    expect(screen.queryByText("You approved hiring Maya")).toBeNull();
    expect(document.body.textContent).toContain("Also, start with Instagram.");
  });
});

function Chat({
  messages,
  onSend = vi.fn(),
  onBackendTurn,
}: {
  messages: Message[];
  onSend?: (text: string) => void;
  onBackendTurn?: () => void;
}) {
  return (
    <ChatContainer
      messages={messages}
      status="ready"
      error={undefined}
      sessionId={SESSION_ID}
      isLoadingSession={false}
      isCreatingSession={false}
      onCreateSession={vi.fn()}
      onSend={onSend}
      onBackendTurn={onBackendTurn}
      onStop={vi.fn()}
    />
  );
}

// What GET /sessions/{id} holds when the card is checked before a send.
function storedSession(rows: Record<string, unknown>[]) {
  return getGetV2GetSessionMockHandler({
    id: SESSION_ID,
    created_at: "2026-10-24T03:00:00Z",
    updated_at: "2026-10-24T03:00:00Z",
    user_id: "user-1",
    messages: rows,
  });
}

const TAGGED_APPROVAL = userRow(
  `Approved: hire Maya (confirmation_id: ${CONFIRMATION_ID}).`,
  {
    kind: "card_decision",
    decisions: [
      {
        verdict: "approved",
        confirmation_id: CONFIRMATION_ID,
        verb: "hire",
        name: "Maya",
      },
    ],
  },
);

function expectNoDecisionButtons() {
  expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Decline" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Send decisions" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Unapprove" })).toBeNull();
}

describe("a proposal card the chat already answered", () => {
  beforeEach(() => {
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
    server.use(
      getGetV2GetPendingReviewsForChatSessionMockHandler([]),
      storedSession([]),
    );
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows its decision and asks nothing after a reload", () => {
    render(<Chat messages={[ask, hireCard, TAGGED_APPROVAL]} />);

    expectNoDecisionButtons();
    expect(screen.getByText("Approved")).toBeDefined();
  });

  it("reads a decline saved before rows were tagged", () => {
    render(
      <Chat
        messages={[
          ask,
          hireCard,
          userRow(
            `Not approved: do not hire Maya, discard that proposal (confirmation_id: ${CONFIRMATION_ID}).`,
          ),
        ]}
      />,
    );

    expectNoDecisionButtons();
    expect(screen.getByText("Declined")).toBeDefined();
  });

  it("resolves as soon as the decision is sent", async () => {
    const user = userEvent.setup();
    render(<Thread initial={[ask, hireCard]} />);

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    expectNoDecisionButtons();
    expect(await screen.findByText("Approved")).toBeDefined();
  });

  it("stays answered while its decision waits behind a running turn", async () => {
    const user = userEvent.setup();
    // The user asked something else after the card; Otto is answering it.
    const followUp: Message = {
      id: `${SESSION_ID}-seq-3`,
      role: "user",
      parts: [{ type: "text", text: "What will Maya cost?" }],
    };
    const answering: Message = {
      id: "streaming-reply",
      role: "assistant",
      parts: [{ type: "text", text: "Maya's plan" }],
    };
    render(<QueuedThread initial={[ask, hireCard, followUp, answering]} />);

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    expect(await screen.findByText("Queued")).toBeDefined();
    await waitFor(() => expectNoDecisionButtons());
    expect(screen.getByText("Approved")).toBeDefined();
    expect(screen.queryByText("Sent")).toBeNull();
  });

  it("stops asking when another tab's answer arrives", () => {
    const { rerender } = render(<Chat messages={[ask, hireCard]} />);
    expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();

    rerender(<Chat messages={[ask, hireCard, TAGGED_APPROVAL]} />);

    expectNoDecisionButtons();
    expect(screen.getByText("Approved")).toBeDefined();
  });

  it("posts one decision however fast Send is pressed", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<Chat messages={[ask, hireCard]} onSend={onSend} />);

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.dblClick(screen.getByRole("button", { name: "Send decisions" }));

    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("in a stale tab, sends only the decisions another tab has not given", async () => {
    server.use(
      storedSession([
        {
          role: "user",
          content: `Approved: hire Maya (confirmation_id: ${CONFIRMATION_ID}).`,
        },
      ]),
    );
    const user = userEvent.setup();
    const onSend = vi.fn();
    const onBackendTurn = vi.fn();
    render(
      <Chat
        messages={[ask, teamCard]}
        onSend={onSend}
        onBackendTurn={onBackendTurn}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Decline" }));
    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    await waitFor(() =>
      expect(onSend).toHaveBeenCalledWith(
        "Approved: hire Theo (confirmation_id: c-theo).",
      ),
    );
    expect(onBackendTurn).not.toHaveBeenCalled();
  });

  it("in a stale tab, shows the answer another tab gave instead of sending a second", async () => {
    server.use(
      storedSession([
        { role: "user", content: "Get someone to run Dewleaf's marketing." },
        {
          role: "user",
          content: `Approved: hire Maya (confirmation_id: ${CONFIRMATION_ID}).`,
          metadata: {
            kind: "card_decision",
            decisions: [
              {
                verdict: "approved",
                confirmation_id: CONFIRMATION_ID,
                verb: "hire",
                name: "Maya",
              },
            ],
          },
        },
      ]),
    );
    const user = userEvent.setup();
    const onSend = vi.fn();
    const onBackendTurn = vi.fn();
    render(
      <Chat
        messages={[ask, hireCard]}
        onSend={onSend}
        onBackendTurn={onBackendTurn}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Decline" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    await waitFor(() => expect(onBackendTurn).toHaveBeenCalledTimes(1));
    expect(onSend).not.toHaveBeenCalled();
    // Nothing is in flight: the card waits for the refetch, not on "Sent".
    await waitFor(() => expect(screen.queryByText("Sent")).toBeNull());
    // Nor does it ask again meanwhile: it shows the answer the check found.
    expectNoDecisionButtons();
    expect(screen.getByText("Approved")).toBeDefined();
  });

  it("in a stale tab, finds another tab's answer older than the newest page", async () => {
    // The other tab answered, then the chat went on for more than a page:
    // the answer and the card are only on the page before the newest one.
    const answerAndCard = [
      {
        role: "tool",
        content: JSON.stringify({ confirmation_id: CONFIRMATION_ID }),
        sequence: 2,
      },
      {
        role: "user",
        content: `Approved: hire Maya (confirmation_id: ${CONFIRMATION_ID}).`,
        sequence: 3,
      },
    ];
    const laterRows = Array.from({ length: 50 }, (_, i) => ({
      role: i % 2 ? "assistant" : "user",
      content: `Later message ${i}`,
      sequence: 4 + i,
    }));
    const pages: (number | null)[] = [];
    server.use(
      getGetV2GetSessionMockHandler(({ request }) => {
        const before = new URL(request.url).searchParams.get("before_sequence");
        pages.push(before === null ? null : Number(before));
        const older = before !== null;
        return {
          id: SESSION_ID,
          created_at: "2026-10-24T03:00:00Z",
          updated_at: "2026-10-24T03:00:00Z",
          user_id: "user-1",
          messages: older ? answerAndCard : laterRows,
          has_more_messages: !older,
          oldest_sequence: older ? 2 : 4,
        };
      }),
    );
    const user = userEvent.setup();
    const onSend = vi.fn();
    const onBackendTurn = vi.fn();
    render(
      <Chat
        messages={[ask, hireCard]}
        onSend={onSend}
        onBackendTurn={onBackendTurn}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Decline" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    await waitFor(() => expect(onBackendTurn).toHaveBeenCalledTimes(1));
    expect(onSend).not.toHaveBeenCalled();
    expect(pages).toEqual([null, 4]);
  });

  it("drops a refused answer once the saved one is on screen", () => {
    render(
      <Chat
        messages={[
          ask,
          hireCard,
          TAGGED_APPROVAL,
          {
            ...userRow(
              `Not approved: do not hire Maya, discard that proposal (confirmation_id: ${CONFIRMATION_ID}).`,
            ),
            id: "optimistic-1",
          },
        ]}
      />,
    );

    expect(screen.getByText("Approved")).toBeDefined();
    expect(screen.getByText("You approved hiring Maya")).toBeDefined();
    expect(screen.queryByText("You declined hiring Maya")).toBeNull();
  });

  it("asks again when the decision could not be sent", async () => {
    const user = userEvent.setup();
    const failed = Promise.reject(new Error("refused"));
    void failed.catch(() => undefined);
    const onSend = vi.fn(() => failed);
    render(<Chat messages={[ask, hireCard]} onSend={onSend} />);

    await user.click(screen.getByRole("button", { name: "Approve" }));
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    expect(
      await screen.findByRole("button", { name: "Send decisions" }),
    ).toBeDefined();
    expect(screen.queryByText("Sent")).toBeNull();
  });
});
