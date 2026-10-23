import { useState } from "react";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getGetV2GetPendingReviewsForChatSessionMockHandler } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
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
