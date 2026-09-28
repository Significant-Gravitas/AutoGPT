import {
  getAnswerSessionMockHandler200,
  getAnswerSessionMockHandler503,
  getGetV2GetSessionMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { useDelegationAnswerStore } from "../../../delegationAnswerStore";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ToolChain } from "../ToolChain";

const { push } = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const ALEX = { id: "exp-alex", name: "Alex", role: "Product Manager" };

function delegatePart(output: unknown, state = "output-available") {
  return {
    type: "tool-delegate_to_expert",
    state,
    toolCallId: "call-1",
    input: { expert_id: "exp-alex", prompt: "Onboarding revamp PRD" },
    output,
  } as unknown as MessagePart;
}

function waitingSubSession() {
  return getGetV2GetSessionMockHandler200({
    id: "sub-1",
    created_at: "2026-09-28T10:00:00Z",
    updated_at: "2026-09-28T10:00:00Z",
    user_id: "u-1",
    chat_status: "idle",
    active_stream: null,
    metadata: {},
    messages: [
      { role: "user", content: "Draft the PRD" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "ask-1",
            function: {
              name: "ask_question",
              arguments: JSON.stringify({
                questions: [
                  {
                    question: "Q4 release train or December mini-launch?",
                    options: ["Q4", "December", "Both"],
                  },
                ],
              }),
            },
          },
        ],
      },
    ],
  });
}

function chain(parts: MessagePart[]) {
  return (
    <CopilotChatActionsProvider onSend={vi.fn()}>
      <ToolChain parts={parts} isStreaming={false} />
    </CopilotChatActionsProvider>
  );
}

describe("a hand-off on the wire", () => {
  afterEach(() => {
    cleanup();
    push.mockReset();
    useDelegationAnswerStore.setState({ answers: {} });
  });

  // Coverage-instrumented CI shards mount the settled chain slowly.
  const SLOW = 20_000;

  async function answerTheQuestion() {
    const node = await screen.findByTestId(
      "handoff-question-node",
      {},
      { timeout: 8000 },
    );
    expect(node.textContent).toContain(
      "Alex asks · paused on “Onboarding revamp PRD”",
    );
    expect(screen.getByText("Alex asked a question")).toBeDefined();
    // One tag on the row, one on the card's header.
    expect(screen.getAllByText("Needs you")).toHaveLength(2);

    fireEvent.click(screen.getByRole("button", { name: "Q4" }));
    fireEvent.change(screen.getByLabelText("Answer Alex"), {
      target: { value: "Keep December as a stretch note" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
  }

  function renderWaitingHandoff() {
    render(
      chain([
        delegatePart({
          status: "completed",
          sub_session_id: "sub-1",
          response: "Q4 release train or December mini-launch?",
          expert: ALEX,
        }),
      ]),
    );
  }

  it(
    "hangs the teammate's question on the wire, chips and all, and posts the answer into their thread",
    async () => {
      const posted: { sessionId: string; body: unknown }[] = [];
      server.use(
        waitingSubSession(),
        getAnswerSessionMockHandler200(async ({ request, params }) => {
          posted.push({
            sessionId: String(params.sessionId),
            body: await request.json(),
          });
          return { session_id: "sub-1", queued: false };
        }),
      );
      renderWaitingHandoff();
      await answerTheQuestion();

      await waitFor(() => expect(posted).toHaveLength(1));
      expect(posted[0]).toEqual({
        sessionId: "sub-1",
        body: { message: "Q4. Keep December as a stretch note" },
      });
      expect(push).not.toHaveBeenCalled();
    },
    SLOW,
  );

  it(
    "drafts the answer into the teammate's thread when posting it fails",
    async () => {
      server.use(waitingSubSession(), getAnswerSessionMockHandler503());
      renderWaitingHandoff();
      await answerTheQuestion();

      await waitFor(() => expect(push).toHaveBeenCalled());
      const href = new URL(push.mock.calls[0][0], "http://x");
      expect(href.searchParams.get("sessionId")).toBe("sub-1");
      expect(href.searchParams.get("prefill")).toBe(
        "Q4. Keep December as a stretch note",
      );
    },
    SLOW,
  );

  it(
    "sends a typed answer on Enter, and a chip picked twice is dropped",
    async () => {
      const posted: unknown[] = [];
      server.use(
        waitingSubSession(),
        getAnswerSessionMockHandler200(async ({ request }) => {
          posted.push(await request.json());
          return { session_id: "sub-1", queued: false };
        }),
      );
      render(
        chain([
          delegatePart({
            status: "needs_input",
            sub_session_id: "sub-1",
            question: "Q4 release train or December mini-launch?",
            question_options: ["Q4", "December"],
            expert: ALEX,
          }),
        ]),
      );
      await screen.findByTestId("handoff-question-node", {}, { timeout: 8000 });
      const chip = screen.getByRole("button", { name: "Q4" });
      fireEvent.click(chip);
      fireEvent.click(chip);
      expect(chip.getAttribute("aria-pressed")).toBe("false");
      const input = screen.getByLabelText("Answer Alex");
      fireEvent.keyDown(input, { key: "Enter" });
      fireEvent.change(input, { target: { value: "December" } });
      fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
      expect(posted).toHaveLength(0);
      fireEvent.keyDown(input, { key: "Enter" });
      await waitFor(() => expect(posted).toEqual([{ message: "December" }]));
    },
    SLOW,
  );

  it("says the user stopped the hand-off once a stopped turn reloads", () => {
    render(chain([delegatePart("")]));
    fireEvent.click(screen.getByRole("button", { expanded: false }));
    expect(
      screen.getByText("You stopped the hand-off to exp-alex"),
    ).toBeDefined();
  });
});
