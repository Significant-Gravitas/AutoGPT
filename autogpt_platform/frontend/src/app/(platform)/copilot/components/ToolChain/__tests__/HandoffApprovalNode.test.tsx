import {
  getGetV2GetPendingReviewsForChatSessionMockHandler200,
  getPostV2ProcessReviewActionMockHandler200,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { heldReview } from "../../ApprovalQueue/__tests__/fixtures";
import { ChatSessionContext } from "../../ChatContainer/components/ChatSessionContext";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import type { HeldOutcome } from "../../ChatMessagesContainer/heldCallRows";
import { HeldOutcomesContext } from "../../ChatMessagesContainer/HeldOutcomesContext";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ToolChain } from "../ToolChain";

const REVIEW_ID = "copilot-node-gate-delegate_to_expert:h1";

const HELD_HANDOFF: MessagePart = {
  type: "tool-delegate_to_expert",
  state: "output-available",
  toolCallId: "call-h1",
  input: { expert_id: "Alex", prompt: "Draft the PRD" },
  output: {
    type: "approval_required",
    tool_name: "delegate_to_expert",
    reason: "Ask First is on.",
    review_id: REVIEW_ID,
    ask: "Hand a task to a teammate",
    object: "Alex",
  },
} as MessagePart;

function chain(outcomes: Map<string, HeldOutcome>, onBackendTurn = vi.fn()) {
  return (
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={onBackendTurn}>
      <ChatSessionContext.Provider value="s1">
        <HeldOutcomesContext.Provider value={outcomes}>
          <ToolChain parts={[HELD_HANDOFF]} isStreaming={false} />
        </HeldOutcomesContext.Provider>
      </ChatSessionContext.Provider>
    </CopilotChatActionsProvider>
  );
}

describe("a hand-off held for approval", () => {
  afterEach(cleanup);

  // Coverage-instrumented CI shards mount the settled chain slowly.
  const SLOW = 20_000;

  it(
    "renders its approval card on the wire, without a row icon",
    async () => {
      server.use(
        getGetV2GetPendingReviewsForChatSessionMockHandler200([
          heldReview({
            id: "h1",
            tool: "delegate_to_expert",
            args: { expert_id: "Alex", prompt: "Draft the PRD" },
            headline: { ask: "Hand a task to a teammate", object: "Alex" },
          }),
        ]),
      );
      render(chain(new Map()));

      expect(
        await screen.findByRole(
          "button",
          { name: /^approve$/i },
          { timeout: 8000 },
        ),
      ).toBeDefined();
      const node = screen.getByTestId("handoff-approval-node");
      expect(node.textContent).toContain("Hand off “Draft the PRD” to Alex");
      expect(node.textContent).toContain("Ask First");
      expect(node.textContent).toContain("Brief");
      expect(node.textContent).toContain("A report in this chat");
      expect(screen.getByRole("button", { name: /reject/i })).toBeDefined();
      expect(screen.queryByText("Waiting for you")).toBeNull();
      // The review takes no edits, so the card offers none.
      expect(screen.queryByRole("button", { name: /edit brief/i })).toBeNull();
      expect(screen.queryByText(/always allow/i)).toBeNull();
    },
    SLOW,
  );

  it(
    "answers from the card and follows the turn the server starts",
    async () => {
      const onBackendTurn = vi.fn();
      server.use(
        getGetV2GetPendingReviewsForChatSessionMockHandler200([
          heldReview({
            id: "h1",
            tool: "delegate_to_expert",
            args: { expert_id: "Alex", prompt: "Draft the PRD" },
          }),
        ]),
        getPostV2ProcessReviewActionMockHandler200({
          approved_count: 1,
          rejected_count: 0,
          failed_count: 0,
          error: null,
        }),
      );
      render(chain(new Map(), onBackendTurn));

      fireEvent.click(
        await screen.findByRole(
          "button",
          { name: /^approve$/i },
          { timeout: 8000 },
        ),
      );
      await waitFor(() => expect(onBackendTurn).toHaveBeenCalled(), {
        timeout: 8000,
      });
    },
    SLOW,
  );

  it(
    "always allows hand-offs to the expert from the card's link",
    async () => {
      const bodies: unknown[] = [];
      server.use(
        getGetV2GetPendingReviewsForChatSessionMockHandler200([
          heldReview({
            id: "h1",
            tool: "delegate_to_expert",
            args: { expert_id: "Alex", prompt: "Draft the PRD" },
            chatRules: ["allow"],
          }),
        ]),
        http.post("*/api/review/action", async ({ request }) => {
          bodies.push(await request.json());
          return HttpResponse.json({
            approved_count: 1,
            rejected_count: 0,
            failed_count: 0,
            error: null,
          });
        }),
      );
      render(chain(new Map()));

      fireEvent.click(
        await screen.findByRole(
          "button",
          { name: /always allow hand-offs to alex/i },
          { timeout: 8000 },
        ),
      );
      await waitFor(() => expect(bodies).toHaveLength(1), { timeout: 8000 });
      expect(bodies[0]).toMatchObject({
        reviews: [
          {
            node_exec_id: REVIEW_ID,
            approved: true,
            chat_rule: "allow",
            chat_rule_scope: "expert",
          },
        ],
      });
    },
    SLOW,
  );

  it(
    "sends an edited brief when the review accepts edits",
    async () => {
      const bodies: unknown[] = [];
      server.use(
        getGetV2GetPendingReviewsForChatSessionMockHandler200([
          {
            ...heldReview({
              id: "h1",
              tool: "delegate_to_expert",
              args: { expert_id: "Alex", prompt: "Draft the PRD" },
            }),
            editable: true,
          },
        ]),
        http.post("*/api/review/action", async ({ request }) => {
          bodies.push(await request.json());
          return HttpResponse.json({
            approved_count: 1,
            rejected_count: 0,
            failed_count: 0,
            error: null,
          });
        }),
      );
      render(chain(new Map()));

      fireEvent.click(
        await screen.findByRole(
          "button",
          { name: /edit brief/i },
          { timeout: 8000 },
        ),
      );
      fireEvent.change(screen.getByLabelText("Brief"), {
        target: { value: "Draft the PRD, scope only" },
      });
      fireEvent.click(screen.getByRole("button", { name: /^approve$/i }));
      await waitFor(() => expect(bodies).toHaveLength(1), { timeout: 8000 });
      expect(bodies[0]).toMatchObject({
        reviews: [
          {
            approved: true,
            reviewed_data: {
              expert_id: "Alex",
              prompt: "Draft the PRD, scope only",
            },
          },
        ],
      });
    },
    SLOW,
  );

  it(
    "closes into a 'You approved' row once the hand-off ran",
    async () => {
      render(
        chain(
          new Map([
            [
              "call-h1",
              {
                outcome: "approved",
                output: {
                  status: "running",
                  sub_session_id: "sub-1",
                  expert: { id: "exp-alex", name: "Alex", role: "PM" },
                },
              },
            ],
          ]),
        ),
      );
      // Coverage-instrumented CI shards render the settled chain slowly.
      expect(
        await screen.findByText(
          "You approved the hand-off to Alex",
          {},
          { timeout: 8000 },
        ),
      ).toBeDefined();
      expect(screen.queryByTestId("handoff-approval-node")).toBeNull();
    },
    SLOW,
  );
});
