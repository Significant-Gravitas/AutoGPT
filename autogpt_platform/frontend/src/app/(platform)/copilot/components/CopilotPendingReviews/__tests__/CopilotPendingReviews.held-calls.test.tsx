import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ReviewRequest } from "@/app/api/__generated__/models/reviewRequest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../CopilotPendingReviews";

const CHAT = "s1";
const NODE = "copilot-node-gate-post_to_chat_platform";

function heldCall(id: string, text: string): PendingHumanReviewModel {
  return {
    node_exec_id: `${NODE}:${id}`,
    node_id: NODE,
    user_id: "u-1",
    session_id: CHAT,
    graph_exec_id: null,
    graph_id: null,
    graph_version: null,
    payload: { tool: "post_to_chat_platform", arguments: { text } },
    instructions: `Post to chat platform — ${text}`,
    editable: false,
    status: "WAITING",
    created_at: new Date(),
  };
}

function serveQueue(reviews: PendingHumanReviewModel[]) {
  const sent: ReviewRequest[] = [];
  server.use(
    http.get(`*/api/review/session/${CHAT}`, () => HttpResponse.json(reviews)),
    getPostV2ProcessReviewActionMockHandler200(async (info) => {
      sent.push((await info.request.json()) as ReviewRequest);
      return { approved_count: 1, rejected_count: 0, failed_count: 0 };
    }),
  );
  return sent;
}

test("each held call is its own card, oldest first, and answering one follows the server's turn", async () => {
  const sent = serveQueue([
    heldCall("a", "first post"),
    heldCall("b", "second post"),
  ]);
  const onSend = vi.fn();
  const onBackendTurn = vi.fn();

  render(
    <CopilotChatActionsProvider onSend={onSend} onBackendTurn={onBackendTurn}>
      <CopilotPendingReviews chatSessionId={CHAT} />
    </CopilotChatActionsProvider>,
  );

  const approves = await screen.findAllByRole("button", { name: "Approve" });
  expect(approves).toHaveLength(2);
  const [first] = screen.getAllByText(/first post/);
  const [second] = screen.getAllByText(/second post/);
  expect(
    first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();

  await userEvent.click(approves[0]);

  await waitFor(() => expect(onBackendTurn).toHaveBeenCalledTimes(1));
  expect(sent).toHaveLength(1);
  expect(sent[0].reviews.map((r) => r.node_exec_id)).toEqual([`${NODE}:a`]);
  expect(onSend).not.toHaveBeenCalled();
});
