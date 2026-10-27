import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";
import {
  getGetV2GetPendingReviewsForChatSessionMockHandler,
  getGetV2GetPendingReviewsForExecutionMockHandler,
  getPostV2ProcessReviewActionMockHandler200,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { ReviewRequest } from "@/app/api/__generated__/models/reviewRequest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { CopilotPendingReviews } from "../CopilotPendingReviews";

const onSend = vi.fn();
vi.mock("../../CopilotChatActionsProvider/useCopilotChatActions", () => ({
  useCopilotChatActions: () => ({ onSend }),
}));

afterEach(() => {
  cleanup();
  onSend.mockReset();
});

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "copilot-node-blk:ab12",
    node_id: "copilot-node-blk",
    user_id: "u-1",
    session_id: "chat-1",
    graph_exec_id: null,
    graph_id: null,
    graph_version: null,
    payload: { path: "/reports" },
    instructions: "Create Folder",
    editable: true,
    status: "WAITING",
    created_at: new Date(),
    ...overrides,
  };
}

test("a chat's queue is read from the chat, and the resume is AutoPilot's", async () => {
  let waiting = [makeReview()];
  server.use(
    getGetV2GetPendingReviewsForChatSessionMockHandler(() => waiting),
    getGetV2GetPendingReviewsForExecutionMockHandler([
      makeReview({ instructions: "Not this chat's review" }),
    ]),
    getPostV2ProcessReviewActionMockHandler200(() => {
      waiting = [];
      return { approved_count: 1, rejected_count: 0, failed_count: 0 };
    }),
  );

  render(<CopilotPendingReviews chatSessionId="chat-1" />);

  expect((await screen.findAllByText("Create Folder")).length).toBeGreaterThan(
    0,
  );
  expect(screen.queryByText("Not this chat's review")).toBeNull();

  await userEvent.click(screen.getAllByRole("button", { name: /^Approve/ })[0]);

  await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1), {
    timeout: 3000,
  });
  expect(onSend.mock.calls[0][0]).toContain("resume_capability");
});

test("an agent run's queue is read from its execution", async () => {
  server.use(
    getGetV2GetPendingReviewsForExecutionMockHandler([
      makeReview({
        session_id: null,
        graph_exec_id: "exec-9",
        graph_id: "g-1",
        graph_version: 1,
        instructions: "Send Email",
      }),
    ]),
    getGetV2GetPendingReviewsForChatSessionMockHandler([
      makeReview({ instructions: "Not the run's review" }),
    ]),
  );

  render(<CopilotPendingReviews runs={[{ graphExecId: "exec-9" }]} />);

  expect((await screen.findAllByText("Send Email")).length).toBeGreaterThan(0);
  expect(screen.queryByText("Not the run's review")).toBeNull();
});

test("every run waiting on one expert's spend approval is approved with one click", async () => {
  const runs = ["exec-1", "exec-2", "exec-3", "exec-4"];
  const waiting = new Set(runs);
  const posted: string[][] = [];
  server.use(
    http.get("*/api/review/execution/:id", ({ params }) => {
      const id = String(params.id);
      return HttpResponse.json(
        waiting.has(id)
          ? [
              makeReview({
                node_exec_id: `expert-spend:e-1:${id}`,
                node_id: "expert-spend:e-1",
                session_id: null,
                graph_exec_id: id,
                graph_id: "g-1",
                graph_version: 1,
                payload: { expert: "Maria", spent_credits: 300 },
                instructions: "Maria needs your approval to keep spending",
                editable: false,
              }),
            ]
          : [],
      );
    }),
    // Like the real endpoint, one run's reviews per request.
    http.post("*/api/review/action", async ({ request }) => {
      const body = (await request.json()) as ReviewRequest;
      const ids = body.reviews.map((r) => r.node_exec_id);
      posted.push(ids);
      const ofRuns = new Set(ids.map((id) => id.split(":").at(-1) ?? ""));
      if (ofRuns.size > 1) {
        return HttpResponse.json(
          {
            detail:
              "All reviews in a single request must belong to the same execution.",
          },
          { status: 400 },
        );
      }
      for (const run of ofRuns) waiting.delete(run);
      return HttpResponse.json({
        approved_count: ids.length,
        rejected_count: 0,
        failed_count: 0,
      });
    }),
  );

  render(
    <CopilotPendingReviews
      runs={runs.map((graphExecId) => ({ graphExecId }))}
    />,
  );

  expect(await screen.findByText("4 reviews")).toBeDefined();
  expect(screen.queryByText(/Auto-approve future executions/)).toBeNull();

  await userEvent.click(
    screen.getByRole("button", { name: "Approve 4 reviews" }),
  );

  await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1), {
    timeout: 5000,
  });
  expect(posted.map((ids) => ids.length)).toEqual([1, 1, 1, 1]);
  for (const id of runs) {
    expect(onSend.mock.calls[0][0]).toContain(`execution_id="${id}"`);
  }
});
