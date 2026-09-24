import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import {
  getGetV2GetPendingReviewsForChatSessionMockHandler,
  getGetV2GetPendingReviewsForExecutionMockHandler,
  getPostV2ProcessReviewActionMockHandler200,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
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

  expect(await screen.findByText("Create Folder")).toBeDefined();
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

  render(<CopilotPendingReviews graphExecId="exec-9" />);

  expect(await screen.findByText("Send Email")).toBeDefined();
  expect(screen.queryByText("Not the run's review")).toBeNull();
});
