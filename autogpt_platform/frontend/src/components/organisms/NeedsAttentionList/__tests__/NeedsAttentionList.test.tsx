import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test } from "vitest";
import {
  getGetV2GetPendingReviewsMockHandler200,
  getPostV2ProcessReviewActionMockHandler200,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { NeedsAttentionList } from "../NeedsAttentionList";

const review: PendingHumanReviewModel = {
  node_exec_id: "ne-1",
  node_id: "n-1",
  user_id: "u-1",
  graph_exec_id: "run-1",
  graph_id: "g-1",
  graph_version: 1,
  payload: { to: "x@y.com" },
  instructions: "Approve outreach email",
  editable: true,
  status: "WAITING",
  expert_id: "exp-1",
  expert_name: "Ana",
  expert_avatar_url: null,
  agent_name: "Lead Finder",
  library_agent_id: "lib-1",
  session_id: null,
  created_at: new Date(),
};

test("renders attributed rows and approves in one tap", async () => {
  let actionBody: unknown;
  server.use(
    getGetV2GetPendingReviewsMockHandler200([review]),
    getPostV2ProcessReviewActionMockHandler200(async (info) => {
      actionBody = await info.request.json();
      return { approved_count: 1, rejected_count: 0, failed_count: 0 };
    }),
  );

  render(<NeedsAttentionList />);
  expect(await screen.findByText("Approve outreach email")).toBeDefined();
  expect(screen.getByText(/Ana/)).toBeDefined();

  await userEvent.click(screen.getByRole("button", { name: /approve/i }));
  await waitFor(() =>
    expect(actionBody).toMatchObject({
      reviews: [{ node_exec_id: "ne-1", approved: true }],
    }),
  );
});

test("skip sends a rejection", async () => {
  let actionBody:
    | { reviews: Array<{ approved: boolean; message?: string }> }
    | undefined;
  server.use(
    getGetV2GetPendingReviewsMockHandler200([review]),
    getPostV2ProcessReviewActionMockHandler200(async (info) => {
      actionBody = (await info.request.json()) as typeof actionBody;
      return { approved_count: 0, rejected_count: 1, failed_count: 0 };
    }),
  );

  render(<NeedsAttentionList />);
  await userEvent.click(await screen.findByRole("button", { name: /skip/i }));
  await waitFor(() =>
    expect(actionBody?.reviews[0]).toMatchObject({
      approved: false,
      message: "Skipped from home",
    }),
  );
});

test("shows a destructive toast when approve fails", async () => {
  server.use(
    getGetV2GetPendingReviewsMockHandler200([review]),
    http.post(
      "/api/proxy/api/review/action",
      () => new HttpResponse(null, { status: 500 }),
    ),
  );

  render(
    <>
      <NeedsAttentionList />
      <Toaster />
    </>,
  );
  await userEvent.click(
    await screen.findByRole("button", { name: /approve/i }),
  );

  expect(await screen.findByText("Failed to approve review")).toBeDefined();
});

test("renders nothing when there are no pending reviews", async () => {
  server.use(getGetV2GetPendingReviewsMockHandler200([]));

  const { container } = render(<NeedsAttentionList />);
  await waitFor(() => expect(container.firstChild).toBeNull());
});
