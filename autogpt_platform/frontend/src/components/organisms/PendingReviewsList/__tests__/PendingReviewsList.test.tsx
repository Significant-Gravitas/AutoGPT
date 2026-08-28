import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { PendingReviewsList } from "../PendingReviewsList";

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "ne-1",
    node_id: "n-1",
    user_id: "u-1",
    graph_exec_id: "run-1",
    graph_id: "g-1",
    graph_version: 1,
    payload: { to: "x@y.com" },
    instructions: "SendEmailBlock",
    editable: true,
    status: "WAITING",
    created_at: new Date(),
    ...overrides,
  };
}

function captureReviewAction() {
  const captured: { body?: any } = {};
  server.use(
    getPostV2ProcessReviewActionMockHandler200(async (info) => {
      captured.body = await info.request.json();
      return { approved_count: 1, rejected_count: 0, failed_count: 0 };
    }),
  );
  return captured;
}

test("a decision covers only the group it was made in", async () => {
  const captured = captureReviewAction();

  render(
    <PendingReviewsList
      reviews={[
        makeReview({ node_exec_id: "ne-1", node_id: "n-1" }),
        makeReview({ node_exec_id: "ne-2", node_id: "n-1" }),
        makeReview({
          node_exec_id: "ne-3",
          node_id: "n-2",
          instructions: "GithubMergePullRequestBlock",
        }),
      ]}
    />,
  );

  await userEvent.click(
    screen.getByRole("button", { name: "Approve 2 reviews" }),
  );

  await waitFor(() => expect(captured.body).toBeDefined());
  const ids = captured.body.reviews.map((r: any) => r.node_exec_id);
  expect(ids.sort()).toEqual(["ne-1", "ne-2"]);
  expect(ids).not.toContain("ne-3");
});

test("every queued review is visible without expanding anything", () => {
  render(
    <PendingReviewsList
      reviews={[
        makeReview({ node_exec_id: "ne-1", payload: { to: "first@y.com" } }),
        makeReview({ node_exec_id: "ne-2", payload: { to: "second@y.com" } }),
      ]}
    />,
  );

  const values = screen
    .getAllByRole("textbox")
    .map((node) => (node as HTMLTextAreaElement).value);
  expect(values.join("\n")).toContain("first@y.com");
  expect(values.join("\n")).toContain("second@y.com");
});

test("rejecting a single-review group submits only that review as rejected", async () => {
  const captured = captureReviewAction();

  render(
    <PendingReviewsList
      reviews={[makeReview({ node_exec_id: "ne-1", node_id: "n-1" })]}
    />,
  );

  await userEvent.click(screen.getByRole("button", { name: "Reject" }));

  await waitFor(() => expect(captured.body).toBeDefined());
  expect(captured.body.reviews).toEqual([
    expect.objectContaining({ node_exec_id: "ne-1", approved: false }),
  ]);
});

test("a collapsed group offers no way to decide it", async () => {
  render(
    <PendingReviewsList
      reviews={[makeReview({ node_exec_id: "ne-1", node_id: "n-1" })]}
    />,
  );

  expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();

  await userEvent.click(
    screen.getByRole("button", { name: /Review required for/ }),
  );

  expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Reject" })).toBeNull();
});
