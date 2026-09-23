import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ReviewRequest } from "@/app/api/__generated__/models/reviewRequest";
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
    action: "Send Email",
    agent_name: "Invoice follow-up",
    editable: true,
    status: "WAITING",
    created_at: new Date(),
    ...overrides,
  };
}

function captureReviewAction() {
  const captured: { body?: ReviewRequest } = {};
  server.use(
    getPostV2ProcessReviewActionMockHandler200(async (info) => {
      captured.body = (await info.request.json()) as ReviewRequest;
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
  const ids = captured.body?.reviews.map((r) => r.node_exec_id) ?? [];
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
  expect(captured.body?.reviews).toEqual([
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

  await userEvent.click(screen.getByRole("button", { name: /Send Email/ }));

  expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Reject" })).toBeNull();
});

test("action-gate approvals offer no auto-approve toggle", () => {
  render(
    <PendingReviewsList
      reviews={[
        makeReview({
          node_exec_id: "copilot-node-gate-bash_exec:abc",
          node_id: "copilot-node-gate-bash_exec",
          editable: false,
        }),
        makeReview({ node_exec_id: "ne-2", node_id: "n-2" }),
      ]}
    />,
  );

  expect(
    screen.getAllByText("Auto-approve future executions of this node"),
  ).toHaveLength(1);
});

test("auto-approve on an ordinary node is sent with the approval", async () => {
  const captured = captureReviewAction();

  render(
    <PendingReviewsList
      reviews={[makeReview({ node_exec_id: "ne-1", node_id: "n-1" })]}
    />,
  );

  await userEvent.click(screen.getByRole("switch"));
  await userEvent.click(screen.getByRole("button", { name: "Approve" }));

  await waitFor(() => expect(captured.body).toBeDefined());
  expect(captured.body?.reviews).toEqual([
    expect.objectContaining({
      node_exec_id: "ne-1",
      approved: true,
      auto_approve_future: true,
    }),
  ]);
});

test("a block's group is titled by its action and names its workflow", () => {
  render(<PendingReviewsList reviews={[makeReview()]} />);

  expect(screen.getByText("Send Email")).toBeDefined();
  expect(screen.getByText("In workflow “Invoice follow-up”")).toBeDefined();
  expect(screen.queryByText(/SendEmailBlock/)).toBeNull();
});

function makeGateReview(subject?: Record<string, string>) {
  return makeReview({
    node_exec_id: "copilot-node-gate-run_capability:abc",
    node_id: "copilot-node-gate-run_capability",
    editable: false,
    payload: {
      tool: "run_capability",
      arguments: {},
      ...(subject ? { subject } : {}),
    },
  });
}

test("a card naming a subject can allow it for the rest of the chat", async () => {
  const captured = captureReviewAction();

  render(
    <PendingReviewsList
      reviews={[
        makeGateReview({
          name: "do_thing on mcp.example.com",
          effect: "external",
        }),
      ]}
    />,
  );

  await userEvent.click(
    screen.getByRole("button", { name: "Allow for this chat" }),
  );

  await waitFor(() => expect(captured.body).toBeDefined());
  expect(captured.body?.reviews).toEqual([
    expect.objectContaining({ approved: true, chat_rule: "allow" }),
  ]);
});

test("a card for a bare tool offers no chat rule", () => {
  render(<PendingReviewsList reviews={[makeGateReview()]} />);

  expect(
    screen.queryByRole("button", { name: "Allow for this chat" }),
  ).toBeNull();
  expect(
    screen.queryByRole("button", { name: "Judge for this chat" }),
  ).toBeNull();
});

test("an AutoPilot action's card neither calls it a workflow nor offers an edit", () => {
  render(
    <PendingReviewsList
      reviews={[
        makeReview({
          node_exec_id: "copilot-node-gate-bash_exec:abc",
          node_id: "copilot-node-gate-bash_exec",
          action: undefined,
          agent_name: undefined,
          instructions: "Bash exec — lists files",
          editable: false,
        }),
      ]}
    />,
  );

  expect(
    screen.getByText(
      "Otto is waiting for your approval before the action below.",
    ),
  ).toBeDefined();
  expect(screen.queryByText(/edit it if needed/)).toBeNull();
  expect(screen.queryByText(/Node #/)).toBeNull();
});
