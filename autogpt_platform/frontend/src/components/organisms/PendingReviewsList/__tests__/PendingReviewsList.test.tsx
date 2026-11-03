import { fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ReviewRequest } from "@/app/api/__generated__/models/reviewRequest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { PendingReviewsList } from "../PendingReviewsList";

const { toastSpy } = vi.hoisted(() => ({ toastSpy: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: toastSpy,
  useToast: () => ({ toast: toastSpy, dismiss: vi.fn() }),
}));

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

// The JSON editor keeps a half-typed draft on screen. Approving must not send
// the last valid JSON in its place: with a trailing comma left behind,
// "b@x.com" would still go out while the reviewer sees it deleted.
test.each([
  ["a draft that doesn't parse", '[\n  "a@x.com",\n]'],
  ["an emptied draft", ""],
])(
  "approving %s is blocked instead of sending stale data",
  async (_, draft) => {
    const captured = captureReviewAction();
    render(
      <PendingReviewsList
        reviews={[
          makeReview({
            payload: [
              "a@x.com",
              "b@x.com",
            ] as unknown as PendingHumanReviewModel["payload"],
          }),
        ]}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText("Edit JSON data"), {
      target: { value: draft },
    });
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Invalid JSON" }),
      ),
    );
    expect(captured.body).toBeUndefined();
  },
);

test("rejecting goes through while the JSON draft doesn't parse", async () => {
  const captured = captureReviewAction();
  const payload = ["a@x.com", "b@x.com"];
  render(
    <PendingReviewsList
      reviews={[
        makeReview({
          payload: payload as unknown as PendingHumanReviewModel["payload"],
        }),
      ]}
    />,
  );

  fireEvent.change(screen.getByPlaceholderText("Edit JSON data"), {
    target: { value: '[\n  "a@x.com",\n]' },
  });
  await userEvent.click(screen.getByRole("button", { name: "Reject" }));

  await waitFor(() => expect(captured.body).toBeDefined());
  expect(captured.body?.reviews[0]).toMatchObject({
    approved: false,
    reviewed_data: payload,
  });
});

function spendReview(overrides: Partial<PendingHumanReviewModel> = {}) {
  return makeReview({
    node_exec_id: "expert-spend:e-1:run-1",
    node_id: "expert-spend:e-1",
    payload: {
      expert: "Maria",
      spent_credits: 300,
      threshold_credits: 250,
      window: "week",
    },
    instructions:
      "Maria has used 300 of 250 credits this week and needs your approval to keep spending",
    action: undefined,
    agent_name: undefined,
    editable: false,
    ...overrides,
  });
}

test("an expert's spend approval is one compact card, beside the other reviews", () => {
  render(<PendingReviewsList reviews={[spendReview(), makeReview()]} />);

  expect(
    screen.getByText("Maria has done 300 credits of work this week"),
  ).toBeDefined();
  expect(
    screen.getByText("That's past your 250 check-in. Keep going?"),
  ).toBeDefined();
  expect(screen.getByRole("img", { name: "300 of 250 credits" })).toBeDefined();
  expect(screen.queryByText(/needs your approval to keep spending/)).toBeNull();
  // The block review keeps its own card.
  expect(screen.getByText("Send Email")).toBeDefined();
});

test.each([
  ["Keep going", true],
  ["Cancel", false],
])(
  "%s on a spend card decides only that expert's runs",
  async (button, approved) => {
    const captured = captureReviewAction();
    render(<PendingReviewsList reviews={[spendReview(), makeReview()]} />);

    await userEvent.click(screen.getByRole("button", { name: button }));

    await waitFor(() => expect(captured.body).toBeDefined());
    expect(captured.body?.reviews).toEqual([
      expect.objectContaining({
        node_exec_id: "expert-spend:e-1:run-1",
        approved,
      }),
    ]);
    expect(captured.body?.reviews[0].auto_approve_future).not.toBe(true);
  },
);

test("an unreadable spend payload keeps the standard approval controls", () => {
  render(
    <PendingReviewsList
      reviews={[
        spendReview({
          payload: {
            expert: "Maria",
            spent_credits: 300,
            threshold_credits: "unknown",
            window: "week",
          },
        }),
      ]}
    />,
  );

  expect(screen.getByText("Your review is needed")).toBeDefined();
  expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();
  expect(screen.getByRole("button", { name: "Reject" })).toBeDefined();
  expect(screen.queryByRole("button", { name: "Keep going" })).toBeNull();
});
