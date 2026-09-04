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

  await userEvent.click(screen.getByRole("button", { name: /^Approve:/ }));
  await waitFor(() =>
    expect(actionBody).toMatchObject({
      reviews: [{ node_exec_id: "ne-1", approved: true }],
    }),
  );
});

test("only the acted row locks while its decision is in flight", async () => {
  const other = {
    ...review,
    node_exec_id: "ne-2",
    instructions: "Approve invoice",
  };
  server.use(
    getGetV2GetPendingReviewsMockHandler200([review, other]),
    http.post("/api/proxy/api/review/action", async () => {
      await new Promise((resolve) => setTimeout(resolve, 200));
      return HttpResponse.json({
        approved_count: 1,
        rejected_count: 0,
        failed_count: 0,
      });
    }),
  );

  render(<NeedsAttentionList />);
  const buttons = await screen.findAllByRole("button", { name: /^Approve:/ });
  await userEvent.click(buttons[0]);

  await waitFor(() => {
    const [first, second] = screen.getAllByRole("button", {
      name: /^Approve:/,
    }) as HTMLButtonElement[];
    expect(first.disabled).toBe(true);
    expect(second.disabled).toBe(false);
  });
});

test("confirms a successful decision with a toast", async () => {
  server.use(
    getGetV2GetPendingReviewsMockHandler200([review]),
    getPostV2ProcessReviewActionMockHandler200({
      approved_count: 1,
      rejected_count: 0,
      failed_count: 0,
    }),
  );

  render(
    <>
      <NeedsAttentionList />
      <Toaster />
    </>,
  );
  await userEvent.click(
    await screen.findByRole("button", { name: /^Approve:/ }),
  );

  expect(await screen.findByText("Approved")).toBeDefined();
});

test("decline sends a rejection", async () => {
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
  // Decline is a hard reject with no undo, so it takes a second, deliberate
  // tap in a flow built for fast tapping.
  await userEvent.click(
    await screen.findByRole("button", { name: /^Decline:/ }),
  );
  expect(actionBody).toBeUndefined();
  await userEvent.click(
    await screen.findByRole("button", { name: /^Confirm decline:/ }),
  );
  await waitFor(() =>
    expect(actionBody?.reviews[0]).toMatchObject({ approved: false }),
  );
  // No canned reason: this surface has no field to write one in, so nothing
  // should reach the agent context / audit trail as if the user typed it.
  expect(actionBody?.reviews[0].message).toBeUndefined();
});

test("armed decline is announced and visually distinct, not just relabelled", async () => {
  server.use(getGetV2GetPendingReviewsMockHandler200([review]));

  render(<NeedsAttentionList />);
  const declineButton = await screen.findByRole("button", {
    name: /^Decline:/,
  });
  const initialClassName = declineButton.className;

  await userEvent.click(declineButton);

  const armed = await screen.findByRole("button", {
    name: /^Confirm decline:/,
  });
  expect(armed.className).not.toBe(initialClassName);
  expect(
    screen.getByText(`Tap again to decline: ${review.instructions}`),
  ).toBeDefined();
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
    await screen.findByRole("button", { name: /^Approve:/ }),
  );

  expect(await screen.findByText("Failed to approve review")).toBeDefined();
});

test("renders nothing when there are no pending reviews", async () => {
  server.use(getGetV2GetPendingReviewsMockHandler200([]));

  const { container } = render(<NeedsAttentionList />);
  await waitFor(() => expect(container.firstChild).toBeNull());
});

test("reports a 200 that carries a failure as a failure, not an approval", async () => {
  // The mutation resolves on non-2xx and a 200 can still say failed_count > 0
  // (review already processed, node execution gone). Toasting "Approved"
  // there leaves the row reappearing with nothing explaining why.
  server.use(
    getGetV2GetPendingReviewsMockHandler200([review]),
    getPostV2ProcessReviewActionMockHandler200({
      approved_count: 0,
      rejected_count: 0,
      failed_count: 1,
      error: "Review already processed",
    }),
  );

  render(
    <>
      <NeedsAttentionList />
      <Toaster />
    </>,
  );
  await userEvent.click(
    await screen.findByRole("button", { name: /^Approve:/ }),
  );

  expect(await screen.findByText("Failed to approve review")).toBeDefined();
  expect(screen.getByText("Review already processed")).toBeDefined();
  expect(screen.queryByText("Approved")).toBeNull();
});

test("a second row's decision does not unlock the first one mid-flight", async () => {
  const other = {
    ...review,
    node_exec_id: "ne-2",
    instructions: "Approve invoice",
  };
  server.use(
    getGetV2GetPendingReviewsMockHandler200([review, other]),
    http.post("/api/proxy/api/review/action", async () => {
      await new Promise((resolve) => setTimeout(resolve, 300));
      return HttpResponse.json({
        approved_count: 1,
        rejected_count: 0,
        failed_count: 0,
      });
    }),
  );

  render(<NeedsAttentionList />);
  const buttons = await screen.findAllByRole("button", { name: /^Approve:/ });
  await userEvent.click(buttons[0]);
  await userEvent.click(buttons[1]);

  await waitFor(() => {
    const [first, second] = screen.getAllByRole("button", {
      name: /^Approve:/,
    }) as HTMLButtonElement[];
    // Both in flight -> both locked. A single pending slot would have
    // re-enabled the first row here, making it double-submittable.
    expect(first.disabled).toBe(true);
    expect(second.disabled).toBe(true);
  });
});
