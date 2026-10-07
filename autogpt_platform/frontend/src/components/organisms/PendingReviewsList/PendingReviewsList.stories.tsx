import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { delay, http, HttpResponse } from "msw";
import { fn, userEvent, within } from "storybook/test";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { PendingReviewsList } from "./PendingReviewsList";

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "node-exec-1",
    node_id: "node-send-email",
    user_id: "user-1",
    graph_exec_id: "run-1",
    graph_id: "graph-1",
    graph_version: 3,
    payload: {
      to: "finance@acme.com",
      subject: "Invoice #4821 is overdue",
    },
    instructions: "SendEmailBlock",
    action: "Send Email",
    agent_name: "Invoice follow-up",
    editable: true,
    status: "WAITING",
    created_at: new Date("2026-10-01T09:00:00Z"),
    ...overrides,
  };
}

const SECOND_EMAIL = makeReview({
  node_exec_id: "node-exec-2",
  payload: { to: "ops@acme.com", subject: "Invoice #4822 is overdue" },
});

const MERGE_PR = makeReview({
  node_exec_id: "node-exec-3",
  node_id: "node-merge-pr",
  instructions: "GithubMergePullRequestBlock",
  action: "Merge Pull Request",
  payload: { repo: "acme/web", pull_request: 482, merge_method: "squash" },
});

const GATE_REVIEW = makeReview({
  node_exec_id: "copilot-node-gate-bash_exec:abc",
  node_id: "copilot-node-gate-bash_exec",
  action: undefined,
  agent_name: undefined,
  instructions: "Bash exec — lists files in the workspace",
  editable: false,
  payload: { command: "ls -la ./workspace" },
});

const REVIEW_ACTION_OK = getPostV2ProcessReviewActionMockHandler200({
  approved_count: 1,
  rejected_count: 0,
  failed_count: 0,
});

const meta = {
  title: "Organisms/PendingReviewsList",
  component: PendingReviewsList,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-full max-w-3xl">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "padded",
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    msw: { handlers: [REVIEW_ACTION_OK] },
    docs: {
      description: {
        component:
          "Human-in-the-loop reviews grouped by node. Each group can be collapsed, approved or rejected on its own, and ordinary nodes offer an auto-approve toggle. Copilot action-gate reviews are worded for Otto and never offer auto-approve.",
      },
    },
  },
  args: {
    reviews: [makeReview()],
    onReviewComplete: fn(),
  },
} satisfies Meta<typeof PendingReviewsList>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SingleReview: Story = {};

export const MultipleReviewsOneNode: Story = {
  args: { reviews: [makeReview(), SECOND_EMAIL] },
};

export const MultipleNodes: Story = {
  args: { reviews: [makeReview(), SECOND_EMAIL, MERGE_PR] },
};

export const ReadOnlyPayload: Story = {
  args: { reviews: [makeReview({ editable: false })] },
};

export const WithoutWorkflowName: Story = {
  args: {
    reviews: [
      makeReview({
        action: undefined,
        agent_name: undefined,
        instructions: "Confirm the refund amount before it is issued.",
        node_id: "8f2c41d7-93ab-4e10-b5c2-77a0d1e9f310",
        payload: { order_id: "ORD-20931", refund_amount: 49.5 },
      }),
    ],
  },
};

export const CopilotActionGate: Story = {
  args: { reviews: [GATE_REVIEW] },
};

export const MixedGateAndNode: Story = {
  args: { reviews: [GATE_REVIEW, MERGE_PR] },
};

export const Empty: Story = {
  args: { reviews: [] },
};

export const CustomEmptyMessage: Story = {
  args: { reviews: [], emptyMessage: "Nothing is waiting on you" },
};

export const CollapsedGroup: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: /Send Email/ }),
    );
  },
};

export const AutoApproveOn: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(within(canvasElement).getByRole("switch"));
  },
};

export const Submitting: Story = {
  parameters: {
    msw: {
      handlers: [
        http.post("*/api/review/action", async () => {
          await delay("infinite");
          return HttpResponse.json({});
        }),
      ],
    },
  },
  play: async ({ canvasElement }) => {
    await userEvent.click(
      within(canvasElement).getByRole("button", { name: "Approve" }),
    );
  },
};
