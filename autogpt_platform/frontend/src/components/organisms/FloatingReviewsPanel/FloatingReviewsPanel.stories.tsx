import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { userEvent, within } from "storybook/test";
import {
  getGetV2GetPendingReviewsForExecutionMockHandler200,
  getPostV2ProcessReviewActionMockHandler200,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { getGetV1GetExecutionDetailsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { FloatingReviewsPanel } from "./FloatingReviewsPanel";

const GRAPH_ID = "graph-1";
const EXECUTION_ID = "run-1";

function execution(status: AgentExecutionStatus): GraphExecution {
  return {
    id: EXECUTION_ID,
    user_id: "user-1",
    graph_id: GRAPH_ID,
    graph_version: 3,
    inputs: {},
    credential_inputs: null,
    nodes_input_masks: null,
    preset_id: null,
    status,
    started_at: new Date("2026-10-01T09:00:00Z"),
    ended_at: null,
    stats: null,
    outputs: {},
  };
}

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "node-exec-1",
    node_id: "node-send-email",
    user_id: "user-1",
    graph_exec_id: EXECUTION_ID,
    graph_id: GRAPH_ID,
    graph_version: 3,
    payload: {
      to: "finance@acme.com",
      subject: "Invoice #4821 is overdue",
    },
    action: "Send Email",
    agent_name: "Invoice follow-up",
    editable: true,
    status: "WAITING",
    created_at: new Date("2026-10-01T09:00:00Z"),
    ...overrides,
  };
}

const REVIEWS = [
  makeReview(),
  makeReview({
    node_exec_id: "node-exec-2",
    node_id: "node-merge-pr",
    action: "Merge Pull Request",
    payload: { repo: "acme/web", pull_request: 482 },
  }),
];

function handlersFor(
  status: AgentExecutionStatus,
  reviews: PendingHumanReviewModel[],
) {
  return [
    getGetV1GetExecutionDetailsMockHandler200(execution(status)),
    getGetV2GetPendingReviewsForExecutionMockHandler200(reviews),
    getPostV2ProcessReviewActionMockHandler200({
      approved_count: 1,
      rejected_count: 0,
      failed_count: 0,
    }),
  ];
}

const meta = {
  title: "Organisms/FloatingReviewsPanel",
  component: FloatingReviewsPanel,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="min-h-screen">
        <Story />
      </div>
    ),
  ],
  beforeEach: () => {
    useGraphStore.setState({
      graphExecutionStatus: AgentExecutionStatus.REVIEW,
    });
    return () => {
      useGraphStore.setState({ graphExecutionStatus: undefined });
    };
  },
  parameters: {
    layout: "fullscreen",
    a11y: { test: "error" },
    msw: { handlers: handlersFor(AgentExecutionStatus.REVIEW, REVIEWS) },
    docs: {
      description: {
        component:
          "Builder overlay pinned to the bottom-right of the canvas. While a run is paused for human review it shows a pending-review count; opening it shows the full pending reviews list for that run. It hides while the run is still running or queued, and when nothing is waiting. The run status also comes from the builder's graph store, which these stories set to REVIEW.",
      },
    },
  },
  args: {
    graphId: GRAPH_ID,
    executionId: EXECUTION_ID,
  },
} satisfies Meta<typeof FloatingReviewsPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Collapsed: Story = {};

export const SingleReviewCollapsed: Story = {
  parameters: {
    msw: {
      handlers: handlersFor(AgentExecutionStatus.REVIEW, [makeReview()]),
    },
  },
};

export const Expanded: Story = {
  play: async ({ canvasElement }) => {
    await userEvent.click(
      await within(canvasElement).findByRole("button", {
        name: /Reviews Pending/,
      }),
    );
  },
};

export const HiddenWhileRunning: Story = {
  parameters: {
    msw: { handlers: handlersFor(AgentExecutionStatus.RUNNING, REVIEWS) },
  },
};
