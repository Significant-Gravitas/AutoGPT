import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { userEvent, within } from "storybook/test";
import {
  getGetV2GetPendingReviewsMockHandler200,
  getPostV2ProcessReviewActionMockHandler200,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { NeedsAttentionList } from "./NeedsAttentionList";

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "node-exec-1",
    node_id: "node-1",
    user_id: "user-1",
    graph_exec_id: "run-1",
    graph_id: "graph-1",
    graph_version: 1,
    payload: { to: "lead@northwind.com" },
    instructions: "Approve outreach email to Northwind",
    editable: true,
    status: "WAITING",
    expert_id: "expert-ana",
    expert_name: "Ana",
    expert_avatar_url: null,
    agent_name: "Lead Finder",
    library_agent_id: "library-agent-1",
    session_id: null,
    created_at: new Date("2026-10-01T09:00:00Z"),
    ...overrides,
  };
}

const REVIEWS = [
  makeReview(),
  makeReview({
    node_exec_id: "node-exec-2",
    instructions: "Approve invoice #4821 reminder",
    expert_id: "expert-marco",
    expert_name: "Marco",
    agent_name: "Invoice follow-up",
  }),
  makeReview({
    node_exec_id: "node-exec-3",
    instructions: "Publish the weekly product update to LinkedIn",
    expert_id: "expert-nova",
    expert_name: "Nova",
    agent_name: "Social Scheduler",
  }),
];

const MANY_REVIEWS = Array.from({ length: 100 }, (_, index) =>
  makeReview({
    node_exec_id: `node-exec-bulk-${index}`,
    instructions: `Approve outreach email #${index + 1}`,
  }),
);

const REVIEW_ACTION_OK = getPostV2ProcessReviewActionMockHandler200({
  approved_count: 1,
  rejected_count: 0,
  failed_count: 0,
});

const meta = {
  title: "Organisms/NeedsAttentionList",
  component: NeedsAttentionList,
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
    a11y: { test: "error" },
    msw: {
      handlers: [
        getGetV2GetPendingReviewsMockHandler200(REVIEWS),
        REVIEW_ACTION_OK,
      ],
    },
    docs: {
      description: {
        component:
          "Home-page triage list of every pending review across the user's agents. Each row approves in one tap; decline needs a second, deliberate tap. Only the row being decided locks while its request is in flight. Renders nothing when there is nothing to review.",
      },
    },
  },
} satisfies Meta<typeof NeedsAttentionList>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const SingleReview: Story = {
  parameters: {
    msw: {
      handlers: [
        getGetV2GetPendingReviewsMockHandler200([makeReview()]),
        REVIEW_ACTION_OK,
      ],
    },
  },
};

export const WithoutAttribution: Story = {
  parameters: {
    msw: {
      handlers: [
        getGetV2GetPendingReviewsMockHandler200([
          makeReview({
            instructions: null,
            expert_id: null,
            expert_name: null,
            agent_name: null,
          }),
        ]),
        REVIEW_ACTION_OK,
      ],
    },
  },
};

export const FullPage: Story = {
  parameters: {
    msw: {
      handlers: [
        getGetV2GetPendingReviewsMockHandler200(MANY_REVIEWS),
        REVIEW_ACTION_OK,
      ],
    },
  },
};

export const Loading: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/review/pending", async () => {
          await delay("infinite");
          return HttpResponse.json([]);
        }),
      ],
    },
  },
};

export const LoadError: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/review/pending", () =>
          HttpResponse.json(
            { detail: "Failed to load pending reviews" },
            { status: 500 },
          ),
        ),
      ],
    },
  },
};

export const Empty: Story = {
  parameters: {
    msw: { handlers: [getGetV2GetPendingReviewsMockHandler200([])] },
  },
};

export const DeclineArmed: Story = {
  play: async ({ canvasElement }) => {
    const [decline] = await within(canvasElement).findAllByRole("button", {
      name: /^Decline:/,
    });
    await userEvent.click(decline);
  },
};

export const ApprovalInFlight: Story = {
  parameters: {
    msw: {
      handlers: [
        getGetV2GetPendingReviewsMockHandler200(REVIEWS),
        http.post("*/api/review/action", async () => {
          await delay("infinite");
          return HttpResponse.json({});
        }),
      ],
    },
  },
  play: async ({ canvasElement }) => {
    const [approve] = await within(canvasElement).findAllByRole("button", {
      name: /^Approve:/,
    });
    await userEvent.click(approve);
  },
};
