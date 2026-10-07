import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { fn } from "storybook/test";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { PendingReviewCard } from "./PendingReviewCard";

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "node-exec-7f3a9c21",
    node_id: "node-4b8e2d10",
    user_id: "user-1",
    graph_exec_id: "run-1",
    graph_id: "graph-1",
    graph_version: 3,
    payload: {
      to: "finance@acme.com",
      subject: "Invoice #4821 is overdue",
      body: "Hi team, a friendly reminder that invoice #4821 is now 14 days overdue.",
    },
    instructions: "Check the follow-up email before it goes out.",
    agent_name: "Invoice follow-up",
    editable: true,
    status: "WAITING",
    created_at: new Date("2026-10-01T09:00:00Z"),
    ...overrides,
  };
}

const DISCORD_BLOCK_SCHEMA = http.get("*/api/builder/blocks/batch", () =>
  HttpResponse.json([
    {
      id: "block-discord",
      name: "SendDiscordMessageBlock",
      inputSchema: {
        type: "object",
        properties: {
          credentials: {
            title: "Credentials",
            credentials_provider: ["discord"],
          },
          message_content: { title: "Message Content", type: "string" },
          webhook_secret: {
            title: "Webhook Secret",
            type: "string",
            secret: true,
          },
        },
      },
    },
  ]),
);

const BLOCK_REVIEW = makeReview({
  block_id: "block-discord",
  action: "Send Discord Message",
  payload: {
    credentials: {
      id: "cred-1",
      provider: "discord",
      type: "api_key",
      title: "Launch bot",
    },
    message_content: "v2.4 is live! Release notes are in #announcements.",
    webhook_secret: "storybook-placeholder",
  },
});

const meta = {
  title: "Organisms/PendingReviewCard",
  component: PendingReviewCard,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-full max-w-xl">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "centered",
    a11y: { test: "error" },
    docs: {
      description: {
        component:
          "One human-in-the-loop review inside a pending reviews list. Object payloads render as labelled inputs (using the block's input schema when the review names a block), primitives as a single input, and non-editable payloads as static text. Credentials and secret fields are never shown in full.",
      },
    },
  },
  args: {
    review: makeReview(),
    onReviewDataChange: fn(),
  },
} satisfies Meta<typeof PendingReviewCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const ReadOnly: Story = {
  args: { review: makeReview({ editable: false }) },
};

export const WithNodeId: Story = {
  args: { nodeId: "node-4b8e2d10" },
};

export const NestedObject: Story = {
  args: {
    review: makeReview({
      payload: {
        customer: { name: "Acme Corp", tier: "Enterprise" },
        amount: 1250,
        send_receipt: true,
        recipients: ["finance@acme.com", "ops@acme.com"],
      },
    }),
  },
};

export const StringPayload: Story = {
  args: {
    review: makeReview({
      payload: "Draft a short thank-you note for the Q3 partners.",
    }),
  },
};

export const NumberPayload: Story = {
  args: { review: makeReview({ payload: 250 }) },
};

export const BooleanPayload: Story = {
  args: { review: makeReview({ payload: true }) },
};

export const ArrayPayload: Story = {
  args: {
    review: makeReview({
      payload: ["finance@acme.com", "ops@acme.com", "ceo@acme.com"],
    }),
  },
};

export const ReadOnlyPrimitive: Story = {
  args: {
    review: makeReview({
      editable: false,
      payload: "rm -rf ./tmp/build-cache",
    }),
  },
};

export const AutoApproveToggle: Story = {
  args: {
    showAutoApprove: true,
    onAutoApproveFutureChange: fn(),
  },
};

export const AutoApproveEnabled: Story = {
  args: {
    showAutoApprove: true,
    autoApproveFuture: true,
    onAutoApproveFutureChange: fn(),
  },
};

export const BlockWithCredentials: Story = {
  args: { review: BLOCK_REVIEW },
  parameters: { msw: { handlers: [DISCORD_BLOCK_SCHEMA] } },
};

export const BlockSchemaLoading: Story = {
  args: { review: BLOCK_REVIEW },
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/builder/blocks/batch", async () => {
          await delay("infinite");
          return HttpResponse.json([]);
        }),
      ],
    },
  },
};
