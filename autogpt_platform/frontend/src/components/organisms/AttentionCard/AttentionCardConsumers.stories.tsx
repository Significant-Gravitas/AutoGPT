import type { Meta, StoryObj } from "@storybook/nextjs";
import type { ComponentProps, ReactNode } from "react";
import {
  getGetV2GetChatShareStateMockHandler200,
  getGetV2ListChatTransportsMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { BriefingRunItem } from "@/app/api/__generated__/models/briefingRunItem";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { ArchivedExpertNotice } from "@/app/(platform)/copilot/components/ChatContainer/components/ArchivedExpertNotice";
import { SharedChatNotice } from "@/app/(platform)/copilot/components/ChatContainer/components/SharedChatNotice";
import { AtResetFilesNotice } from "@/app/(platform)/copilot/components/ChatInput/components/AtResetFilesNotice";
import { AttachmentCapNotice } from "@/app/(platform)/copilot/components/ChatInput/components/AttachmentCapNotice";
import { NoAIConnectionNotice } from "@/app/(platform)/copilot/components/ChatInput/components/NoAIConnectionNotice/NoAIConnectionNotice";
import { CredentialRejectionNotice } from "@/app/(platform)/copilot/components/CredentialRejectionNotice/CredentialRejectionNotice";
import { ScheduledChatNoticeCard } from "@/app/(platform)/copilot/components/ScheduledChatNoticeCard/ScheduledChatNoticeCard";
import { BriefingCard } from "@/components/organisms/BriefingCard/BriefingCard";
import { NeedsAttentionRow } from "@/components/organisms/NeedsAttentionList/components/NeedsAttentionRow";
import { PendingReviewCard } from "@/components/organisms/PendingReviewCard/PendingReviewCard";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { TrialCardContent } from "@/components/organisms/TrialCard/TrialCard";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";

// Every card built on AttentionCard, rendered through its real component, so
// a before/after screenshot of this file proves the move kept every pixel.

const meta = {
  title: "Organisms/AttentionCard/Consumers",
  parameters: { layout: "fullscreen" },
  // Icon-only Buttons (AttachmentCapNotice's dismiss) label themselves with a
  // Tooltip.
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
} satisfies Meta;

export default meta;
type Story = StoryObj<typeof meta>;

const review: PendingHumanReviewModel = {
  node_exec_id: "ne-1",
  node_id: "n-1",
  user_id: "u-1",
  graph_exec_id: "run-1",
  graph_id: "g-1",
  graph_version: 1,
  payload: "Hi Sam, following up on our call about the Q3 rollout.",
  instructions: "Approve outreach email",
  editable: true,
  status: "WAITING",
  expert_id: "exp-1",
  expert_name: "Ana",
  expert_avatar_url: null,
  agent_name: "Lead Finder",
  library_agent_id: "lib-1",
  session_id: null,
  created_at: new Date("2026-01-15T11:00:00Z"),
};

function Frame({ children }: { children: ReactNode }) {
  return <div className="flex max-w-2xl flex-col gap-4 p-4">{children}</div>;
}

export const NeedsAttention: Story = {
  render: () => (
    <Frame>
      <NeedsAttentionRow
        review={review}
        isProcessing={false}
        onApprove={() => {}}
        onDecline={() => {}}
      />
      <NeedsAttentionRow
        review={{
          ...review,
          node_exec_id: "ne-2",
          expert_name: null,
          agent_name: null,
          instructions:
            "A very long review title that has to truncate before it reaches the approve and decline buttons on narrow screens",
        }}
        isProcessing
        onApprove={() => {}}
        onDecline={() => {}}
      />
    </Frame>
  ),
};

export const ComposerNotices: Story = {
  parameters: {
    msw: {
      handlers: [getGetV2ListChatTransportsMockHandler200({ transports: [] })],
    },
  },
  render: () => (
    <Frame>
      <AttachmentCapNotice refusedCount={2} onDismiss={() => {}} />
      <AtResetFilesNotice />
      <AtResetFilesNotice className="mt-1.5" />
      <NoAIConnectionNotice enabled />
    </Frame>
  ),
};

export const ChatNotices: Story = {
  parameters: {
    msw: {
      handlers: [
        getGetV2GetChatShareStateMockHandler200({
          is_shared: true,
          share_token: "storybook-share-token",
        }),
      ],
    },
  },
  render: () => (
    <Frame>
      <SharedChatNotice sessionId="session-1" />
      <CredentialRejectionNotice
        rejection={{
          provider: "github",
          credential_title: "GitHub (work)",
          status_code: 401,
          detail: "Bad credentials: the token was revoked or has expired.",
        }}
      />
      <CredentialRejectionNotice
        rejection={{ provider: "github", detail: "The provider refused it." }}
      />
      <ArchivedExpertNotice expertName="Ana" reason="fired" />
      <ArchivedExpertNotice expertName="Ana" reason="unavailable" />
    </Frame>
  ),
};

export const ScheduledChatNotices: Story = {
  render: () => (
    <Frame>
      <ScheduledChatNoticeCard
        notice={{
          reason: "usage",
          headline: "Paused until your usage resets",
          body: "This routine ran out of usage for today. It picks up again when your limit resets.",
          scheduleKey: "sched-1",
          actions: [
            { kind: "link", label: "View plans", href: "/settings/billing" },
            { kind: "run_less_often", label: "Run less often" },
            { kind: "resume", label: "Resume" },
          ],
        }}
      />
      <ScheduledChatNoticeCard
        notice={{
          reason: "resumed",
          headline: "Routine resumed",
          body: "It runs on its usual schedule again.",
          scheduleKey: null,
          actions: [],
        }}
      />
    </Frame>
  ),
};

export const PendingReviews: Story = {
  render: () => (
    <Frame>
      <PendingReviewsList
        reviews={[
          review,
          { ...review, node_exec_id: "ne-2", payload: "Second draft" },
          {
            ...review,
            node_exec_id: "ne-3",
            node_id: "n-2",
            agent_name: null,
            instructions: "Post the weekly summary",
            payload: 42,
            editable: false,
          },
        ]}
      />
    </Frame>
  ),
};

const spendReview: PendingHumanReviewModel = {
  ...review,
  node_exec_id: "expert-spend:exp-1:run-1",
  node_id: "expert-spend:exp-1",
  payload: {
    expert: "Ana",
    spent_credits: 300,
    threshold_credits: 250,
    window: "week",
  },
  instructions:
    "Ana has used 300 of 250 credits this week and needs your approval to keep spending",
  editable: false,
};

export const PendingSpendApproval: Story = {
  render: () => (
    <Frame>
      <PendingReviewsList reviews={[spendReview]} />
      <PendingReviewsList
        reviews={[
          spendReview,
          { ...spendReview, node_exec_id: "expert-spend:exp-1:run-2" },
          { ...spendReview, node_exec_id: "expert-spend:exp-1:run-3" },
        ]}
      />
    </Frame>
  ),
};

export const PendingReviewsEmpty: Story = {
  render: () => (
    <Frame>
      <PendingReviewsList reviews={[]} />
    </Frame>
  ),
};

export const PendingReviewCardStandalone: Story = {
  render: () => (
    <Frame>
      <PendingReviewCard
        review={{ ...review, payload: true }}
        nodeId="node-123456789"
        onReviewDataChange={() => {}}
        onAutoApproveFutureChange={() => {}}
      />
    </Frame>
  ),
};

function runItem(index: number, status = "COMPLETED"): BriefingRunItem {
  return {
    expert_id: `exp-${index}`,
    expert_name: `Expert ${index}`,
    expert_avatar_url: null,
    agent_name: `Workflow ${index}`,
    graph_id: `g-${index}`,
    execution_id: `run-${index}`,
    library_agent_id: `lib-${index}`,
    status,
    summary: `Summary of what run ${index} found overnight.`,
    link: index % 2 ? `/library/agents/lib-${index}` : null,
  };
}

function briefing(count: number) {
  return {
    id: "briefing-1",
    briefing_date: new Date("2026-01-15T00:00:00Z"),
    created_at: new Date("2026-01-15T00:00:00Z"),
    delivered_at: null,
    content: {
      generated_at: new Date("2026-01-15T00:00:00Z"),
      timezone: "UTC",
      zero_expert_fallback: false,
      run_items: Array.from({ length: count }, (_, i) =>
        runItem(i + 1, i === 1 ? "FAILED" : "COMPLETED"),
      ),
      decision_items: [],
    },
  };
}

export const Briefing: Story = {
  render: () => (
    <Frame>
      <BriefingCard briefing={briefing(3)} />
      <BriefingCard briefing={briefing(6)} />
    </Frame>
  ),
};

type Controller = ComponentProps<typeof TrialCardContent>["controller"];

const offer = {
  token: "a".repeat(64),
  version: "storybook-trial",
  duration_days: 7,
  tier: "PRO",
  billing_cycle: "monthly",
  unit_amount: 2000,
  currency: "usd",
  onboarding_credit_amount: 300,
} as const;

const controller = {
  userID: "user-1",
  trial: { eligible: true, offer },
  isLoading: false,
  error: null,
  queryError: false,
  retry: () => {},
  isStarting: false,
  isBlocked: false,
  startTrial: async () => {},
  isCanceling: false,
  isResuming: false,
  cancelTrial: async () => {},
  resumeTrial: async () => {},
  showCanceledDialog: false,
  dismissCanceledDialog: () => {},
  subscribeNow: () => {},
} as unknown as Controller;

export const TrialOnboarding: Story = {
  render: () => (
    <Frame>
      <TrialCardContent returnTo="onboarding" controller={controller} />
    </Frame>
  ),
};

export const TrialBilling: Story = {
  render: () => (
    <Frame>
      <TrialCardContent returnTo="billing" controller={controller} />
    </Frame>
  ),
};
