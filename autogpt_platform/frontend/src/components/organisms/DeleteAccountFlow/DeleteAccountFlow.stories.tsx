import {
  getPostAccountDeleteAccountMockHandler200,
  getPostAccountSendExitFeedbackMockHandler204,
} from "@/app/api/__generated__/endpoints/account/account.msw";
import { getGetSubscriptionStatusMockHandler200 } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import {
  getDeleteV1DeleteCredentialsMockHandler200,
  getGetV1ListCredentialsMockHandler200,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getGetV2ListMySkillSubmissionsMockHandler200,
  getGetV2ListMySubmissionsMockHandler200,
  getGetV2ListMySubmissionsResponseMock200,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { useState, type ComponentProps } from "react";
import { fn } from "storybook/test";
import { DeleteAccountFlow } from "./DeleteAccountFlow";

const PERIOD_END = Date.parse("2030-09-17T15:00:00Z") / 1000;

const APPS: CredentialsMetaResponse[] = [
  "Gmail",
  "Google Calendar",
  "Notion",
].map((name, index) => ({
  id: `cred-${index}`,
  provider: "google",
  type: "oauth2",
  title: name,
  scopes: [],
  username: null,
  is_managed: false,
  service_name: name,
}));

function plan(
  overrides: Partial<SubscriptionStatusResponse> = {},
): SubscriptionStatusResponse {
  return {
    tier: "PRO",
    monthly_cost: 5000,
    tier_costs: { PRO: 5000, MAX: 32000 },
    proration_credit_cents: 0,
    has_active_stripe_subscription: true,
    current_period_end: PERIOD_END,
    pending_tier: null,
    pending_tier_effective_at: null,
    ...overrides,
  };
}

interface Scenario {
  subscription?: SubscriptionStatusResponse;
  trial?: TrialStatusResponse;
  hasListings?: boolean;
}

function handlers({
  subscription = plan(),
  trial = { eligible: false, active: false },
  hasListings = false,
}: Scenario = {}) {
  return [
    getGetSubscriptionStatusMockHandler200(subscription),
    getGetTrialsGetTrialStatusMockHandler200(trial),
    getGetV1ListCredentialsMockHandler200(APPS),
    getGetV2ListMySubmissionsMockHandler200(
      hasListings
        ? getGetV2ListMySubmissionsResponseMock200()
        : getGetV2ListMySubmissionsResponseMock200({ submissions: [] }),
    ),
    getGetV2ListMySkillSubmissionsMockHandler200([]),
    getDeleteV1DeleteCredentialsMockHandler200({
      deleted: true,
      revoked: true,
    }),
    getPostAccountSendExitFeedbackMockHandler204(),
    getPostAccountDeleteAccountMockHandler200({ status: "scheduled" }),
  ];
}

const meta = {
  title: "Organisms/DeleteAccountFlow",
  component: DeleteAccountFlow,
  args: { source: "settings", open: true, onOpenChange: fn() },
  loaders: [
    async () => {
      useAuthStore.setState({
        user: {
          id: "storybook-user",
          email: "jane@example.com",
          user_metadata: {},
        },
        hasLoadedUser: true,
        isUserLoading: false,
      });
      return {};
    },
  ],
  decorators: [
    (Story) => (
      <BackendAPIProvider>
        <Story />
      </BackendAPIProvider>
    ),
  ],
  parameters: { msw: { handlers: handlers() } },
  render: function Render(args) {
    return <ReopenableFlow {...args} />;
  },
} satisfies Meta<typeof DeleteAccountFlow>;

export default meta;
type Story = StoryObj<typeof meta>;

function ReopenableFlow(args: ComponentProps<typeof DeleteAccountFlow>) {
  const [open, setOpen] = useState(args.open);

  function handleOpenChange(next: boolean) {
    args.onOpenChange(next);
    setOpen(next);
  }

  return (
    <>
      <Button variant="secondary" onClick={() => setOpen(true)}>
        Delete account
      </Button>
      <DeleteAccountFlow
        source={args.source}
        open={open}
        onOpenChange={handleOpenChange}
      />
    </>
  );
}

/** Pro plan from Settings: pick a reason to see its alternative. */
export const ProFromSettings: Story = {};

export const MaxWithPendingCancel: Story = {
  parameters: {
    msw: {
      handlers: handlers({
        subscription: plan({
          tier: "MAX",
          pending_tier: "NO_TIER",
          pending_tier_effective_at: new Date(PERIOD_END * 1000),
        }),
        hasListings: true,
      }),
    },
  },
};

export const TrialEndingSoon: Story = {
  parameters: {
    msw: {
      handlers: handlers({
        subscription: plan({ tier: "TRIAL", current_period_end: null }),
        trial: {
          eligible: false,
          active: true,
          status: "trialing",
          cancel_at_period_end: true,
          ends_at: new Date(PERIOD_END * 1000),
        },
      }),
    },
  },
};

/** From the paywall the privacy alternative disconnects apps in place. */
export const NoPlanFromPaywall: Story = {
  args: { source: "paywall" },
  parameters: {
    msw: {
      handlers: handlers({
        subscription: plan({
          tier: "NO_TIER",
          has_active_stripe_subscription: false,
          current_period_end: null,
        }),
      }),
    },
  },
};

export const DeleteRejected: Story = {
  parameters: {
    msw: {
      handlers: [
        http.post("*/api/account/deletion", () =>
          HttpResponse.json(
            {
              detail:
                "We couldn't cancel your plan. Nothing was deleted. Try again.",
            },
            { status: 502 },
          ),
        ),
        ...handlers(),
      ],
    },
  },
};
