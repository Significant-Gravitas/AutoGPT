import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { screen, userEvent, within } from "storybook/test";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { TrialCard } from "./TrialCard";

const OFFER = {
  token: "a".repeat(64),
  version: "storybook-trial",
  duration_days: 7,
  tier: "PRO",
  billing_cycle: "monthly",
  unit_amount: 2000,
  currency: "usd",
  onboarding_credit_amount: 300,
} as const;

const ACTIVE_TRIAL: TrialStatusResponse = {
  eligible: false,
  offer: OFFER,
  status: "active",
  active: true,
  ends_at: new Date("2030-09-17T15:00:00Z"),
  allowance_used_percent: 42.4,
  cancel_at_period_end: false,
};

function trialStatus(trial: TrialStatusResponse) {
  return http.get("*/api/credits/trial", () => HttpResponse.json(trial));
}

function pending(method: "get" | "post", path: string) {
  return http[method](path, async () => {
    await delay("infinite");
    return HttpResponse.json({});
  });
}

const meta = {
  title: "Organisms/TrialCard",
  component: TrialCard,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-full max-w-3xl">
        <Story />
      </div>
    ),
  ],
  beforeEach: () => {
    useAuthStore.setState({
      user: {
        id: "storybook-user",
        email: "trial@example.com",
        user_metadata: {},
      },
      isUserLoading: false,
      hasLoadedUser: true,
    });
    return () => {
      useAuthStore.setState({ user: null, hasLoadedUser: false });
    };
  },
  parameters: {
    layout: "padded",
    a11y: { test: "error" },
    msw: { handlers: [trialStatus({ eligible: true, offer: OFFER })] },
    docs: {
      description: {
        component:
          "Loads the signed-in user's trial status and shows either the trial offer (with a Start trial checkout) or the current trial's status (with Cancel trial). The billing surface wraps it in a plain card under a 'Free trial' / 'Your plan' label; the onboarding surface uses a gradient frame. Renders nothing once the trial converted or while a checkout is pending. These stories sign in a stub user through the auth store and mock the trial endpoints with MSW.",
      },
    },
  },
  args: { returnTo: "billing" },
} satisfies Meta<typeof TrialCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Offer: Story = {};

export const OfferOnboarding: Story = {
  args: { returnTo: "onboarding" },
};

export const YearlyOffer: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus({
          eligible: true,
          offer: {
            ...OFFER,
            tier: "MAX",
            billing_cycle: "yearly",
            unit_amount: 192000,
            duration_days: 14,
          },
        }),
      ],
    },
  },
};

export const StartingCheckout: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus({ eligible: true, offer: OFFER }),
        pending("post", "*/api/credits/trial"),
      ],
    },
  },
  play: async ({ canvasElement }) => {
    await userEvent.click(
      await within(canvasElement).findByRole("button", {
        name: /Start 7-day trial/,
      }),
    );
  },
};

export const CheckoutFailed: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus({ eligible: true, offer: OFFER }),
        http.post("*/api/credits/trial", () =>
          HttpResponse.json(
            { detail: "Unable to start trial checkout." },
            { status: 500 },
          ),
        ),
      ],
    },
  },
  play: async ({ canvasElement }) => {
    await userEvent.click(
      await within(canvasElement).findByRole("button", {
        name: /Start 7-day trial/,
      }),
    );
  },
};

export const ActiveTrial: Story = {
  parameters: { msw: { handlers: [trialStatus(ACTIVE_TRIAL)] } },
};

export const ActiveTrialOnboarding: Story = {
  args: { returnTo: "onboarding" },
  parameters: { msw: { handlers: [trialStatus(ACTIVE_TRIAL)] } },
};

export const CancellationScheduled: Story = {
  parameters: {
    msw: {
      handlers: [trialStatus({ ...ACTIVE_TRIAL, cancel_at_period_end: true })],
    },
  },
};

export const ConfirmCancel: Story = {
  parameters: { msw: { handlers: [trialStatus(ACTIVE_TRIAL)] } },
  play: async ({ canvasElement }) => {
    await userEvent.click(
      await within(canvasElement).findByRole("button", {
        name: "Cancel trial",
      }),
    );
    await screen.findByRole("button", { name: "End trial now" });
  },
};

export const Canceling: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus(ACTIVE_TRIAL),
        pending("post", "*/api/credits/trial/cancel"),
      ],
    },
  },
  play: async ({ canvasElement }) => {
    await userEvent.click(
      await within(canvasElement).findByRole("button", {
        name: "Cancel trial",
      }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "End trial now" }),
    );
  },
};

export const Canceled: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus({ ...ACTIVE_TRIAL, active: false, status: "canceled" }),
      ],
    },
  },
};

export const Ended: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus({ ...ACTIVE_TRIAL, active: false, status: "inactive" }),
      ],
    },
  },
};

export const IntroductoryOfferUsed: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus({
          ...ACTIVE_TRIAL,
          active: false,
          status: "canceled",
          rejection_reason: "intro_offer_already_used",
        }),
      ],
    },
  },
};

export const Loading: Story = {
  parameters: {
    msw: { handlers: [pending("get", "*/api/credits/trial")] },
  },
};

export const LoadingOnboarding: Story = {
  args: { returnTo: "onboarding" },
  parameters: {
    msw: { handlers: [pending("get", "*/api/credits/trial")] },
  },
};

export const LoadError: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/credits/trial", () =>
          HttpResponse.json(
            { detail: "Trial service unavailable" },
            { status: 500 },
          ),
        ),
      ],
    },
  },
};

export const HiddenAfterConversion: Story = {
  parameters: {
    msw: {
      handlers: [
        trialStatus({ ...ACTIVE_TRIAL, active: false, converted: true }),
      ],
    },
  },
};
