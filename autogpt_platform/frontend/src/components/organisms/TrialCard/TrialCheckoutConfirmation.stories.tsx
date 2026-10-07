import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { TrialCheckoutConfirmation } from "./TrialCheckoutConfirmation";

const RETURN_FROM_CHECKOUT = {
  appDirectory: true,
  navigation: { pathname: "/settings/billing", query: { trial: "success" } },
};

const meta = {
  title: "Organisms/TrialCheckoutConfirmation",
  component: TrialCheckoutConfirmation,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-96">
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
    layout: "centered",
    a11y: { test: "error" },
    nextjs: RETURN_FROM_CHECKOUT,
    msw: {
      handlers: [
        http.post("*/api/credits/trial/confirm", async () => {
          await delay("infinite");
          return HttpResponse.json({});
        }),
      ],
    },
    docs: {
      description: {
        component:
          "Shown when the user lands back from Stripe checkout with `?trial=success`. It confirms the trial with the backend, shows a status line while that runs, an error card with retry if it fails, and nothing once the trial is confirmed or when the page was not a checkout return. These stories set the query string through the Next.js navigation parameters and sign in a stub user through the auth store.",
      },
    },
  },
} satisfies Meta<typeof TrialCheckoutConfirmation>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Confirming: Story = {};

export const ConfirmationFailed: Story = {
  parameters: {
    msw: {
      handlers: [
        http.post("*/api/credits/trial/confirm", () =>
          HttpResponse.json(
            { detail: "Could not confirm your trial." },
            { status: 500 },
          ),
        ),
      ],
    },
  },
};

export const TrialNotActive: Story = {
  parameters: {
    msw: {
      handlers: [
        http.post("*/api/credits/trial/confirm", () =>
          HttpResponse.json({
            eligible: false,
            active: false,
            status: "inactive",
          }),
        ),
      ],
    },
  },
};
