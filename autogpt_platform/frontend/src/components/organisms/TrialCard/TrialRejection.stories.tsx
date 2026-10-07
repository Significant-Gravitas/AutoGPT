import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { TrialRejection } from "./TrialRejection";

const meta = {
  title: "Organisms/TrialRejection",
  component: TrialRejection,
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
          "Explains why a trial was not activated, either because the introductory offer was already used or because the card could not be verified, and links to support.",
      },
    },
  },
  args: { reason: "intro_offer_already_used" },
} satisfies Meta<typeof TrialRejection>;

export default meta;
type Story = StoryObj<typeof meta>;

export const IntroductoryOfferUsed: Story = {};

export const CardVerificationFailed: Story = {
  args: { reason: "card_verification_failed" },
};
