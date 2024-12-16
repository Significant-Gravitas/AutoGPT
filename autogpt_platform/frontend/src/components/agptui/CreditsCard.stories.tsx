import type { Meta, StoryObj } from "@storybook/react";
import CreditsCard from "./CreditsCard";
import { userEvent, within } from "@storybook/test";

const meta: Meta<typeof CreditsCard> = {
  title: "AGPT UI/Credits Card",
  component: CreditsCard,
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof CreditsCard>;

export const Default: Story = {
  args: {
    credits: 0,
  },
};

export const SmallNumber: Story = {
  args: {
    credits: 10,
  },
};

export const LargeNumber: Story = {
  args: {
    credits: 1000000,
  },
};

export const InteractionTest: Story = {
  args: {
    credits: 100,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const refreshButton = canvas.getByRole("button", {
      name: /refresh credits/i,
    });
    await userEvent.click(refreshButton);
  },
};
