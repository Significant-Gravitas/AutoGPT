import { Meta, StoryObj } from "@storybook/react";
import CreditsCard from "./CreditsCard";

const meta: Meta<typeof CreditsCard> = {
  title: "AGPT UI/CreditsCard",
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
