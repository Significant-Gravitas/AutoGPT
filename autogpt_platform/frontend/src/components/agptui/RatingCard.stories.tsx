import type { Meta, StoryObj } from "@storybook/react";
import { RatingCard } from "./RatingCard";

const meta = {
  title: "AGPT UI/RatingCard",
  component: RatingCard,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof RatingCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    agentName: "Test Agent",
    // onSubmit: (rating) => {
    //   console.log("Rating submitted:", rating);
    // },
    // onClose: () => {
    //   console.log("Rating card closed");
    // },
    storeListingVersionId: "1",
  },
};

export const LongAgentName: Story = {
  args: {
    agentName: "Very Long Agent Name That Might Need Special Handling",
    // onSubmit: (rating) => {
    //   console.log("Rating submitted:", rating);
    // },
    // onClose: () => {
    //   console.log("Rating card closed");
    // },
    storeListingVersionId: "1",
  },
};

export const WithoutCallbacks: Story = {
  args: {
    agentName: "Test Agent",
    storeListingVersionId: "1",
  },
};
