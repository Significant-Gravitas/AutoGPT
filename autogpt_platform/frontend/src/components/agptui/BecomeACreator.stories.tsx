import type { Meta, StoryObj } from "@storybook/react";
import { BecomeACreator } from "./BecomeACreator";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPT UI/Become A Creator",
  component: BecomeACreator,
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    title: { control: "text" },
    description: { control: "text" },
    buttonText: { control: "text" },
    onButtonClick: { action: "buttonClicked" },
  },
} satisfies Meta<typeof BecomeACreator>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    title: "Want to contribute?",
    description: "Join our ever-growing community of hackers and tinkerers",
    buttonText: "Become a Creator",
    onButtonClick: () => console.log("Button clicked"),
  },
};

export const CustomText: Story = {
  args: {
    title: "Become a Creator Today!",
    description: "Share your ideas and build amazing AI agents with us",
    buttonText: "Start Creating",
    onButtonClick: () => console.log("Custom button clicked"),
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const button = canvas.getByText("Become a Creator");

    await userEvent.click(button);
  },
};
