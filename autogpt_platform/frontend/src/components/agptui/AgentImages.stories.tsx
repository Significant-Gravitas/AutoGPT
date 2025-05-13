import type { Meta, StoryObj } from "@storybook/react";
import { AgentImages } from "./AgentImages";

const meta = {
  title: "Agpt UI/marketing/Agent Images",
  component: AgentImages,
  decorators: [
    (Story) => (
      <div className="mx-auto flex h-full w-[80%] items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    images: { control: "object" },
  },
} satisfies Meta<typeof AgentImages>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    images: [
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
      "https://youtu.be/KWonAsyKF3g?si=JMibxlN_6OVo6LhJ",
      "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    ],
  },
};
