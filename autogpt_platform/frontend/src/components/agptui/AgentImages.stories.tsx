import type { Meta, StoryObj } from "@storybook/react";
import { AgentImages } from "./AgentImages";

const meta = {
  title: "AGPT UI/Agent Images",
  component: AgentImages,
  parameters: {
    layout: {
      center: true,
      fullscreen: true,
      padding: 0,
    },
  },
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
      "https://storage.googleapis.com/agpt-dev-website-media/DJINeo.mp4",
    ],
  },
};

export const OnlyImages: Story = {
  args: {
    images: [
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    ],
  },
};

export const WithVideos: Story = {
  args: {
    images: [
      "https://storage.googleapis.com/agpt-dev-website-media/DJINeo.mp4",
      "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
      "https://youtu.be/KWonAsyKF3g?si=JMibxlN_6OVo6LhJ",
    ],
  },
};

export const SingleItem: Story = {
  args: {
    images: [
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    ],
  },
};
