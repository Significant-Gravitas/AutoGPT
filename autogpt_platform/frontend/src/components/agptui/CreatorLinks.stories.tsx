import type { Meta, StoryObj } from "@storybook/react";
import { CreatorLinks } from "./CreatorLinks";

const meta = {
  title: "AGPT UI/Creator Links",
  component: CreatorLinks,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    links: {
      control: "object",
      description: "Object containing various social and web links",
    },
  },
} satisfies Meta<typeof CreatorLinks>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    links: [
      "https://example.com",
      "https://linkedin.com/in/johndoe",
      "https://github.com/johndoe",
      "https://twitter.com/johndoe",
      "https://medium.com/@johndoe",
    ],
  },
};

export const WebsiteOnly: Story = {
  args: {
    links: ["https://example.com"],
  },
};

export const SocialLinks: Story = {
  args: {
    links: [
      "https://linkedin.com/in/janedoe",
      "https://github.com/janedoe",
      "https://twitter.com/janedoe",
    ],
  },
};

export const NoLinks: Story = {
  args: {
    links: [],
  },
};

export const MultipleOtherLinks: Story = {
  args: {
    links: [
      "https://example.com",
      "https://linkedin.com/in/creator",
      "https://github.com/creator",
      "https://twitter.com/creator",
      "https://medium.com/@creator",
      "https://youtube.com/@creator",
      "https://tiktok.com/@creator",
    ],
  },
};
