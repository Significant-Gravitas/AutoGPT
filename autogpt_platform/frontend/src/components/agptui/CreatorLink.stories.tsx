import type { Meta, StoryObj } from "@storybook/react";
import CreatorLink from "./CreatorLink";

const meta: Meta<typeof CreatorLink> = {
  title: "Agpt UI/marketing/CreatorLink",
  component: CreatorLink,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
  argTypes: {
    href: { control: "text" },
    children: { control: "text" },
  },
};

export default meta;
type Story = StoryObj<typeof CreatorLink>;

export const Default: Story = {
  args: {
    href: "https://linkedin.com",
    children: "View Creator Profile",
  },
};
