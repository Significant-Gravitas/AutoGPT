import type { Meta, StoryObj } from "@storybook/react";
import { Navbar } from "./Navbar";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPTUI/Navbar",
  component: Navbar,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    userName: { control: "text" },
    links: { control: "object" },
    activeLink: { control: "text" },
    onProfileClick: { action: "profileClicked" },
  },
} satisfies Meta<typeof Navbar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    userName: "John Doe",
    links: [
      { name: "Marketplace", href: "/" },
      { name: "Library", href: "/agents" },
      { name: "Build", href: "/tasks" },
    ],
    activeLink: "/",
    onProfileClick: () => console.log("Profile clicked"),
  },
};

export const WithActiveLink: Story = {
  args: {
    ...Default.args,
    activeLink: "/agents",
  },
};

export const LongUserName: Story = {
  args: {
    ...Default.args,
    userName: "John Doe with a Very Long Name",
  },
};

export const ManyLinks: Story = {
  args: {
    userName: "Jane Smith",
    links: [
      { name: "Home", href: "/" },
      { name: "Agents", href: "/agents" },
      { name: "Tasks", href: "/tasks" },
      { name: "Analytics", href: "/analytics" },
      { name: "Settings", href: "/settings" },
    ],
    activeLink: "/analytics",
    onProfileClick: () => console.log("Profile clicked"),
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const profileElement = canvas.getByText("John Doe");

    await userEvent.click(profileElement);
  },
};
