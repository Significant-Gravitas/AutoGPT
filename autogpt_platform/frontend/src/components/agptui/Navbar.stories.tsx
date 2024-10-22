import type { Meta, StoryObj } from "@storybook/react";
import { Navbar } from "./Navbar";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../ui/icons";

const meta = {
  title: "AGPT UI/Navbar",
  component: Navbar,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
  argTypes: {
    isLoggedIn: { control: "boolean" },
    userName: { control: "text" },
    links: { control: "object" },
    activeLink: { control: "text" },
    avatarSrc: { control: "text" },
    userEmail: { control: "text" },
    menuItemGroups: { control: "object" },
  },
} satisfies Meta<typeof Navbar>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultMenuItemGroups = [
  {
    items: [
      { icon: IconType.Edit, text: "Edit profile", href: "/profile/edit" },
    ],
  },
  {
    items: [
      {
        icon: IconType.LayoutDashboard,
        text: "Creator Dashboard",
        href: "/dashboard",
      },
      {
        icon: IconType.UploadCloud,
        text: "Publish an agent",
        href: "/publish",
      },
    ],
  },
  {
    items: [{ icon: IconType.Settings, text: "Settings", href: "/settings" }],
  },
  {
    items: [
      {
        icon: IconType.LogOut,
        text: "Log out",
        onClick: () => console.log("Logged out"),
      },
    ],
  },
];

const defaultLinks = [
  { name: "Marketplace", href: "/marketplace" },
  { name: "Library", href: "/library" },
  { name: "Build", href: "/builder" },
];

export const Default: Story = {
  args: {
    isLoggedIn: true,
    userName: "John Doe",
    links: defaultLinks,
    activeLink: "/marketplace",
    avatarSrc: "https://avatars.githubusercontent.com/u/123456789?v=4",
    userEmail: "john.doe@example.com",
    menuItemGroups: defaultMenuItemGroups,
  },
};

export const WithActiveLink: Story = {
  args: {
    ...Default.args,
    activeLink: "/library",
  },
};

export const LongUserName: Story = {
  args: {
    ...Default.args,
    userName: "Alexander Bartholomew Christopherson III",
    userEmail: "alexander@example.com",
    avatarSrc: "https://avatars.githubusercontent.com/u/987654321?v=4",
  },
};

export const NoAvatar: Story = {
  args: {
    ...Default.args,
    avatarSrc: undefined,
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const profileTrigger = canvas.getByRole("button");

    await userEvent.click(profileTrigger);

    // Wait for the popover to appear
    await canvas.findByText("Edit profile");
  },
};

export const NotLoggedIn: Story = {
  args: {
    ...Default.args,
    isLoggedIn: false,
    userName: undefined,
    userEmail: undefined,
    avatarSrc: undefined,
  },
};
