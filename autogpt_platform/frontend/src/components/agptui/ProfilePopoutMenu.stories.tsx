import type { Meta, StoryObj } from "@storybook/react";
import { ProfilePopoutMenu } from "./ProfilePopoutMenu";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../ui/icons";

const meta = {
  title: "AGPT UI/Profile Popout Menu",
  component: ProfilePopoutMenu,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    userName: { control: "text" },
    userEmail: { control: "text" },
    avatarSrc: { control: "text" },
    menuItemGroups: { control: "object" },
    hideNavBarUsername: { control: "boolean" },
  },
} satisfies Meta<typeof ProfilePopoutMenu>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultMenuItemGroups = [
  {
    // Creator actions group
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
    // Profile management group
    items: [
      { icon: IconType.Edit, text: "Edit profile", href: "/profile/edit" },
      { icon: IconType.Settings, text: "Settings", href: "/settings" },
    ],
  },
  {
    // Logout group
    items: [
      {
        icon: IconType.LogOut,
        text: "Log out",
        onClick: () => console.log("Logged out"),
      },
    ],
  },
];

export const Default: Story = {
  args: {
    userName: "John Doe",
    userEmail: "john.doe@example.com",
    avatarSrc: "https://avatars.githubusercontent.com/u/123456789?v=4",
    menuItemGroups: defaultMenuItemGroups,
  },
};

export const NoAvatar: Story = {
  args: {
    userName: "Jane Smith",
    userEmail: "jane.smith@example.com",
    menuItemGroups: defaultMenuItemGroups,
  },
};

export const LongUserName: Story = {
  args: {
    userName: "Alexander Bartholomew Christopherson III",
    userEmail: "alexander@example.com",
    avatarSrc: "https://avatars.githubusercontent.com/u/987654321?v=4",
    menuItemGroups: defaultMenuItemGroups,
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const profileTrigger = canvas.getByText("John Doe");

    await userEvent.click(profileTrigger);

    // Wait for the popover to appear
    await canvas.findByText("Edit profile");
  },
};
