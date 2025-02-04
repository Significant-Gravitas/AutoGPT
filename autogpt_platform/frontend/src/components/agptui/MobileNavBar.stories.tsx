import type { Meta, StoryObj } from "@storybook/react";
import { MobileNavBar } from "./MobileNavBar";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../ui/icons";

const meta = {
  title: "AGPT UI/Mobile Nav Bar",
  component: MobileNavBar,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    userName: { control: "text" },
    userEmail: { control: "text" },
    avatarSrc: { control: "text" },
    menuItemGroups: { control: "object" },
  },
} satisfies Meta<typeof MobileNavBar>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultMenuItemGroups = [
  {
    items: [
      { icon: IconType.Marketplace, text: "Marketplace", href: "/marketplace" },
      { icon: IconType.Library, text: "Library", href: "/library" },
      { icon: IconType.Builder, text: "Builder", href: "/builder" },
    ],
  },
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
    const menuTrigger = canvas.getByRole("button");

    await userEvent.click(menuTrigger);

    // Wait for the popover to appear
    await canvas.findByText("Edit profile");
  },
};
