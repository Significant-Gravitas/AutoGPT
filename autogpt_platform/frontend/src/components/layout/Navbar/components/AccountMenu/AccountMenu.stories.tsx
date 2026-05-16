import { IconType } from "@/components/__legacy__/ui/icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { AccountMenu } from "./AccountMenu";

const meta: Meta<typeof AccountMenu> = {
  title: "Layout/Navbar/AccountMenu",
  component: AccountMenu,
  parameters: {
    layout: "centered",
  },
};

export default meta;
type Story = StoryObj<typeof AccountMenu>;

const userGroups = [
  {
    items: [
      {
        icon: IconType.Edit,
        text: "Account",
        href: "/settings/profile",
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.LayoutDashboard,
        text: "Creator Dashboard",
        href: "/settings/creator-dashboard",
      },
      {
        icon: IconType.UploadCloud,
        text: "Publish an agent",
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.Settings,
        text: "Settings",
        href: "/settings",
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.LogOut,
        text: "Log out",
      },
    ],
  },
];

const adminGroups = [
  ...userGroups.slice(0, 2),
  {
    items: [
      {
        icon: IconType.Sliders,
        text: "Admin",
        href: "/admin/marketplace",
      },
    ],
  },
  ...userGroups.slice(2),
];

export const Default: Story = {
  args: {
    userName: "abhimanyu",
    userEmail: "abhi@agpt.co",
    menuItemGroups: userGroups,
  },
};

export const AdminUser: Story = {
  args: {
    userName: "abhimanyu",
    userEmail: "abhi@agpt.co",
    menuItemGroups: adminGroups,
  },
};

export const Loading: Story = {
  args: {
    isLoading: true,
    menuItemGroups: userGroups,
  },
};

export const LongValues: Story = {
  args: {
    userName: "a-really-long-username-that-should-truncate",
    userEmail: "a-very-long-email-address@example-company-domain.com",
    menuItemGroups: userGroups,
  },
};
