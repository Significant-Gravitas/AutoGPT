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
        text: "Profile",
        href: "/settings/profile",
      },
      {
        icon: IconType.Settings,
        text: "Settings",
        href: "/settings/account",
      },
      {
        icon: IconType.Billing,
        text: "Billing",
        href: "/settings/billing",
      },
      {
        icon: IconType.LayoutDashboard,
        text: "Creator Dashboard",
        href: "/settings/creator-dashboard",
      },
      {
        icon: IconType.Help,
        text: "Help & Docs",
        href: "https://agpt.co/docs",
        external: true,
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
  userGroups[0],
  {
    items: [
      {
        icon: IconType.Sliders,
        text: "Admin",
        href: "/admin/marketplace",
      },
    ],
  },
  userGroups[1],
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
