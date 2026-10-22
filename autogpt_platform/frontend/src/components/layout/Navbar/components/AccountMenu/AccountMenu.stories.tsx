import { IconType } from "@/components/__legacy__/ui/icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { AccountMenu } from "./AccountMenu";
import { fixedCopy } from "@/lib/replay-privacy";

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
        text: fixedCopy("Profile"),
        href: "/settings/profile",
      },
      {
        icon: IconType.Settings,
        text: fixedCopy("Settings"),
        href: "/settings/account",
      },
      {
        icon: IconType.Billing,
        text: fixedCopy("Billing"),
        href: "/settings/billing",
      },
      {
        icon: IconType.LayoutDashboard,
        text: fixedCopy("Creator Dashboard"),
        href: "/settings/creator-dashboard",
      },
      {
        icon: IconType.Help,
        text: fixedCopy("Help & Docs"),
        href: "https://agpt.co/docs",
        external: true,
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.LogOut,
        text: fixedCopy("Log out"),
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
        text: fixedCopy("Admin"),
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
