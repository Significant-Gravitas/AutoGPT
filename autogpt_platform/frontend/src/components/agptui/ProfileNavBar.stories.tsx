import type { Meta, StoryObj } from "@storybook/react";
import { ProfileNavBar } from "./ProfileNavBar";
import { IconType } from "@/components/ui/icons";

const meta: Meta<typeof ProfileNavBar> = {
  title: "AGPT UI/Profile/Profile NavBar",
  component: ProfileNavBar,
  parameters: {
    layout: "fullscreen",
  },
  tags: ['autodocs'],
  argTypes: {
    userName: { 
      control: "text",
      description: "The display name of the user shown in the navbar" 
    },
    userEmail: { 
      control: "text",
      description: "The email address shown in the profile popout menu" 
    },
    credits: { 
      control: "number",
      description: "Number of credits the user has" 
    },
    avatarSrc: { 
      control: "text",
      description: "URL of the user's profile picture" 
    },
    onRefreshCredits: { 
      description: "Callback function when refresh credits button is clicked" 
    },
    menuItemGroups: { 
      control: "object",
      description: "Array of menu item groups for the profile popout menu" 
    },
  },
};

export default meta;
type Story = StoryObj<typeof ProfileNavBar>;

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

export const Default: Story = {
  args: {
    userName: "John Doe",
    userEmail: "john@example.com",
    credits: 1500,
    menuItemGroups: defaultMenuItemGroups,
    onRefreshCredits: () => console.log("Refresh credits clicked"),
  },
};

export const LowCredits: Story = {
  args: {
    ...Default.args,
    credits: 50,
  },
};