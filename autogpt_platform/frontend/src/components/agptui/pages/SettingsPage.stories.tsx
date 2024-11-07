import type { Meta, StoryObj } from "@storybook/react";
import { SettingsPage } from "./SettingsPage";
import { IconType } from "../../ui/icons";

const meta: Meta<typeof SettingsPage> = {
  title: "AGPT UI/Settings/Settings Page",
  component: SettingsPage,
  parameters: {
    layout: {
      center: true,
      fullscreen: true,
      padding: 0,
    },
  },
  tags: ["autodocs"],
  argTypes: {
    isLoggedIn: { control: "boolean" },
    userName: { control: "text" },
    userEmail: { control: "text" },
    navLinks: { control: "object" },
    activeLink: { control: "text" },
    menuItemGroups: { control: "object" },
    sidebarLinkGroups: { control: "object" },
  },
};

export default meta;
type Story = StoryObj<typeof SettingsPage>;

const mockNavLinks = [
  { name: "Marketplace", href: "/" },
  { name: "Library", href: "/library" },
  { name: "Build", href: "/build" },
];

const mockMenuItemGroups = [
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

const mockSidebarLinkGroups = [
  {
    links: [
      { text: "Agent dashboard", href: "/dashboard" },
      { text: "Integrations", href: "/integrations" },
      { text: "Profile", href: "/profile" },
      { text: "Settings", href: "/settings" },
    ],
  },
];

export const Default: Story = {
  args: {
    isLoggedIn: true,
    userName: "John Doe",
    userEmail: "john@example.com",
    navLinks: mockNavLinks,
    activeLink: "/settings",
    menuItemGroups: mockMenuItemGroups,
    sidebarLinkGroups: mockSidebarLinkGroups,
  },
};

export const LoggedOut: Story = {
  args: {
    ...Default.args,
    isLoggedIn: false,
    userName: "",
    userEmail: "",
  },
};
