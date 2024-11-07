import type { Meta, StoryObj } from "@storybook/react";
import ProfilePage from "./ProfilePage";
import { AVAILABLE_CATEGORIES } from "../ProfileInfoForm";
import {
  IconPersonFill,
  IconSettings,
  IconLogOut,
  IconType,
} from "@/components/ui/icons";

const meta: Meta<typeof ProfilePage> = {
  title: "AGPT UI/Profile/Profile Page",
  component: ProfilePage,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
};

export default meta;
type Story = StoryObj<typeof ProfilePage>;

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

export const Empty: Story = {
  args: {
    userName: "",
    userEmail: "",
    displayName: "",
    handle: "",
    bio: "",
    links: [],
    categories: [],
    credits: 0,
    menuItemGroups: defaultMenuItemGroups,
    isLoggedIn: true,
    avatarSrc: undefined,
  },
};

export const Filled: Story = {
  args: {
    userName: "Alex Chen",
    userEmail: "alex.chen@aigpt.dev",
    displayName: "Alex Chen",
    handle: "@aidev_alex",
    bio: "AI Developer & Tech Enthusiast | Building intelligent solutions with AutoGPT | Passionate about making AI accessible and ethical | Contributing to open-source AI projects and sharing knowledge about autonomous agents.",
    links: [
      { id: 1, url: "alexchen.dev" },
      { id: 2, url: "github.com/alexchen-ai" },
      { id: 3, url: "twitter.com/aidev_alex" },
      { id: 4, url: "medium.com/@alexchen-ai" },
      { id: 5, url: "linkedin.com/in/alexchen-ai" },
    ],
    categories: [
      { id: 1, name: "AI Development" },
      { id: 2, name: "Tech" },
      { id: 3, name: "Open Source" },
      { id: 4, name: "Education" },
      { id: 5, name: "Content creation" },
    ].filter((cat) => AVAILABLE_CATEGORIES.includes(cat.name as any)),
    credits: 1500,
    menuItemGroups: defaultMenuItemGroups,
    isLoggedIn: true,
    avatarSrc: "https://github.com/alexchen-ai.png",
  },
};

export const LoggedOut: Story = {
  args: {
    ...Empty.args,
    isLoggedIn: false,
  },
};
