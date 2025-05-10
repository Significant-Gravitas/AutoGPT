import type { Meta, StoryObj } from "@storybook/react";
import { Navbar } from "./Navbar";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../ui/icons";
import { ProfileDetails } from "@/lib/autogpt-server-api/types";

const mockProfileData: ProfileDetails = {
  name: "John Doe",
  username: "johndoe",
  description: "",
  links: [],
  avatar_url: "https://avatars.githubusercontent.com/u/123456789?v=4",
};

const mockCreditData = {
  credits: 1500,
};

const meta = {
  title: "Agpt Custom UI/general/Navbar",
  component: Navbar,
  decorators: [
    (Story) => (
      <div className="flex h-screen w-full justify-center">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    // isLoggedIn: { control: "boolean" },
    // avatarSrc: { control: "text" },
    links: { control: "object" },
    // activeLink: { control: "text" },
    menuItemGroups: { control: "object" },
    // params: { control: { type: "object", defaultValue: { lang: "en" } } },
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
  {
    name: "Home",
    href: "/library",
  },
  {
    name: "Marketplace",
    href: "/marketplace",
  },

  {
    name: "Build",
    href: "/build",
  },
];

export const Default: Story = {
  args: {
    // params: { lang: "en" },
    // isLoggedIn: true,
    links: defaultLinks,
    // activeLink: "/marketplace",
    // avatarSrc: mockProfileData.avatar_url,
    menuItemGroups: defaultMenuItemGroups,
  },
};
