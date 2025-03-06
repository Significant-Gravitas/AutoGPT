import type { Meta, StoryObj } from "@storybook/react";
import { Navbar } from "./Navbar";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../ui/icons";
import { ProfileDetails } from "@/lib/autogpt-server-api/types";
// You can't import this here, jest is not available in storybook and will crash it
// import { jest } from "@jest/globals";

// Mock the API responses
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

// Mock the API module
// jest.mock("@/lib/autogpt-server-api", () => {
//   return function () {
//     return {
//       getStoreProfile: () => Promise.resolve(mockProfileData),
//       getUserCredit: () => Promise.resolve(mockCreditData),
//     };
//   };
// });

const meta = {
  title: "AGPT UI/Navbar",
  component: Navbar,
  parameters: {
    layout: "fullscreen",
  },
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
  { name: "Marketplace", href: "/marketplace" },
  { name: "Library", href: "/library" },
  { name: "Build", href: "/builder" },
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

export const WithActiveLink: Story = {
  args: {
    ...Default.args,
    // activeLink: "/library",
  },
};

export const LongUserName: Story = {
  args: {
    ...Default.args,
    // avatarSrc: "https://avatars.githubusercontent.com/u/987654321?v=4",
  },
};

export const NoAvatar: Story = {
  args: {
    ...Default.args,
    // avatarSrc: undefined,
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
    // isLoggedIn: false,
    // avatarSrc: undefined,
  },
};

export const WithCredits: Story = {
  args: {
    ...Default.args,
  },
};

export const WithLargeCredits: Story = {
  args: {
    ...Default.args,
  },
};

export const WithZeroCredits: Story = {
  args: {
    ...Default.args,
  },
};
