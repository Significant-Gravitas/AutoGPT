import type { Meta, StoryObj } from "@storybook/react";
import { CreatorDashboardPage } from "./CreatorDashboardPage";
import { IconType } from "../../ui/icons";

const meta: Meta<typeof CreatorDashboardPage> = {
  title: "AGPT UI/Agent Store/Creator Dashboard Page",
  component: CreatorDashboardPage,
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
  },
};

export default meta;
type Story = StoryObj<typeof CreatorDashboardPage>;

const sampleNavLinks = [
  { name: "Home", href: "/" },
  { name: "Marketplace", href: "/marketplace" },
  { name: "Dashboard", href: "/dashboard" },
];

const sampleMenuItemGroups = [
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

const sampleSidebarLinkGroups = [
  {
    links: [{ text: "Integrations", href: "/integrations" }],
  },
  {
    links: [
      { text: "Profile", href: "/profile" },
      { text: "Settings", href: "/settings" },
    ],
  },
];

const sampleAgents = [
  {
    agentName: "Super Coder",
    description: "An AI agent that writes clean, efficient code",
    imageSrc:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/11/47/114784105a9b180e08e117cbf2612e5b.jpg",
    dateSubmitted: "2023-05-15",
    status: "Active",
    runs: 1500,
    rating: 4.8,
    onEdit: () => console.log("Edit Super Coder"),
  },
  {
    agentName: "Data Analyzer",
    description: "Processes and analyzes large datasets with ease",
    imageSrc:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/40/f7/40f7bc97c952f8df0f9c88d29defe8d4.jpg",
    dateSubmitted: "2023-05-10",
    status: "Active",
    runs: 1200,
    rating: 4.5,
    onEdit: () => console.log("Edit Data Analyzer"),
  },
];

export const Default: Story = {
  args: {
    isLoggedIn: true,
    userName: "John Doe",
    userEmail: "john.doe@example.com",
    navLinks: sampleNavLinks,
    activeLink: "Dashboard",
    menuItemGroups: sampleMenuItemGroups,
    sidebarLinkGroups: sampleSidebarLinkGroups,
    agents: sampleAgents,
  },
};

export const NoAgents: Story = {
  args: {
    ...Default.args,
    agents: [],
  },
};

export const ManyAgents: Story = {
  args: {
    ...Default.args,
    agents: Array(10)
      .fill(sampleAgents[0])
      .map((agent, index) => ({
        ...agent,
        agentName: `Agent ${index + 1}`,
        onEdit: () => console.log(`Edit Agent ${index + 1}`),
      })),
  },
};
