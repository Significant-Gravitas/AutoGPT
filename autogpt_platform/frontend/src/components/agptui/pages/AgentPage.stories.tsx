import type { Meta, StoryObj } from "@storybook/react";
import { AgentPage } from "./AgentPage";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../../ui/icons";

const meta = {
  title: "AGPT UI/Agent Store/Agent Page",
  component: AgentPage,
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
    agentInfo: { control: "object" },
    agentImages: { control: "object" },
    otherAgentsByCreator: { control: "object" },
    similarAgents: { control: "object" },
  },
} satisfies Meta<typeof AgentPage>;

export default meta;
type Story = StoryObj<typeof meta>;

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

const mockAgentInfo = {
  name: "Super SEO Optimizer",
  creator: "AI Labs",
  description:
    "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
  rating: 4.9,
  runs: 100000,
  categories: ["SEO", "Marketing", "Content"],
  lastUpdated: "2023-05-15",
  version: "2.1.0",
};

const mockAgentImages = [
  "https://ddz4ak4pa3d19.cloudfront.net/cache/cc/11/cc1172271dcf723a34f488a3344e82b2.jpg",
  "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
];

const mockOtherAgentsByCreator = [
  {
    agentName: "Content Wizard",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    description:
      "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
    runs: 75000,
    rating: 4.7,
    avatarSrc: "https://example.com/avatar1.jpg",
  },
  {
    agentName: "Data Analyzer Pro",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/07/78/0778415062f8dff56a046a7eca44567c.jpg",
    description:
      "Powerful tool for analyzing large datasets and generating insights.",
    runs: 50000,
    rating: 5,
    avatarSrc: "https://github.com/shadcn.png",
  },
  {
    agentName: "AI Copywriter",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/14/9e/149ebb9014aa8c0097e72ed89845af0e.jpg",
    description:
      "AI-powered copywriting assistant for creating compelling marketing copy.",
    runs: 62000,
    rating: 4.8,
    avatarSrc: "https://example.com/avatar4.jpg",
  },
];

const mockSimilarAgents = [
  {
    agentName: "SEO Master",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/59/b9/59b9415d4044f48f9b9e318c4c5a7984.jpg",
    description:
      "Comprehensive SEO tool for website optimization and ranking improvement.",
    runs: 80000,
    rating: 4.8,
    avatarSrc: "https://example.com/avatar2.jpg",
  },
  {
    agentName: "Keyword Genius",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/11/47/114784105a9b180e08e117cbf2612e5b.jpg",
    description:
      "Advanced keyword research and analysis tool for SEO professionals.",
    runs: 60000,
    rating: 4.6,
    avatarSrc: "https://example.com/avatar3.jpg",
  },
  {
    agentName: "Backlink Builder",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/14/9e/149ebb9014aa8c0097e72ed89845af0e.jpg",
    description:
      "Automated tool for building high-quality backlinks to improve SEO performance.",
    runs: 55000,
    rating: 4.7,
    avatarSrc: "https://example.com/avatar5.jpg",
  },
];

export const Default: Story = {
  args: {
    isLoggedIn: true,
    userName: "John Doe",
    userEmail: "john.doe@example.com",
    navLinks: mockNavLinks,
    activeLink: "/marketplace",
    menuItemGroups: mockMenuItemGroups,
    agentInfo: mockAgentInfo,
    agentImages: mockAgentImages,
    otherAgentsByCreator: mockOtherAgentsByCreator,
    similarAgents: mockSimilarAgents,
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Click on the "Run Agent" button
    const runAgentButton = canvas.getByText("Run Agent");
    await userEvent.click(runAgentButton);

    // Click on an "Other agents by creator" card
    const otherAgentCard = canvas.getByText("Content Wizard");
    await userEvent.click(otherAgentCard);

    // Click on the "Become a Creator" button
    const becomeCreatorButton = canvas.getByText("Become a Creator");
    await userEvent.click(becomeCreatorButton);
  },
};

export const LongLists: Story = {
  args: {
    ...Default.args,
    otherAgentsByCreator: Array(10).fill(mockOtherAgentsByCreator[0]),
    similarAgents: Array(10).fill(mockSimilarAgents[0]),
  },
};
