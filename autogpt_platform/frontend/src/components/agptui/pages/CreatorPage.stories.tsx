import type { Meta, StoryObj } from "@storybook/react";
import { CreatorPage } from "./CreatorPage";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../../ui/icons";

const meta = {
  title: "AGPT UI/Agent Store/Creator Page",
  component: CreatorPage,
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
    creatorInfo: { control: "object" },
    creatorAgents: { control: "object" },
  },
} satisfies Meta<typeof CreatorPage>;

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

const mockCreatorInfo = {
  name: "AI Labs",
  username: "ailabs",
  description:
    "Our agents are designed to bring happiness and positive vibes to your daily routine. Each template helps you create and live the life of your dreams while guiding you to become your best every day",
  avgRating: 4.8,
  agentCount: 15,
  topCategories: ["SEO", "Marketing", "Data Analysis"],
  avatarSrc: "https://example.com/avatar1.jpg",
  otherLinks: {
    website: "https://ailabs.com",
    github: "https://github.com/ailabs",
    linkedin: "https://linkedin.com/company/ailabs",
  },
};

const mockCreatorAgents = [
  {
    agentName: "Super SEO Optimizer",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/cc/11/cc1172271dcf723a34f488a3344e82b2.jpg",
    description:
      "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
    runs: 100000,
    rating: 4.9,
    avatarSrc: "https://example.com/avatar1.jpg",
  },
  {
    agentName: "Content Wizard",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    description:
      "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
    runs: 75000,
    rating: 4.7,
    avatarSrc: "https://example.com/avatar2.jpg",
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
    agentName: "Image Recognition Master",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/59/b9/59b9415d4044f48f9b9e318c4c5a7984.jpg",
    description:
      "Accurately identify and classify objects in images using state-of-the-art machine learning algorithms.",
    runs: 60000,
    rating: 4.6,
    avatarSrc: "https://example.com/avatar2.jpg",
  },
  {
    agentName: "Natural Language Processor",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/11/47/114784105a9b180e08e117cbf2612e5b.jpg",
    description:
      "Analyze and understand human language with advanced NLP techniques.",
    runs: 80000,
    rating: 4.7,
    avatarSrc: "https://example.com/avatar3.jpg",
  },
  {
    agentName: "Sentiment Analyzer",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/07/7c/077c975fdb404609532364dd5a2e2354.jpg",
    description:
      "Determine the emotional tone of text data for customer feedback analysis.",
    runs: 45000,
    rating: 4.5,
    avatarSrc: "https://example.com/avatar4.jpg",
  },
  {
    agentName: "Chatbot Builder",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/63/96/6396cb4e44284b80ead9300d9b47c544.jpg",
    description:
      "Create intelligent chatbots for customer service and engagement.",
    runs: 55000,
    rating: 4.4,
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
    creatorInfo: mockCreatorInfo,
    creatorAgents: mockCreatorAgents,
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Click on an agent card
    const agentCard = canvas.getByText("Super SEO Optimizer");
    await userEvent.click(agentCard);

    // Click on a creator's social link
    const githubLink = canvas.getByText("GitHub");
    await userEvent.click(githubLink);
  },
};

export const ManyAgents: Story = {
  args: {
    ...Default.args,
    creatorAgents: Array(10).fill(mockCreatorAgents[0]),
  },
};

export const NewCreator: Story = {
  args: {
    ...Default.args,
    creatorInfo: {
      ...mockCreatorInfo,
      avgRating: 0,
      agentCount: 1,
      topCategories: ["AI"],
    },
    creatorAgents: [mockCreatorAgents[0]],
  },
};
