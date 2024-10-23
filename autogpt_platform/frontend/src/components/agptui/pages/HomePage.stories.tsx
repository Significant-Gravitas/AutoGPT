import type { Meta, StoryObj } from "@storybook/react";
import { Page } from "./HomePage";
import { userEvent, within } from "@storybook/test";
import { IconType } from "../../ui/icons";

const meta = {
  title: "AGPT UI/Agent Store/Home Page",
  component: Page,
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
    navLinks: { control: "object" },
    activeLink: { control: "text" },
    featuredAgents: { control: "object" },
    topAgents: { control: "object" },
    featuredCreators: { control: "object" },
    menuItemGroups: { control: "object" },
  },
} satisfies Meta<typeof Page>;

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

const mockFeaturedAgents = [
  {
    agentName: "Super SEO Optimizer",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/cc/11/cc1172271dcf723a34f488a3344e82b2.jpg",
    creatorName: "AI Labs",
    description:
      "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
    runs: 100000,
    rating: 4.9,
  },
  {
    agentName: "Content Wizard",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    creatorName: "WriteRight Inc.",
    description:
      "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
    runs: 75000,
    rating: 4.7,
  },
];

const mockTopAgents = [
  {
    agentName: "Data Analyzer Pro",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/07/78/0778415062f8dff56a046a7eca44567c.jpg",
    avatarSrc: "https://github.com/shadcn.png",
    description:
      "Powerful tool for analyzing large datasets and generating insights.",
    runs: 50000,
    rating: 5,
  },
  {
    agentName: "Image Recognition Master",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/59/b9/59b9415d4044f48f9b9e318c4c5a7984.jpg",
    avatarSrc: "https://example.com/avatar2.jpg",
    description:
      "Accurately identify and classify objects in images using state-of-the-art machine learning algorithms.",
    runs: 60000,
    rating: 4.6,
  },
  {
    agentName: "Natural Language Processor",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/11/47/114784105a9b180e08e117cbf2612e5b.jpg",
    avatarSrc: "https://example.com/avatar3.jpg",
    description:
      "Analyze and understand human language with advanced NLP techniques.",
    runs: 80000,
    rating: 4.7,
  },
  {
    agentName: "Sentiment Analyzer",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/07/7c/077c975fdb404609532364dd5a2e2354.jpg",
    avatarSrc: "https://example.com/avatar4.jpg",
    description:
      "Determine the emotional tone of text data for customer feedback analysis.",
    runs: 45000,
    rating: 4.5,
  },
  {
    agentName: "Chatbot Builder",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/63/96/6396cb4e44284b80ead9300d9b47c544.jpg",
    avatarSrc: "https://example.com/avatar5.jpg",
    description:
      "Create intelligent chatbots for customer service and engagement.",
    runs: 55000,
    rating: 4.4,
  },
  {
    agentName: "Predictive Analytics Tool",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/53/b2/53b2bc7d7900f0e1e60bf64ebf38032d.jpg",
    avatarSrc: "https://example.com/avatar6.jpg",
    description:
      "Forecast future trends and outcomes based on historical data.",
    runs: 40000,
    rating: 4.0,
  },
  {
    agentName: "Text-to-Speech Converter",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    avatarSrc: "https://example.com/avatar7.jpg",
    description:
      "Convert written text into natural-sounding speech in multiple languages.",
    runs: 35000,
    rating: 3.0,
  },
  {
    agentName: "Code Generator AI",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/40/f7/40f7bc97c952f8df0f9c88d29defe8d4.jpg",
    avatarSrc: "https://example.com/avatar8.jpg",
    description:
      "Automatically generate code snippets and boilerplate for various programming languages.",
    runs: 70000,
    rating: 2.5,
  },
  {
    agentName: "Virtual Assistant Creator",
    agentImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/14/9e/149ebb9014aa8c0097e72ed89845af0e.jpg",
    avatarSrc: "https://example.com/avatar9.jpg",
    description:
      "Build customized virtual assistants for various business needs and personal use.",
    runs: 65000,
    rating: 1.4,
  },
];

const mockFeaturedCreators = [
  {
    creatorName: "AI Labs",
    creatorImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/53/b2/53b2bc7d7900f0e1e60bf64ebf38032d.jpg",
    bio: "Pioneering AI solutions for everyday problems",
    agentsUploaded: 25,
    avatarSrc: "https://github.com/shadcn.png",
  },
  {
    creatorName: "WriteRight Inc.",
    creatorImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/40/f7/40f7bc97c952f8df0f9c88d29defe8d4.jpg",
    bio: "Empowering content creators with AI-driven tools",
    agentsUploaded: 18,
    avatarSrc: "https://example.com/writeright-avatar.jpg",
  },
  {
    creatorName: "DataMasters",
    creatorImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/14/9e/149ebb9014aa8c0097e72ed89845af0e.jpg",
    bio: "Transforming data into actionable insights",
    agentsUploaded: 30,
    avatarSrc: "https://example.com/datamasters-avatar.jpg",
  },
  {
    creatorName: "AInovators",
    creatorImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/53/b2/53b2bc7d7900f0e1e60bf64ebf38032d.jpg",
    bio: "Pushing the boundaries of artificial intelligence",
    agentsUploaded: 22,
    avatarSrc: "https://github.com/shadcn.png",
  },
  {
    creatorName: "CodeCrafters",
    creatorImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/40/f7/40f7bc97c952f8df0f9c88d29defe8d4.jpg",
    bio: "Building intelligent coding assistants for developers",
    agentsUploaded: 28,
    avatarSrc: "https://example.com/codecrafters-avatar.jpg",
  },
  {
    creatorName: "EcoTech Solutions",
    creatorImage:
      "https://ddz4ak4pa3d19.cloudfront.net/cache/14/9e/149ebb9014aa8c0097e72ed89845af0e.jpg",
    bio: "Developing AI-powered tools for environmental sustainability",
    agentsUploaded: 20,
    avatarSrc: "https://example.com/ecotech-avatar.jpg",
  },
];

export const Default: Story = {
  args: {
    isLoggedIn: true,
    userName: "John Doe",
    userEmail: "john.doe@example.com",
    navLinks: mockNavLinks,
    activeLink: "/",
    featuredAgents: mockFeaturedAgents,
    topAgents: mockTopAgents,
    featuredCreators: mockFeaturedCreators,
    menuItemGroups: mockMenuItemGroups,
  },
};

export const WithInteraction: Story = {
  args: {
    ...Default.args,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Simulate a search
    const searchInput = canvas.getByPlaceholderText(/Search for tasks/i);
    await userEvent.type(searchInput, "SEO optimization");
    await userEvent.keyboard("{Enter}");

    // Click on a featured agent card
    const featuredAgentCard = canvas.getByText("Super SEO Optimizer");
    await userEvent.click(featuredAgentCard);

    // Click on the "Become a Creator" button
    const becomeCreatorButton = canvas.getByText("Become a Creator");
    await userEvent.click(becomeCreatorButton);
  },
};

export const EmptyState: Story = {
  args: {
    isLoggedIn: false,
    userName: "Jane Smith",
    userEmail: "jane.smith@example.com",
    navLinks: mockNavLinks,
    activeLink: "Marketplace",
    featuredAgents: [],
    topAgents: [],
    featuredCreators: [],
    menuItemGroups: mockMenuItemGroups,
  },
};

export const LongLists: Story = {
  args: {
    ...Default.args,
    featuredAgents: Array(10).fill(mockFeaturedAgents[0]),
    topAgents: Array(20).fill(mockTopAgents[0]),
    featuredCreators: Array(8).fill(mockFeaturedCreators[0]),
  },
};
