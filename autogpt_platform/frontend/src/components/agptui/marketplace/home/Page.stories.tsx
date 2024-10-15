import type { Meta, StoryObj } from "@storybook/react";
import { Page } from "./Page";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPTUI/Marketplace/Home/Page",
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
    userName: { control: "text" },
    navLinks: { control: "object" },
    activeLink: { control: "text" },
    featuredAgents: { control: "object" },
    topAgents: { control: "object" },
    featuredCreators: { control: "object" },
  },
} satisfies Meta<typeof Page>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockNavLinks = [
  { name: "Marketplace", href: "/" },
  { name: "Library", href: "/library" },
  { name: "Build", href: "/build" },
];

const mockFeaturedAgents = [
  {
    agentName: "Super SEO Optimizer",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
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
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    avatarSrc: "https://example.com/avatar1.jpg",
    description:
      "Powerful tool for analyzing large datasets and generating insights.",
    runs: 50000,
    rating: 5,
  },
  {
    agentName: "Image Recognition Master",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    avatarSrc: "https://example.com/avatar2.jpg",
    description:
      "Accurately identify and classify objects in images using state-of-the-art machine learning algorithms.",
    runs: 60000,
    rating: 4.6,
  },
  {
    agentName: "Natural Language Processor",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    avatarSrc: "https://example.com/avatar3.jpg",
    description:
      "Analyze and understand human language with advanced NLP techniques.",
    runs: 80000,
    rating: 4.7,
  },
  {
    agentName: "Sentiment Analyzer",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    avatarSrc: "https://example.com/avatar4.jpg",
    description:
      "Determine the emotional tone of text data for customer feedback analysis.",
    runs: 45000,
    rating: 4.5,
  },
  {
    agentName: "Chatbot Builder",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    avatarSrc: "https://example.com/avatar5.jpg",
    description:
      "Create intelligent chatbots for customer service and engagement.",
    runs: 55000,
    rating: 4.4,
  },
  {
    agentName: "Predictive Analytics Tool",
    agentImage:
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
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
      "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
    avatarSrc: "https://example.com/avatar8.jpg",
    description:
      "Automatically generate code snippets and boilerplate for various programming languages.",
    runs: 70000,
    rating: 2.5,
  },
  {
    agentName: "Virtual Assistant Creator",
    agentImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
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
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Pioneering AI solutions for everyday problems",
    agentsUploaded: 25,
    avatarSrc: "https://example.com/ailabs-avatar.jpg",
  },
  {
    creatorName: "WriteRight Inc.",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Empowering content creators with AI-driven tools",
    agentsUploaded: 18,
    avatarSrc: "https://example.com/writeright-avatar.jpg",
  },
  {
    creatorName: "DataMasters",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Transforming data into actionable insights",
    agentsUploaded: 30,
    avatarSrc: "https://example.com/datamasters-avatar.jpg",
  },
  {
    creatorName: "AInovators",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Pushing the boundaries of artificial intelligence",
    agentsUploaded: 22,
    avatarSrc: "https://example.com/ainovators-avatar.jpg",
  },
  {
    creatorName: "CodeCrafters",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Building intelligent coding assistants for developers",
    agentsUploaded: 28,
    avatarSrc: "https://example.com/codecrafters-avatar.jpg",
  },
  {
    creatorName: "EcoTech Solutions",
    creatorImage:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    bio: "Developing AI-powered tools for environmental sustainability",
    agentsUploaded: 20,
    avatarSrc: "https://example.com/ecotech-avatar.jpg",
  },
];

export const Default: Story = {
  args: {
    userName: "John Doe",
    navLinks: mockNavLinks,
    activeLink: "/",
    featuredAgents: mockFeaturedAgents,
    topAgents: mockTopAgents,
    featuredCreators: mockFeaturedCreators,
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
    userName: "Jane Smith",
    navLinks: mockNavLinks,
    activeLink: "Marketplace",
    featuredAgents: [],
    topAgents: [],
    featuredCreators: [],
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
