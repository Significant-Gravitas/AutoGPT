import type { Meta, StoryObj } from "@storybook/react";
import { Agent, AgentsSection } from "./AgentsSection";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "Agpt Custom UI/marketing/Agents Section",
  component: AgentsSection,

  decorators: [
    (Story) => (
      <div className="flex items-center justify-center py-4 md:p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    sectionTitle: { control: "text" },
    agents: { control: "object" },
    hideAvatars: { control: "boolean" },
    margin: { control: "text" },
    className: { control: "text" },
  },
} satisfies Meta<typeof AgentsSection>;

export default meta;
type Story = StoryObj<typeof meta>;

const defaultAgentImage = "default_agent_image.jpg";
const defaultAvatarImage = "default_avatar.png";

const mockTopAgents = [
  {
    agent_name: "SEO Optimizer Pro",
    agent_image: defaultAgentImage,
    description:
      "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
    runs: 50000,
    rating: 4.7,
    creator_avatar: defaultAvatarImage,
    slug: "seo-optimizer-pro",
    creator: "John Doe",
    sub_heading: "SEO Expert",
  },
  {
    agent_name: "Content Writer AI",
    agent_image: defaultAgentImage,
    description:
      "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
    runs: 75000,
    rating: 4.5,
    creator_avatar: defaultAvatarImage,
    slug: "content-writer-ai",
    creator: "Jane Doe",
    sub_heading: "Content Writer",
  },
  {
    agent_name: "Data Analyzer Lite",
    agent_image: defaultAgentImage,
    description: "A basic tool for analyzing small to medium-sized datasets.",
    runs: 10000,
    rating: 3.8,
    creator_avatar: defaultAvatarImage,
    slug: "data-analyzer-lite",
    creator: "John Doe",
    sub_heading: "Data Analyst",
  },
] satisfies Agent[];

export const Default: Story = {
  args: {
    sectionTitle: "Top Agents",
    agents: mockTopAgents,
  },
};

export const SingleAgent: Story = {
  args: {
    sectionTitle: "Featured Agent",
    agents: [mockTopAgents[0]],
  },
};

export const NoAgents: Story = {
  args: {
    sectionTitle: "Recommended Agents",
    agents: [],
  },
};

export const WithInteraction: Story = {
  args: {
    sectionTitle: "Popular Agents",
    agents: mockTopAgents,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    // Using the proper test ID that's defined in the StoreCard component
    const firstCard = canvas.getAllByTestId("store-card")[0];
    await userEvent.hover(firstCard);
    await new Promise((resolve) => setTimeout(resolve, 300));
    await userEvent.click(firstCard);
  },
};

export const MultiRowAgents: Story = {
  args: {
    sectionTitle: "All Agents",
    agents: [
      ...mockTopAgents,
      {
        agent_name: "Image Recognition AI",
        agent_image: defaultAgentImage,
        description:
          "Accurately identify and classify objects in images using state-of-the-art machine learning algorithms.",
        runs: 60000,
        rating: 4.6,
        creator_avatar: defaultAvatarImage,
        slug: "image-recognition-ai",
        creator: "Alex Smith",
        sub_heading: "Image Recognition",
      },
      {
        agent_name: "Natural Language Processor",
        agent_image: defaultAgentImage,
        description:
          "Analyze and understand human language with advanced NLP techniques.",
        runs: 80000,
        rating: 4.8,
        creator_avatar: defaultAvatarImage,
        slug: "natural-language-processor",
        creator: "Maria Garcia",
        sub_heading: "Natural Language Processing",
      },
      {
        agent_name: "Sentiment Analyzer",
        agent_image: defaultAgentImage,
        description:
          "Determine the emotional tone of text data for customer feedback analysis.",
        runs: 45000,
        rating: 4.3,
        creator_avatar: defaultAvatarImage,
        slug: "sentiment-analyzer",
        creator: "Robert Johnson",
        sub_heading: "Sentiment Analysis",
      },
      {
        agent_name: "Chatbot Builder",
        agent_image: defaultAgentImage,
        description:
          "Create intelligent chatbots for customer service and engagement.",
        runs: 55000,
        rating: 4.4,
        creator_avatar: defaultAvatarImage,
        slug: "chatbot-builder",
        creator: "Emma Wilson",
        sub_heading: "Chatbot Developer",
      },
      {
        agent_name: "Predictive Analytics Tool",
        agent_image: defaultAgentImage,
        description:
          "Forecast future trends and outcomes based on historical data.",
        runs: 40000,
        rating: 4.2,
        creator_avatar: defaultAvatarImage,
        slug: "predictive-analytics-tool",
        creator: "David Lee",
        sub_heading: "Predictive Analytics",
      },
      {
        agent_name: "Text-to-Speech Converter",
        agent_image: defaultAgentImage,
        description:
          "Convert written text into natural-sounding speech in multiple languages.",
        runs: 35000,
        rating: 4.1,
        creator_avatar: defaultAvatarImage,
        slug: "text-to-speech-converter",
        creator: "Sarah Brown",
        sub_heading: "Text-to-Speech",
      },
    ],
  },
};

export const HiddenAvatars: Story = {
  args: {
    ...Default.args,
    hideAvatars: true,
    sectionTitle: "Agents with Hidden Avatars",
  },
};

export const LongAgentNames: Story = {
  args: {
    sectionTitle: "Test Cases",
    agents: [
      {
        agent_name:
          "Universal Language Translator Pro with Advanced Neural Network Technology",
        agent_image: defaultAgentImage,
        description:
          "Breaking language barriers with cutting-edge AI translation technology for global communication.",
        runs: 120000,
        rating: 4.9,
        creator_avatar: defaultAvatarImage,
        slug: "universal-language-translator",
        creator: "Global Linguistics Technologies",
        sub_heading: "Translation Services",
      },
      ...mockTopAgents.slice(0, 2),
    ],
  },
};

export const LongDescriptions: Story = {
  args: {
    sectionTitle: "Test Cases",
    agents: [
      {
        agent_name: "AI Writing Assistant",
        agent_image: defaultAgentImage,
        description:
          "Enhance your writing with our advanced AI-powered assistant. It offers real-time suggestions for grammar, style, and tone, helps with research and fact-checking, and provides vocabulary enhancements for more engaging content. Perfect for content creators, marketers, and writers of all levels.",
        runs: 75000,
        rating: 4.7,
        creator_avatar: defaultAvatarImage,
        slug: "ai-writing-assistant",
        creator: "ContentGenius",
        sub_heading: "Writing Enhancement",
      },
      ...mockTopAgents.slice(0, 2),
    ],
  },
};

export const ZeroValues: Story = {
  args: {
    sectionTitle: "New Agents",
    agents: [
      {
        agent_name: "New Project Template",
        agent_image: defaultAgentImage,
        description:
          "A basic template for new projects with no user data or ratings yet.",
        runs: 0,
        rating: 0,
        creator_avatar: defaultAvatarImage,
        slug: "new-project-template",
        creator: "Template Systems",
        sub_heading: "Project Templates",
      },
      ...mockTopAgents.slice(0, 2),
    ],
  },
};

export const MobileView: Story = {
  args: {
    sectionTitle: "Mobile Optimized Agents",
    agents: mockTopAgents,
  },
  parameters: {
    viewport: {
      defaultViewport: "mobile2",
    },
  },
};
