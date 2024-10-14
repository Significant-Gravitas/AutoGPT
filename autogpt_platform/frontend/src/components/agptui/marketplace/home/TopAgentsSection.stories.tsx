import type { Meta, StoryObj } from "@storybook/react";
import { TopAgentsSection } from "./TopAgentsSection";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "AGPTUI/Marketplace/Home/TopAgentsSection",
  component: TopAgentsSection,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    topAgents: { control: "object" },
    onCardClick: { action: "clicked" },
  },
} satisfies Meta<typeof TopAgentsSection>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockTopAgents = [
  {
    agentName: "SEO Optimizer Pro",
    description:
      "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
    runs: 50000,
    rating: 4.7,
  },
  {
    agentName: "Content Writer AI",
    description:
      "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
    runs: 75000,
    rating: 4.5,
  },
  {
    agentName: "Data Analyzer Lite",
    description: "A basic tool for analyzing small to medium-sized datasets.",
    runs: 10000,
    rating: 3.8,
  },
];

export const Default: Story = {
  args: {
    topAgents: mockTopAgents,
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
};

export const SingleAgent: Story = {
  args: {
    topAgents: [mockTopAgents[0]],
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
};

export const NoAgents: Story = {
  args: {
    topAgents: [],
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
};

export const WithInteraction: Story = {
  args: {
    topAgents: mockTopAgents,
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const firstCard = canvas.getAllByRole("button")[0];
    await userEvent.click(firstCard);
  },
};

export const MultiRowAgents: Story = {
  args: {
    topAgents: [
      ...mockTopAgents,
      {
        agentName: "Image Recognition AI",
        description:
          "Accurately identify and classify objects in images using state-of-the-art machine learning algorithms.",
        runs: 60000,
        rating: 4.6,
      },
      {
        agentName: "Natural Language Processor",
        description:
          "Analyze and understand human language with advanced NLP techniques.",
        runs: 80000,
        rating: 4.8,
      },
      {
        agentName: "Sentiment Analyzer",
        description:
          "Determine the emotional tone of text data for customer feedback analysis.",
        runs: 45000,
        rating: 4.3,
      },
      {
        agentName: "Chatbot Builder",
        description:
          "Create intelligent chatbots for customer service and engagement.",
        runs: 55000,
        rating: 4.4,
      },
      {
        agentName: "Predictive Analytics Tool",
        description:
          "Forecast future trends and outcomes based on historical data.",
        runs: 40000,
        rating: 4.2,
      },
      {
        agentName: "Text-to-Speech Converter",
        description:
          "Convert written text into natural-sounding speech in multiple languages.",
        runs: 35000,
        rating: 4.1,
      },
    ],
    onCardClick: (agentName: string) => console.log(`Clicked on ${agentName}`),
  },
};
