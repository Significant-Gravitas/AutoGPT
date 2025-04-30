import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedSection } from "./FeaturedSection";
import { userEvent, within, expect } from "@storybook/test";
import { StoreAgent } from "@/lib/autogpt-server-api";

const meta = {
  title: "AGPT UI/Composite/Featured Agents",
  component: FeaturedSection,
  parameters: {
    layout: {
      center: true,
      fullscreen: true,
      padding: 0,
    },
    viewport: {
      defaultViewport: "responsive",
    },
  },
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center py-4 pl-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    featuredAgents: { control: "object" },
  },
} satisfies Meta<typeof FeaturedSection>;

export default meta;
type Story = StoryObj<typeof meta>;

const mockFeaturedAgents = [
  {
    agent_name: "Personalized Morning Coffee Newsletter example of three lines",
    sub_heading:
      "Transform ideas into breathtaking images with this AI-powered Image Generator.",
    creator: "AI Solutions Inc.",
    description:
      "Elevate your web content with this powerful AI Webpage Copy Improver. Designed for marketers, SEO specialists, and web developers, this tool analyses and enhances website copy for maximum impact. Using advanced language models, it optimizes text for better clarity, SEO performance, and increased conversion rates.",
    runs: 50000,
    rating: 4.7,
    agent_image:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creator_avatar:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    slug: "personalized-morning-coffee-newsletter",
  },
  {
    agent_name: "Data Analyzer Lite",
    sub_heading: "Basic data analysis tool",
    creator: "DataTech",
    description:
      "A lightweight data analysis tool for basic data processing needs.",
    runs: 10000,
    rating: 2.8,
    agent_image:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creator_avatar:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    slug: "data-analyzer-lite",
  },
  {
    agent_name: "CodeAssist AI",
    sub_heading: "Your AI coding companion",
    creator: "DevTools Co.",
    description:
      "An intelligent coding assistant that helps developers write better code faster.",
    runs: 1000000,
    rating: 4.9,
    agent_image:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creator_avatar:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    slug: "codeassist-ai",
  },
  {
    agent_name: "MultiTasker",
    sub_heading: "All-in-one productivity suite",
    creator: "Productivity Plus",
    description:
      "A comprehensive productivity suite that combines task management, note-taking, and project planning into one seamless interface. Features include smart task prioritization, automated scheduling, and AI-powered insights to help you work more efficiently.",
    runs: 75000,
    rating: 4.5,
    agent_image:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creator_avatar:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    slug: "multitasker",
  },
  {
    agent_name: "QuickTask",
    sub_heading: "Fast task automation",
    creator: "EfficientWorks",
    description: "Simple and efficient task automation tool.",
    runs: 50000,
    rating: 4.2,
    agent_image:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    creator_avatar:
      "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
    slug: "quicktask",
  },
] satisfies StoreAgent[];

export const Default: Story = {
  args: {
    featuredAgents: mockFeaturedAgents,
  },
};

export const SingleAgent: Story = {
  args: {
    featuredAgents: [mockFeaturedAgents[0]],
  },
};

export const NoAgents: Story = {
  args: {
    featuredAgents: [],
  },
};

export const WithMissingData: Story = {
  args: {
    featuredAgents: [
      {
        ...mockFeaturedAgents[0],
        sub_heading: "",
        agent_image: "",
        creator_avatar: "",
      },
      mockFeaturedAgents[1],
    ],
  },
};

export const WithExtremelyLongTexts: Story = {
  args: {
    featuredAgents: [
      {
        ...mockFeaturedAgents[0],
        agent_name:
          "Universal Language Translator Pro with Advanced Neural Network Technology and Cross-Cultural Communication Capabilities for International Business and Education with Extended Support for Rare Dialects",
        sub_heading:
          "Breaking language barriers with cutting-edge AI translation technology that revolutionizes global communication for businesses and individuals across continents while preserving cultural nuances and contextual meanings in more than 150 languages including endangered dialects and specialized terminology",
        description:
          "Experience seamless communication across 150+ languages with our advanced neural translation engine. Perfect for international businesses, travelers, and language enthusiasts. Features real-time conversation translation, document processing, and cultural context adaptation to ensure your message is delivered exactly as intended in any language. Our proprietary machine learning algorithms continuously improve translation accuracy with each interaction, adapting to regional dialects and specialized terminology. The system includes voice recognition capabilities, image-to-text translation for signs and documents, and can operate offline in emergency situations where internet connectivity is limited. With dedicated mobile apps for iOS and Android plus browser extensions, you'll never encounter language barriers again, whether in business negotiations, academic research, or while exploring new destinations.",
        creator:
          "Global Linguistics Technologies International Corporation and Research Institute for Cross-Cultural Communication",
        runs: 999999999999,
      },
      mockFeaturedAgents[1],
    ],
  },
};

export const WithManyAgents: Story = {
  args: {
    featuredAgents: Array(20)
      .fill(null)
      .map((_, i) => ({
        ...mockFeaturedAgents[i % mockFeaturedAgents.length],
        agent_name: `Agent ${i + 1}: ${mockFeaturedAgents[i % mockFeaturedAgents.length].agent_name}`,
        slug: `agent-${i + 1}`,
      })),
  },
};

export const WithInvalidImageURLs: Story = {
  args: {
    featuredAgents: [
      {
        ...mockFeaturedAgents[0],
        agent_image: "https://invalid-url.jpg",
        creator_avatar: "https://another-invalid-url.jpg",
      },
      mockFeaturedAgents[1],
    ],
  },
};

export const WithSpecialCharacters: Story = {
  args: {
    featuredAgents: [
      {
        ...mockFeaturedAgents[0],
        agent_name: "Special &<>\"'/ Characters",
        creator: "User with @#$%^&*()_+{}[]",
        slug: "special-characters!@#$%",
      },
      mockFeaturedAgents[1],
    ],
  },
};

export const MobileView: Story = {
  args: {
    featuredAgents: mockFeaturedAgents,
  },
  parameters: {
    viewport: {
      defaultViewport: "mobile2",
    },
  },
};

export const WithCardInteraction: Story = {
  args: {
    featuredAgents: mockFeaturedAgents,
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Find and interact with first card
    const cards = canvas.getAllByTestId("featured-store-card");
    await expect(cards.length).toBeGreaterThan(0);

    const firstCard = cards[0];
    await userEvent.hover(firstCard);
    await new Promise((resolve) => setTimeout(resolve, 300));

    // Check that link is present and clickable
    const cardLink = firstCard.closest("a");
    await expect(cardLink).toBeInTheDocument();

    await userEvent.click(firstCard);
  },
};
