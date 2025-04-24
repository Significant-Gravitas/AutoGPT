import type { Meta, StoryObj } from "@storybook/react";
import { FeaturedAgentCard } from "./FeaturedAgentCard";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Featured Store Card",
  component: FeaturedAgentCard,
  parameters: {
    layout: {
      center: true,
      padding: 0,
    },
  },
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    agent: {
      agent_name: { control: "text" },
      sub_heading: { control: "text" },
      agent_image: { control: "text" },
      creator_avatar: { control: "text" },
      creator: { control: "text" },
      runs: { control: "number" },
      rating: { control: "number", min: 0, max: 5, step: 0.1 },
      slug: { control: "text" },
    },
    backgroundColor: {
      control: "color",
    },
  },
} satisfies Meta<typeof FeaturedAgentCard>;

export default meta;
type Story = StoryObj<typeof meta>;

const BACKGROUND_COLORS = [
  "bg-violet-100 hover:bg-violet-200 dark:bg-violet-800",
  "bg-blue-100 hover:bg-blue-200 dark:bg-blue-800",
  "bg-green-100 hover:bg-green-200 dark:bg-green-800",
];

export const Default: Story = {
  args: {
    agent: {
      slug: "ai-writing-assistant",
      agent_name:
        "Personalized Morning Coffee Newsletter example of three lines",
      sub_heading:
        "Transform ideas into breathtaking images with this AI-powered Image Generator.",
      description:
        "Transform ideas into breathtaking images with this AI-powered Image Generator.",
      agent_image:
        "https://framerusercontent.com/images/KCIpxr9f97EGJgpaoqnjKsrOPwI.jpg",
      creator_avatar: "/default_avatar.avif",
      creator: "John Ababesh",
      runs: 200000,
      rating: 4.6,
    },
    backgroundColor: BACKGROUND_COLORS[0],
  },
};

export const WithInteraction: Story = {
  args: {
    agent: {
      ...Default.args.agent,
      runs: 200000,
      rating: 4.6,
    },
    backgroundColor: BACKGROUND_COLORS[1],
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const card = canvas.getByTestId("featured-store-card");

    await expect(card).toBeInTheDocument();

    await userEvent.hover(card);
    await new Promise((resolve) => setTimeout(resolve, 300));

    const description = canvas.getByTestId("agent-description");
    await expect(description).toBeVisible();
    const creatorAvatar = canvas.getByAltText(
      `${Default.args.agent.creator} avatar`,
    );
    const agentImage = canvas.getByAltText(
      `${Default.args.agent.agent_name} preview`,
    );
    await expect(creatorAvatar).toHaveStyle({ opacity: "0" });
    await expect(agentImage).toHaveStyle({ opacity: "0" });

    await userEvent.unhover(card);
    await new Promise((resolve) => setTimeout(resolve, 300));
  },
};

export const ExtraLarge: Story = {
  args: {
    agent: {
      agent_name:
        "Universal Language Translator Pro with Advanced Neural Network Technology and Cross-Cultural Communication Capabilities",
      sub_heading:
        "Breaking language barriers with cutting-edge AI translation technology that revolutionizes global communication for businesses and individuals across continents while preserving cultural nuances and contextual meanings",
      description:
        "Experience seamless communication across 150+ languages with our advanced neural translation engine. Perfect for international businesses, travelers, and language enthusiasts. Features real-time conversation translation, document processing, and cultural context adaptation to ensure your message is delivered exactly as intended in any language. Our proprietary machine learning algorithms continuously improve translation accuracy with each interaction, adapting to regional dialects and specialized terminology. The system includes voice recognition capabilities, image-to-text translation for signs and documents, and can operate offline in emergency situations where internet connectivity is limited. With dedicated mobile apps for iOS and Android plus browser extensions, you'll never encounter language barriers again, whether in business negotiations, academic research, or while exploring new destinations.",
      agent_image: Default.args.agent.agent_image,
      creator_avatar: Default.args.agent.creator_avatar,
      creator:
        "Global Linguistics Technologies International Corporation and Research Institute for Cross-Cultural Communication",
      runs: 1000000,
      rating: 4.9,
      slug: "universal-translator-pro-with-advanced-neural-networks-and-multilingual-support-for-global-enterprise-solutions-and-individual-travelers",
    },
    backgroundColor: BACKGROUND_COLORS[2],
  },
};

export const MinimalText: Story = {
  args: {
    agent: {
      agent_name: "A",
      sub_heading: "B",
      description: "C",
      agent_image: Default.args.agent.agent_image,
      creator_avatar: Default.args.agent.creator_avatar,
      creator: "D",
      runs: 0,
      rating: 0,
      slug: Default.args.agent.slug,
    },
    backgroundColor: BACKGROUND_COLORS[0],
  },
};

export const FullRating: Story = {
  args: {
    agent: {
      ...Default.args.agent,
      rating: 5,
    },
    backgroundColor: BACKGROUND_COLORS[1],
  },
};

export const DecimalRating: Story = {
  args: {
    agent: {
      ...Default.args.agent,
      rating: 3.5,
    },
    backgroundColor: BACKGROUND_COLORS[2],
  },
};

export const LargeRuns: Story = {
  args: {
    agent: {
      ...Default.args.agent,
      runs: 1000000000,
    },
    backgroundColor: BACKGROUND_COLORS[0],
  },
};

export const MissingImage: Story = {
  args: {
    agent: {
      ...Default.args.agent,
      agent_image: "",
    },
    backgroundColor: BACKGROUND_COLORS[2],
  },
};

export const MissingAvatar: Story = {
  args: {
    agent: {
      ...Default.args.agent,
      creator_avatar: "",
    },
    backgroundColor: BACKGROUND_COLORS[2],
  },
};
