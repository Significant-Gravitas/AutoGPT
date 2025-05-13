import type { Meta, StoryObj } from "@storybook/react";
import { StoreCard } from "./StoreCard";
import { userEvent, within } from "@storybook/test";

const meta = {
  title: "Agpt UI/marketing/StoreCard",
  component: StoreCard,

  decorators: [
    (Story) => (
      <div className="flex items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
  tags: ["autodocs"],
  argTypes: {
    agentName: { control: "text" },
    agentImage: { control: "text" },
    description: { control: "text" },
    runs: { control: "number" },
    rating: { control: "number", min: 0, max: 5, step: 0.1 },
    onClick: { action: "clicked" },
    avatarSrc: { control: "text" },
    hideAvatar: { control: "boolean" },
    creatorName: { control: "text" },
  },
} satisfies Meta<typeof StoreCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    agentName: "SEO Optimizer",
    agentImage: "/testing_agent_image.jpg",
    description:
      "Optimize your website's SEO with AI-powered suggestions and best practices. Get detailed reports and actionable recommendations.",
    runs: 10000,
    rating: 4.5,
    onClick: () => console.log("Default StoreCard clicked"),
    avatarSrc: "/testing_avatar.png",
    creatorName: "AI Solutions Inc.",
  },
};

export const WithInteraction: Story = {
  args: {
    agentName: "Task Planner",
    agentImage: Default.args.agentImage,
    description:
      "Plan and organize your tasks efficiently with AI assistance. Set priorities, deadlines, and track progress.",
    runs: 50000,
    rating: 4.2,
    onClick: () => console.log("WithInteraction StoreCard clicked"),
    avatarSrc: Default.args.avatarSrc,
    creatorName: "Productivity Plus",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const storeCard = canvas.getByTestId("store-card");

    await userEvent.hover(storeCard);
    await new Promise((resolve) => setTimeout(resolve, 300));
    await userEvent.click(storeCard);
  },
};

export const LongContent: Story = {
  args: {
    agentName:
      "AI Writing Assistant that can help you to write some long content so you do not have to hire for writing some basic arrangement of letters",
    agentImage: Default.args.agentImage,
    description:
      "Enhance your writing with our advanced AI-powered assistant. It offers real-time suggestions for grammar, style, and tone, helps with research and fact-checking, and provides vocabulary enhancements for more engaging content. Perfect for content creators, marketers, and writers of all levels.",
    runs: 75000,
    rating: 4.7,
    onClick: () => console.log("LongContent StoreCard clicked"),
    avatarSrc: Default.args.avatarSrc,
    creatorName:
      "The person who created the multiverst, including earth no. 631 and more..",
  },
};

export const SmallContent: Story = {
  args: {
    agentName: "Quick Notes",
    agentImage: Default.args.agentImage,
    description: "Simple note-taking assistant.",
    runs: 3000,
    rating: 4.0,
    onClick: () => console.log("SmallContent StoreCard clicked"),
    avatarSrc: Default.args.avatarSrc,
    creatorName: "Note Systems",
  },
};
