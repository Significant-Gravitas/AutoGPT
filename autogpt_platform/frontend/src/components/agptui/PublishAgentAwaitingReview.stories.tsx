import type { Meta, StoryObj } from "@storybook/react";
import { PublishAgentAwaitingReview } from "./PublishAgentAwaitingReview";

const meta: Meta<typeof PublishAgentAwaitingReview> = {
  title: "AGPT UI/Publish Agent Awaiting Review",
  component: PublishAgentAwaitingReview,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
  },
};

export default meta;
type Story = StoryObj<typeof PublishAgentAwaitingReview>;

export const Filled: Story = {
  args: {
    agentName: "AI Video Generator",
    subheader: "Create Viral-Ready Content in Seconds",
    description:
      "AI Shortform Video Generator: Create Viral-Ready Content in Seconds Transform trending topics into engaging shortform videos with this cutting-edge AI Video Generator. Perfect for content creators, social media managers, and marketers looking to capitalize on the latest news and viral trends. Simply input your desired video count and source website, and watch as the AI scours the internet for the hottest stories, crafting them into attention-grabbing scripts optimized for platforms like TikTok, Instagram Reels, and YouTube Shorts. Key features include: - Customizable video count (1-5 per generation) - Flexible source selection for trending topics - AI-driven script writing following best practices for shortform content - Hooks that capture attention in the first 3 seconds - Dual narrative storytelling for maximum engagement - SEO-optimized content to boost discoverability - Integration with video generation tools for seamless production From hook to conclusion, each script is meticulously crafted to maintain viewer interest, incorporating proven techniques like 'but so' storytelling, visual metaphors, and strategically placed calls-to-action. The AI Shortform Video Generator streamlines your content creation process, allowing you to stay ahead of trends and consistently produce viral-worthy videos that resonate with your audience.",
    thumbnailSrc: "https://picsum.photos/seed/video/500/350",
    onClose: () => console.log("Close clicked"),
    onDone: () => console.log("Done clicked"),
    onViewProgress: () => console.log("View progress clicked"),
  },
};
