import type { Meta, StoryObj } from "@storybook/react";
import { PublishAgentInfo } from "./PublishAgentSelectInfo";

const meta: Meta<typeof PublishAgentInfo> = {
  title: "AGPT UI/Publish Agent Info",
  component: PublishAgentInfo,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div style={{ maxWidth: "670px", margin: "0 auto" }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof PublishAgentInfo>;

export const Default: Story = {
  args: {
    onBack: () => console.log("Back clicked"),
    onSubmit: () => console.log("Submit clicked"),
    onClose: () => console.log("Close clicked"),
  },
};

export const Filled: Story = {
  args: {
    ...Default.args,
    initialData: {
      agent_id: "1",
      slug: "super-seo-optimizer",
      title: "Super SEO Optimizer",
      subheader: "Boost your website's search engine rankings",
      thumbnailSrc: "https://picsum.photos/seed/seo/500/350",
      youtubeLink: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      category: "SEO",
      description:
        "This AI agent specializes in analyzing websites and providing actionable recommendations to improve search engine optimization. It can perform keyword research, analyze backlinks, and suggest content improvements.",
    },
  },
};

export const ThreeImages: Story = {
  args: {
    ...Default.args,
    initialData: {
      agent_id: "1",
      slug: "super-seo-optimizer",
      title: "Multi-Image Agent",
      subheader: "Showcasing multiple images",
      thumbnailSrc: "https://picsum.photos/seed/initial/500/350",
      youtubeLink: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      category: "SEO",
      description:
        "This agent allows you to upload and manage multiple images.",
      additionalImages: [
        "https://picsum.photos/seed/second/500/350",
        "https://picsum.photos/seed/third/500/350",
      ],
    },
  },
};

export const SixImages: Story = {
  args: {
    ...Default.args,
    initialData: {
      agent_id: "1",
      slug: "super-seo-optimizer",
      title: "Gallery Agent",
      subheader: "Showcasing a gallery of images",
      thumbnailSrc: "https://picsum.photos/seed/gallery1/500/350",
      youtubeLink: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      category: "SEO",
      description: "This agent displays a gallery of six images.",
      additionalImages: [
        "https://picsum.photos/seed/gallery2/500/350",
        "https://picsum.photos/seed/gallery3/500/350",
        "https://picsum.photos/seed/gallery4/500/350",
        "https://picsum.photos/seed/gallery5/500/350",
        "https://picsum.photos/seed/gallery6/500/350",
      ],
    },
  },
};
