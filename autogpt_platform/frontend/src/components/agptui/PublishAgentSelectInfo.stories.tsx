import type { Meta, StoryObj } from "@storybook/react";
import { PublishAgentInfo } from "./PublishAgentSelectInfo";
import { expect, userEvent, within } from "@storybook/test";

const meta: Meta<typeof PublishAgentInfo> = {
  title: "Agpt Custom UI/marketing/Publish Agent Select Info",
  component: PublishAgentInfo,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="backdrop-blur-4 flex h-screen items-center justify-center bg-black/40">
        <Story />
      </div>
    ),
  ],
  argTypes: {
    onBack: { action: "back clicked" },
    onSubmit: { action: "submit clicked" },
    onClose: { action: "close clicked" },
  },
};

export default meta;
type Story = StoryObj<typeof PublishAgentInfo>;

export const Default: Story = {
  args: {
    onBack: () => console.log("Back clicked"),
    onSubmit: () => console.log("Submit clicked"),
    onClose: () => console.log("Close clicked"),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const titleInput = canvas.getByLabelText(/title/i);

    await userEvent.type(titleInput, "Test Agent");
    await expect(titleInput).toHaveValue("Test Agent");
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
      category: "marketing",
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
      category: "marketing",
      description:
        "This agent allows you to upload and manage multiple images.",
      additionalImages: [
        "https://picsum.photos/seed/second/500/350",
        "https://picsum.photos/seed/third/500/350",
      ],
    },
  },
};

export const MaxImages: Story = {
  args: {
    ...Default.args,
    initialData: {
      agent_id: "1",
      slug: "super-seo-optimizer",
      title: "Gallery Agent",
      subheader: "Showcasing maximum allowed images",
      thumbnailSrc: "https://picsum.photos/seed/gallery1/500/350",
      youtubeLink: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      category: "marketing",
      description: "This agent displays the maximum number of allowed images.",
      additionalImages: [
        "https://picsum.photos/seed/gallery2/500/350",
        "https://picsum.photos/seed/gallery3/500/350",
        "https://picsum.photos/seed/gallery4/500/350",
      ],
    },
  },
};
