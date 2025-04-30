import type { Meta, StoryObj } from "@storybook/react";
import { AgentInfo } from "./AgentInfo";
import { userEvent, within, expect } from "@storybook/test";

const meta = {
  title: "AGPT UI/Agent Info",
  component: AgentInfo,
  parameters: {
    layout: {
      center: true,
      padding: 0,
    },
  },
  tags: ["autodocs"],
  argTypes: {
    name: { control: "text" },
    creator: { control: "text" },
    shortDescription: { control: "text" },
    longDescription: { control: "text" },
    rating: { control: "number", min: 0, max: 5, step: 0.1 },
    runs: { control: "number" },
    categories: { control: "object" },
    lastUpdated: { control: "text" },
    version: { control: "text" },
    storeListingVersionId: { control: "text" },
  },
  decorators: [
    (Story) => (
      <div className="flex items-center justify-center p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof AgentInfo>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    name: "AI Video Generator",
    storeListingVersionId: "123abc456def",
    creator: "Toran Richards",
    shortDescription:
      "Transform ideas into breathtaking videos with this AI-powered Video Generator.",
    longDescription: `Create Viral-Ready Content in Seconds! Transform trending topics into engaging videos with this cutting-edge AI Video Generator. Perfect for content creators, social media managers, and marketers looking to quickly produce high-quality content.

Key features include:
- Customizable video output
- 15+ pre-made templates
- Auto scene detection
- Smart text-to-speech
- Multiple export formats
- SEO-optimized suggestions`,
    rating: 4.7,
    runs: 1500,
    categories: ["Video", "Content Creation", "Social Media"],
    lastUpdated: "2 days ago",
    version: "1.2.0",
  },
};

export const ExtraLongName: Story = {
  args: {
    ...Default.args,
    name: "Super Advanced Ultra-Intelligent Universal Comprehensive AI-Powered Video Generator Pro Plus Premium Enterprise Edition With Extended Capabilities",
    creator: "Toran Richards & Company International",
    shortDescription:
      "Transform ideas into breathtaking videos with this AI-powered Video Generator.",
    rating: 4.7,
    runs: 1500,
  },
};

export const ExtraLongCreator: Story = {
  args: {
    ...Default.args,
    creator:
      "Global Artificial Intelligence Research and Development Consortium for Advanced Technology Implementation and Enterprise Solutions",
  },
};

export const ExtraLongDescription: Story = {
  args: {
    ...Default.args,
    longDescription: `Create Viral-Ready Content in Seconds! Transform trending topics into engaging videos with this cutting-edge AI Video Generator. Perfect for content creators, social media managers, and marketers looking to quickly produce high-quality content.

Our advanced AI algorithms analyze current trends and viewer preferences to generate videos that are more likely to engage your target audience and achieve better conversion rates. The system adapts to your brand voice and style guidelines to maintain consistency across all your content.

Key features include:
- Customizable video output with adjustable parameters for length, style, pacing, and transition effects
- 15+ pre-made templates for different content types including explainer videos, product demonstrations, social media stories, and advertisements
- Auto scene detection that intelligently segments your content into engaging chapters
- Smart text-to-speech with over 50 natural-sounding voices in multiple languages and dialects
- Multiple export formats optimized for different platforms (YouTube, Instagram, TikTok, Facebook, Twitter)
- SEO-optimized suggestions for titles, descriptions, and tags to maximize discoverability
- Analytics dashboard to track performance metrics and audience engagement
- Collaborative workspace for team projects with permission management
- Regular updates with new features and improvements based on user feedback

The AI Video Generator integrates seamlessly with your existing workflow and content management systems. You can import assets from Adobe Creative Suite, Canva, and other popular design tools. Our cloud-based processing ensures fast rendering without taxing your local system resources.

With our enterprise plan, you'll get priority support, custom template development, and advanced branding options to ensure your videos stand out in today's crowded digital landscape.`,
  },
};

export const ManyCategories: Story = {
  args: {
    ...Default.args,
    categories: [
      "Video",
      "Content Creation",
      "Social Media",
      "Marketing",
      "Artificial Intelligence",
      "Machine Learning",
      "Neural Networks",
      "Deep Learning",
      "Computer Vision",
      "NLP",
      "Automation",
      "Productivity Tools",
    ],
  },
};

export const NoCategories: Story = {
  args: {
    ...Default.args,
    categories: [],
  },
};

export const HighRuns: Story = {
  args: {
    ...Default.args,
    name: "Code Assistant",
    creator: "DevAI",
    shortDescription:
      "Get AI-powered coding help for various programming languages",
    longDescription:
      "An advanced AI coding assistant that supports multiple programming languages and frameworks. Features include code completion, refactoring suggestions, and bug detection.",
    rating: 4.8,
    runs: 1000000000,
    categories: ["Programming", "AI", "Developer Tools"],
    lastUpdated: "1 day ago",
    version: "2.1.3",
  },
};

export const Downloading: Story = {
  args: {
    ...Default.args,
    name: "Download Only Agent",
    shortDescription:
      "This agent can be downloaded but not added to library (for non-logged in users)",
  },
  parameters: {
    mockData: [
      {
        url: "/api/auth/user",
        method: "GET",
        status: 200,
        response: null,
      },
    ],
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Test download button presence for non-logged in users
    const downloadButton = canvas.getByText("Download Agent as File");
    await expect(downloadButton).toBeInTheDocument();

    await userEvent.hover(downloadButton);
    await new Promise((resolve) => setTimeout(resolve, 3000));

    await userEvent.click(downloadButton);

    // Should show downloading state
    const loadingButton = canvas.getByText("Downloading...");
    await expect(loadingButton).toBeInTheDocument();
  },
};

export const TestingInteractions: Story = {
  args: {
    ...Default.args,
    name: "Task Planner Pro",
    creator: "Productivity AI",
    shortDescription: "Plan and organize your tasks efficiently with AI",
    longDescription:
      "An intelligent task management system that helps you organize, prioritize, and complete your tasks more efficiently. Features smart scheduling and AI-powered suggestions.",
    rating: 4.2,
    runs: 50000,
    categories: ["Productivity", "Task Management", "AI"],
    lastUpdated: "3 days ago",
    version: "1.5.2",
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // Test add to library button
    const addButton = canvas.getByText("Download Agent as File");
    await userEvent.hover(addButton);

    // Test category interaction
    const category = canvas.getByText("Productivity");
    await userEvent.hover(category);
  },
};
