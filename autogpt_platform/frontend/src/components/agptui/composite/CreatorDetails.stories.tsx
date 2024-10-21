import type { Meta, StoryObj } from "@storybook/react";
import { CreatorDetails } from "./CreatorDetails";

const meta = {
  title: "AGPT UI/Composite/Creator Details",
  component: CreatorDetails,
  parameters: {
    layout: "centered",
  },
  tags: ["autodocs"],
  argTypes: {
    name: { control: "text" },
    username: { control: "text" },
    description: { control: "text" },
    avgRating: { control: "number", min: 0, max: 5, step: 0.1 },
    agentCount: { control: "number" },
    topCategories: { control: "object" },
    otherLinks: { control: "object" },
    avatarSrc: { control: "text" }, // Added avatarSrc to argTypes
  },
} satisfies Meta<typeof CreatorDetails>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    name: "John Doe",
    username: "johndoe",
    description:
      "Our agents are designed to bring happiness and positive vibes to your daily routine. Each template helps you create and live the life of your dreams while guiding you to become your best every day",
    avgRating: 4.5,
    agentCount: 10,
    topCategories: ["AI", "NLP", "Machine Learning"],
    otherLinks: {
      website: "https://johndoe.com",
      github: "https://github.com/johndoe",
      linkedin: "https://linkedin.com/in/johndoe",
      medium: "https://medium.com/@johndoe",
      youtube: "https://youtube.com/@johndoe",
      tiktok: "https://tiktok.com/@johndoe",
    },
    avatarSrc: "https://github.com/shadcn.png", // Added avatarSrc to Default args
  },
};

export const NewCreator: Story = {
  args: {
    ...Default.args,
    name: "Jane Smith",
    username: "janesmith",
    description:
      "Aspiring AI enthusiast with a focus on computer vision and image processing.",
    avgRating: 3.8,
    agentCount: 2,
    topCategories: ["Computer Vision", "Image Processing"],
    otherLinks: {
      tiktok: "https://tiktok.com/@johndoe",
    },
    avatarSrc: "https://example.com/avatar2.jpg", // Added avatarSrc to NewCreator args
  },
};

export const ExperiencedCreator: Story = {
  args: {
    ...Default.args,
    name: "AI Labs Inc.",
    username: "ailabs",
    description:
      "Leading AI research company developing cutting-edge solutions for various industries.",
    avgRating: 4.9,
    agentCount: 50,
    topCategories: ["AI Research", "Deep Learning", "Robotics", "NLP"],
    otherLinks: {
      website: "https://ailabs.com",
      github: "https://github.com/ailabs",
      linkedin: "https://linkedin.com/company/ailabs",
    },
    avatarSrc: "https://example.com/avatar3.jpg", // Added avatarSrc to ExperiencedCreator args
  },
};

export const LongDescription: Story = {
  args: {
    ...Default.args,
    name: "Tech Innovations",
    username: "techinnovations",
    description:
      "We are a team of passionate developers and researchers dedicated to pushing the boundaries of artificial intelligence. Our focus spans across multiple domains including natural language processing, computer vision, and reinforcement learning. With years of experience in both academia and industry, we strive to create AI agents that are not only powerful but also ethical and user-friendly.",
    avgRating: 4.7,
    agentCount: 25,
    topCategories: ["AI", "Innovation", "Research", "Deep Learning"],
    otherLinks: {
      website: "https://techinnovations.ai",
      facebook: "https://facebook.com/techinnovations",
      linkedin: "https://linkedin.com/company/techinnovations",
    },
    avatarSrc: "https://example.com/avatar4.jpg", // Added avatarSrc to LongDescription args
  },
};

export const NoLinks: Story = {
  args: {
    ...Default.args,
    name: "Tech Innovations",
    username: "techinnovations",
    description:
      "We are a team of passionate developers and researchers dedicated to pushing the boundaries of artificial intelligence. Our focus spans across multiple domains including natural language processing, computer vision, and reinforcement learning. With years of experience in both academia and industry, we strive to create AI agents that are not only powerful but also ethical and user-friendly.",
    avgRating: 4.7,
    agentCount: 25,
    otherLinks: {},
    topCategories: ["AI", "Innovation", "Research", "Deep Learning"],
    avatarSrc: "https://example.com/avatar4.jpg", // Added avatarSrc to LongDescription args
  },
};
