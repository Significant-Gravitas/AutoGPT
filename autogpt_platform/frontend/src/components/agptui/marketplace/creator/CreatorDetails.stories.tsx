import type { Meta, StoryObj } from "@storybook/react";
import { CreatorDetails } from "./CreatorDetails";

const meta = {
  title: "AGPTUI/Marketplace/Creator/CreatorDetails",
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
  },
} satisfies Meta<typeof CreatorDetails>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    name: "John Doe",
    username: "johndoe",
    description:
      "Experienced AI developer specializing in natural language processing and machine learning algorithms.",
    avgRating: 4.5,
    agentCount: 10,
    topCategories: ["AI", "NLP", "Machine Learning"],
    otherLinks: {
      website: "https://johndoe.com",
      github: "https://github.com/johndoe",
      linkedin: "https://linkedin.com/in/johndoe",
    },
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
      github: "https://github.com/janesmith",
    },
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
      github: "https://github.com/techinnovations",
      linkedin: "https://linkedin.com/company/techinnovations",
    },
  },
};
