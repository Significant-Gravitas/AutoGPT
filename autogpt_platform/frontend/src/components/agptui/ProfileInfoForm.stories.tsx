import type { Meta, StoryObj } from "@storybook/react";
import { ProfileInfoForm } from "./ProfileInfoForm";

const meta: Meta<typeof ProfileInfoForm> = {
  title: "AGPT UI/Profile/Profile Info Form",
  component: ProfileInfoForm,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
  argTypes: {
    profile: {
      control: "object",
      description: "The profile details of the user",
      displayName: {
        control: "text",
        description: "The display name of the user",
      },
      handle: {
        control: "text",
        description: "The user's handle/username",
      },
      bio: {
        control: "text",
        description: "User's biography text",
      },
      profileImage: {
        control: "text",
        description: "URL of the user's profile image",
      },
      links: {
        control: "object",
        description: "Array of social media links",
      },
      categories: {
        control: "object",
        description: "Array of selected categories",
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof ProfileInfoForm>;

export const Empty: Story = {
  args: {
    profile: {
      name: "",
      username: "",
      description: "",
      avatar_url: "",
      links: [],
      top_categories: [],
      agent_rating: 0,
      agent_runs: 0,
    },
  },
};

export const Filled: Story = {
  args: {
    profile: {
      name: "Olivia Grace",
      username: "@ograce1421",
      description:
        "Our agents are designed to bring happiness and positive vibes to your daily routine. Each template helps you create and live more efficiently.",
      avatar_url: "https://via.placeholder.com/130x130",
      links: [
        "www.websitelink.com",
        "twitter.com/oliviagrace",
        "github.com/ograce",
      ],
      top_categories: ["Entertainment", "Blog", "Content creation"],
      agent_rating: 4.5,
      agent_runs: 100,
    },
  },
};
