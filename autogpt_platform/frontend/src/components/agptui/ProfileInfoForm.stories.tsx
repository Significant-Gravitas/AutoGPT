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
};

export default meta;
type Story = StoryObj<typeof ProfileInfoForm>;

export const Empty: Story = {
  args: {
    displayName: "",
    handle: "",
    bio: "",
    profileImage: undefined,
    links: [],
    categories: [],
  },
};

export const Filled: Story = {
  args: {
    displayName: "Olivia Grace",
    handle: "@ograce1421",
    bio: "Our agents are designed to bring happiness and positive vibes to your daily routine. Each template helps you create and live more efficiently.",
    profileImage: "https://via.placeholder.com/130x130",
    links: [
      { id: 1, url: "www.websitelink.com" },
      { id: 2, url: "twitter.com/oliviagrace" },
      { id: 3, url: "github.com/ograce" },
    ],
    categories: [
      { id: 1, name: "Entertainment" },
      { id: 2, name: "Blog" },
      { id: 3, name: "Content creation" },
    ],
  },
};
