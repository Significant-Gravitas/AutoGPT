import type { Meta, StoryObj } from "@storybook/nextjs";
import { CredentialsNeededPrompt } from "./CredentialsNeededPrompt";

const meta = {
  title: "Molecules/CredentialsNeededPrompt",
  component: CredentialsNeededPrompt,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
  args: {
    onSetupCredentials: () => console.log("Setup credentials clicked"),
    onCancel: () => console.log("Cancel clicked"),
  },
} satisfies Meta<typeof CredentialsNeededPrompt>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ApiKey: Story = {
  args: {
    provider: "openai",
    providerName: "OpenAI",
    credentialType: "api_key",
    title: "GPT Agent",
    message: "To run GPT Agent, you need to add OpenAI credentials.",
  },
};

export const OAuth: Story = {
  args: {
    provider: "github",
    providerName: "GitHub",
    credentialType: "oauth2",
    title: "GitHub Integration Agent",
    message:
      "To run GitHub Integration Agent, you need to add GitHub credentials.",
  },
};

export const UserPassword: Story = {
  args: {
    provider: "database",
    providerName: "Database Server",
    credentialType: "user_password",
    title: "Database Query Agent",
    message:
      "To run Database Query Agent, you need to add Database Server credentials.",
  },
};

export const HostScoped: Story = {
  args: {
    provider: "custom_api",
    providerName: "Custom API",
    credentialType: "host_scoped",
    title: "Custom API Agent",
    message: "To run Custom API Agent, you need to add Custom API credentials.",
  },
};

export const LongMessage: Story = {
  args: {
    provider: "slack",
    providerName: "Slack",
    credentialType: "oauth2",
    title: "Slack Notification Agent for Team Collaboration and Updates",
    message:
      "To run Slack Notification Agent for Team Collaboration and Updates, you need to add Slack credentials. This will allow the agent to send messages, create channels, and manage workspace settings on your behalf.",
  },
};
