import type { Meta, StoryObj } from "@storybook/react";
import { ChatCredentialsSetup } from "./ChatCredentialsSetup";

const meta: Meta<typeof ChatCredentialsSetup> = {
  title: "Chat/ChatCredentialsSetup",
  component: ChatCredentialsSetup,
  parameters: {
    layout: "centered",
  },
  argTypes: {
    onAllCredentialsComplete: { action: "all credentials complete" },
    onCancel: { action: "cancelled" },
  },
};

export default meta;
type Story = StoryObj<typeof ChatCredentialsSetup>;

export const SingleAPIKey: Story = {
  args: {
    credentials: [
      {
        provider: "openai",
        providerName: "OpenAI",
        credentialType: "api_key",
        title: "OpenAI API",
      },
    ],
    agentName: "GPT Assistant",
    message: "To run GPT Assistant, you need to add credentials.",
  },
};

export const SingleOAuth: Story = {
  args: {
    credentials: [
      {
        provider: "github",
        providerName: "GitHub",
        credentialType: "oauth2",
        title: "GitHub Integration",
        scopes: ["repo", "read:user"],
      },
    ],
    agentName: "GitHub Agent",
    message: "To run GitHub Agent, you need to add credentials.",
  },
};

export const MultipleCredentials: Story = {
  args: {
    credentials: [
      {
        provider: "github",
        providerName: "GitHub",
        credentialType: "oauth2",
        title: "GitHub Integration",
        scopes: ["repo", "read:user"],
      },
      {
        provider: "openai",
        providerName: "OpenAI",
        credentialType: "api_key",
        title: "OpenAI API",
      },
      {
        provider: "notion",
        providerName: "Notion",
        credentialType: "oauth2",
        title: "Notion Integration",
      },
    ],
    agentName: "Multi-Service Agent",
    message: "To run Multi-Service Agent, you need to add 3 credentials.",
  },
};

export const MixedCredentialTypes: Story = {
  args: {
    credentials: [
      {
        provider: "openai",
        providerName: "OpenAI",
        credentialType: "api_key",
        title: "OpenAI API",
      },
      {
        provider: "github",
        providerName: "GitHub",
        credentialType: "oauth2",
        title: "GitHub Integration",
        scopes: ["repo"],
      },
      {
        provider: "database",
        providerName: "Database",
        credentialType: "user_password",
        title: "Database Connection",
      },
      {
        provider: "custom_api",
        providerName: "Custom API",
        credentialType: "host_scoped",
        title: "Custom API Headers",
      },
    ],
    agentName: "Full Stack Agent",
    message: "To run Full Stack Agent, you need to add 4 credentials.",
  },
};

export const LongAgentName: Story = {
  args: {
    credentials: [
      {
        provider: "openai",
        providerName: "OpenAI",
        credentialType: "api_key",
        title: "OpenAI API",
      },
    ],
    agentName:
      "Super Complex Multi-Step Data Processing and Analysis Agent with Machine Learning",
    message:
      "To run Super Complex Multi-Step Data Processing and Analysis Agent with Machine Learning, you need to add credentials.",
  },
};
