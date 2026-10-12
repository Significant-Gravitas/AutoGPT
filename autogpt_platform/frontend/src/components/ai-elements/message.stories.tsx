import type { Meta, StoryObj } from "@storybook/nextjs";

import { Message, MessageContent, MessageResponse } from "./message";

const meta: Meta<typeof Message> = {
  title: "AI Elements/Message",
  component: Message,
  parameters: {
    layout: "padded",
  },
};

export default meta;
type Story = StoryObj<typeof Message>;

type MessageStoryProps = { children: string };

function AssistantMessage({ children }: MessageStoryProps) {
  return (
    <Message from="assistant">
      <MessageContent>
        <MessageResponse>{children}</MessageResponse>
      </MessageContent>
    </Message>
  );
}

function UserMessage({ children }: MessageStoryProps) {
  return (
    <Message from="user">
      <MessageContent>
        <MessageResponse>{children}</MessageResponse>
      </MessageContent>
    </Message>
  );
}

export const Default: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <AssistantMessage>
        {
          "Here is a response with **bold text**, *italic text*, and `inline code`."
        }
      </AssistantMessage>
    </div>
  ),
};

export const UserMessageStory: Story = {
  name: "User Message",
  render: () => (
    <div className="flex flex-col gap-4">
      <UserMessage>{"How do I download my workspace files?"}</UserMessage>
    </div>
  ),
};

export const WithLinks: Story = {
  name: "With Links (Internal & External)",
  render: () => (
    <div className="flex flex-col gap-4">
      <AssistantMessage>
        {[
          "Here are some links:\n\n",
          "- Internal link: [Download file](/api/proxy/api/v1/workspace/files/download/abc123)\n",
          "- External link: [GitHub](https://github.com/Significant-Gravitas/AutoGPT)\n",
          "- Another external: [Documentation](https://docs.agpt.co)\n\n",
          "Internal links should open directly. External links should show a safety modal.\n\n",
          "**Try clicking each link to verify behavior.**",
        ].join("")}
      </AssistantMessage>
    </div>
  ),
};

export const LinkSafetyModal: Story = {
  name: "LinkSafetyModal",
  render: () => (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-muted-foreground">
        Click the external link below to trigger the link safety modal. Verify
        that both &quot;Copy link&quot; and &quot;Open link&quot; buttons are
        visible.
      </p>
      <AssistantMessage>
        {
          "Click this external link to see the safety modal: [Example Site](https://example.com)"
        }
      </AssistantMessage>
    </div>
  ),
};

export const WithTable: Story = {
  render: () => (
    <div className="flex max-w-3xl flex-col gap-4">
      <AssistantMessage>
        {[
          "Here are the **10 most recent unresolved issues** in `significant-gravitas`, ranked by impact:\n\n",
          "| # | Issue | Events | Users | Last seen |\n",
          "|---|---|---|---|---|\n",
          "| 1 | **[BUILDER-62F]** LaunchDarkly `waitForInitialization` timeout warning | 186,914 | 19,785 | 2 min ago |\n",
          "| 2 | **[AUTOGPT-SERVER-96T]** UserPaywalledError: subscription required (Scheduler) | 137,759 | 0 | now |\n",
          "| 3 | **[BUILDER-9]** ApplicationNotResponding ≥5000 ms | 15,867 | 0 | now |\n",
          "| 4 | **[BUILDER-8KQ]** Local storage is not available | 1,914 | 0 | 2 min ago |\n",
          "| 5 | **[BUILDER-3S3]** WebSocket error `[object Event]` (`/onboarding`) | 1,555 | 989 | now |\n",
          "| 6 | **[AUTOGPT-SERVER-9QY]** model_router: `claude-opus-4-8` default refused by catalog | 218 | 0 | now |\n",
          "| 7 | **[BUILDER-Q]** NetworkError fetching resource | 69 | 21 | 2 min ago |\n",
          "| 8 | **[AUTOGPT-SERVER-9R1]** copilot routing refused `claude-opus-4-8` slug | 67 | 0 | now |\n",
          "| 9 | **[BUILDER-26P]** NetworkError (dev-builder.agpt.co) | 48 | 19 | 2 min ago |\n",
          "| 10 | **[BUILDER-8TA]** Synchronous XMLHttpRequest deprecation | 20 | 1 | 2 min ago |\n\n",
          "Two stand out: **BUILDER-3S3** (WebSocket error hitting 989 real users on onboarding) and the **model_router** pair refusing `claude-opus-4-8`.",
        ].join("")}
      </AssistantMessage>
    </div>
  ),
};

export const Conversation: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <UserMessage>{"Can you help me with my workspace files?"}</UserMessage>
      <AssistantMessage>
        {[
          "Sure! Here's how to manage your workspace files:\n\n",
          "1. Upload files using the attachment button\n",
          "2. Download files by clicking the link in chat\n",
          "3. View all files in the [workspace panel](/workspace)\n\n",
          "For more details, check the [documentation](https://docs.agpt.co/workspace).",
        ].join("")}
      </AssistantMessage>
    </div>
  ),
};
