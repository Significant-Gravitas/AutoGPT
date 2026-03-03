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
