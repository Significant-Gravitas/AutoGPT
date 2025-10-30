import type { Meta, StoryObj } from "@storybook/nextjs";
import { ToolCallMessage } from "./ToolCallMessage";

const meta = {
  title: "Molecules/ToolCallMessage",
  component: ToolCallMessage,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof ToolCallMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Simple: Story = {
  args: {
    toolId: "tool_abc123def456ghi789jkl012mno345",
    toolName: "search_database",
    arguments: {
      query: "SELECT * FROM users WHERE active = true",
      limit: 10,
    },
  },
};

export const NoArguments: Story = {
  args: {
    toolId: "tool_xyz987wvu654tsr321qpo098nml765",
    toolName: "get_current_time",
  },
};

export const ComplexArguments: Story = {
  args: {
    toolId: "tool_def456ghi789jkl012mno345pqr678",
    toolName: "process_data",
    arguments: {
      data: {
        source: "api",
        format: "json",
        filters: ["active", "verified"],
      },
      options: {
        validate: true,
        timeout: 30000,
        retry: 3,
      },
      callback_url: "https://example.com/webhook",
    },
  },
};

export const NestedArguments: Story = {
  args: {
    toolId: "tool_ghi789jkl012mno345pqr678stu901",
    toolName: "send_email",
    arguments: {
      to: ["user@example.com", "admin@example.com"],
      subject: "Test Email",
      body: {
        text: "This is a test email",
        html: "<p>This is a <strong>test</strong> email</p>",
      },
      attachments: [
        {
          filename: "report.pdf",
          content_type: "application/pdf",
          size: 1024000,
        },
      ],
      metadata: {
        campaign_id: "camp_123",
        tags: ["automated", "test"],
      },
    },
  },
};

export const LargeArguments: Story = {
  args: {
    toolId: "tool_jkl012mno345pqr678stu901vwx234",
    toolName: "analyze_text",
    arguments: {
      text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
      options: {
        detect_language: true,
        extract_entities: true,
        sentiment_analysis: true,
        keyword_extraction: true,
        summarization: true,
      },
      max_results: 100,
    },
  },
};
