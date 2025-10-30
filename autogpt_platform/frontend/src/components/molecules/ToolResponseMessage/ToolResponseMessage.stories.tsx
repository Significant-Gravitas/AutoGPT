import type { Meta, StoryObj } from "@storybook/nextjs";
import { ToolResponseMessage } from "./ToolResponseMessage";

const meta = {
  title: "Molecules/ToolResponseMessage",
  component: ToolResponseMessage,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof ToolResponseMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SuccessString: Story = {
  args: {
    toolId: "tool_abc123def456ghi789jkl012mno345",
    toolName: "search_database",
    result: "Found 15 matching records",
    success: true,
  },
};

export const SuccessObject: Story = {
  args: {
    toolId: "tool_xyz987wvu654tsr321qpo098nml765",
    toolName: "get_user_info",
    result: {
      id: "user_123",
      name: "John Doe",
      email: "john@example.com",
      status: "active",
      created_at: "2024-01-15T10:30:00Z",
    },
    success: true,
  },
};

export const FailedString: Story = {
  args: {
    toolId: "tool_def456ghi789jkl012mno345pqr678",
    toolName: "send_email",
    result: "Failed to send email: SMTP connection timeout",
    success: false,
  },
};

export const FailedObject: Story = {
  args: {
    toolId: "tool_ghi789jkl012mno345pqr678stu901",
    toolName: "api_request",
    result: {
      error: "Authentication failed",
      code: "AUTH_ERROR",
      status: 401,
      message: "Invalid API key provided",
    },
    success: false,
  },
};

export const LongStringResult: Story = {
  args: {
    toolId: "tool_jkl012mno345pqr678stu901vwx234",
    toolName: "analyze_text",
    result:
      "Analysis complete. The text contains 150 words, 8 sentences, and 5 paragraphs. Sentiment: Positive (0.85). Key topics: technology, innovation, future. Entities detected: 3 organizations, 5 people, 2 locations. The overall tone is optimistic and forward-looking. Primary language: English. Reading level: Grade 10. Estimated reading time: 45 seconds. The text demonstrates strong coherence with clear topic progression. Main themes include digital transformation, artificial intelligence, and sustainable development.",
    success: true,
  },
};

export const ComplexNestedObject: Story = {
  args: {
    toolId: "tool_mno345pqr678stu901vwx234yza567",
    toolName: "process_data",
    result: {
      status: "completed",
      processed_items: 1250,
      duration_ms: 3450,
      results: {
        valid: 1200,
        invalid: 50,
        errors: [
          { line: 45, message: "Invalid format" },
          { line: 128, message: "Missing required field" },
        ],
      },
      summary: {
        categories: {
          type_a: 450,
          type_b: 600,
          type_c: 150,
        },
        average_score: 87.5,
        confidence: 0.94,
      },
    },
    success: true,
  },
};

export const VeryLongObjectResult: Story = {
  args: {
    toolId: "tool_pqr678stu901vwx234yza567bcd890",
    toolName: "generate_report",
    result: {
      report_id: "rep_20240130_1530",
      generated_at: "2024-01-30T15:30:00Z",
      data: {
        metrics: {
          total_users: 15234,
          active_users: 12890,
          new_users: 456,
          retention_rate: 84.6,
          engagement_score: 7.8,
        },
        performance: {
          response_time_avg: 245,
          error_rate: 0.02,
          uptime_percentage: 99.95,
        },
        revenue: {
          total: 125000.0,
          currency: "USD",
          breakdown: {
            subscriptions: 100000,
            one_time: 25000,
          },
        },
      },
      insights:
        "User engagement increased by 15% compared to last month. Response times improved by 8%. Revenue growth of 12% quarter over quarter. Retention rate remains stable.",
    },
    success: true,
  },
};
