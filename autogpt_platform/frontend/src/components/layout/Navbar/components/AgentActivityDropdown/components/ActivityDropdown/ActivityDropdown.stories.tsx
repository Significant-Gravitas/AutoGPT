import type { Meta, StoryObj } from "@storybook/react";
import { ActivityDropdown } from "./ActivityDropdown";

const mockExecutions = [
  {
    type: "running" as const,
    agent_name: "Web Scraper Agent",
    agent_description: "Scrapes data from websites",
    execution_id: "exec-1",
    started_at: new Date().toISOString(),
    stats: { nodes_total: 10, nodes_completed: 5, nodes_failed: 0 },
  },
  {
    type: "completed" as const,
    agent_name: "Data Analyzer",
    agent_description: "Analyzes datasets",
    execution_id: "exec-2",
    started_at: new Date(Date.now() - 3600000).toISOString(),
    ended_at: new Date().toISOString(),
    stats: { nodes_total: 8, nodes_completed: 8, nodes_failed: 0 },
  },
  {
    type: "failed" as const,
    agent_name: "Email Sender",
    agent_description: "Sends automated emails",
    execution_id: "exec-3",
    started_at: new Date(Date.now() - 7200000).toISOString(),
    ended_at: new Date(Date.now() - 3600000).toISOString(),
    stats: { nodes_total: 5, nodes_completed: 3, nodes_failed: 2 },
  },
];

const meta: Meta<typeof ActivityDropdown> = {
  title: "Layout/Navbar/ActivityDropdown",
  component: ActivityDropdown,
  parameters: {
    layout: "centered",
  },
  decorators: [
    (Story) => (
      <div style={{ width: 320, background: "white", borderRadius: 8, boxShadow: "0 4px 12px rgba(0,0,0,0.15)" }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof ActivityDropdown>;

export const WithActivity: Story = {
  args: {
    activeExecutions: [mockExecutions[0]] as any,
    recentCompletions: [mockExecutions[1]] as any,
    recentFailures: [mockExecutions[2]] as any,
  },
};

export const Empty: Story = {
  args: {
    activeExecutions: [],
    recentCompletions: [],
    recentFailures: [],
  },
};

export const ManyItems: Story = {
  args: {
    activeExecutions: Array(5).fill(mockExecutions[0]).map((e, i) => ({ ...e, execution_id: `running-${i}` })) as any,
    recentCompletions: Array(10).fill(mockExecutions[1]).map((e, i) => ({ ...e, execution_id: `completed-${i}` })) as any,
    recentFailures: Array(3).fill(mockExecutions[2]).map((e, i) => ({ ...e, execution_id: `failed-${i}` })) as any,
  },
};
