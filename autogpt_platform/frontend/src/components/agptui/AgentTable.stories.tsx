import type { Meta, StoryObj } from "@storybook/react";
import { AgentTable } from "./AgentTable";
import { userEvent, within, expect, fn } from "@storybook/test";
import { StatusType } from "./Status";

const meta = {
  title: "AGPT UI/Agent Table",
  component: AgentTable,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="container mx-auto p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof AgentTable>;

export default meta;
type Story = StoryObj<typeof meta>;

const sampleAgents = [
  {
    id: 43,
    agentName: "Super Coder",
    description: "An AI agent that writes clean, efficient code",
    imageSrc: [
      "https://ddz4ak4pa3d19.cloudfront.net/cache/53/b2/53b2bc7d7900f0e1e60bf64ebf38032d.jpg",
    ],
    dateSubmitted: "2023-05-15",
    status: "approved" as StatusType,
    runs: 1500,
    rating: 4.8,
    agent_id: "43",
    agent_version: 1,
    sub_heading: "Super Coder",
    date_submitted: "2023-05-15",
  },
  {
    id: 44,
    agentName: "Data Analyzer",
    description: "Processes and analyzes large datasets with ease",
    imageSrc: [
      "https://ddz4ak4pa3d19.cloudfront.net/cache/40/f7/40f7bc97c952f8df0f9c88d29defe8d4.jpg",
    ],
    dateSubmitted: "2023-05-10",
    status: "awaiting_review" as StatusType,
    runs: 1200,
    rating: 4.5,
    agent_id: "44",
    agent_version: 1,
    sub_heading: "Data Analyzer",
    date_submitted: "2023-05-10",
  },
  {
    id: 45,
    agentName: "UI Designer",
    description: "Creates beautiful and intuitive user interfaces",
    imageSrc: [
      "https://ddz4ak4pa3d19.cloudfront.net/cache/14/9e/149ebb9014aa8c0097e72ed89845af0e.jpg",
    ],
    dateSubmitted: "2023-05-05",
    status: "draft" as StatusType,
    runs: 800,
    rating: 4.2,
    agent_id: "45",
    agent_version: 1,
    sub_heading: "UI Designer",
    date_submitted: "2023-05-05",
  },
];

export const Default: Story = {
  args: {
    agents: sampleAgents,
    onEditSubmission: fn(),
    onDeleteSubmission: fn(),
  },
};

export const EmptyTable: Story = {
  args: {
    agents: [],
    onEditSubmission: fn(),
    onDeleteSubmission: fn(),
  },
};

export const LongAgentNames: Story = {
  args: {
    agents: [
      {
        ...sampleAgents[0],
        agentName:
          "Super Advanced Artificial Intelligence Code Generator and Optimizer with Machine Learning Capabilities",
        sub_heading:
          "A very advanced AI system that can generate and optimize code using cutting-edge machine learning techniques",
      },
      ...sampleAgents.slice(1),
    ],
    onEditSubmission: fn(),
    onDeleteSubmission: fn(),
  },
};

export const ManyAgents: Story = {
  args: {
    agents: Array(20)
      .fill(null)
      .map((_, index) => ({
        ...sampleAgents[index % 3],
        id: 100 + index,
        agent_id: `${100 + index}`,
        agentName: `Test Agent ${index + 1}`,
      })),
    onEditSubmission: fn(),
    onDeleteSubmission: fn(),
  },
};

export const WithInteraction: Story = {
  args: {
    agents: sampleAgents,
    onEditSubmission: fn(),
    onDeleteSubmission: fn(),
  },
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);

    const table = canvas.getByRole("table");
    await expect(table).toBeInTheDocument();

    const checkboxes = canvas.getAllByTestId("dropdown-button");
    await expect(checkboxes.length).toBeGreaterThan(0);
  },
};

export const EmptyTableTest: Story = {
  args: {
    agents: [],
    onEditSubmission: fn(),
    onDeleteSubmission: fn(),
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const emptyMessages = canvas.getAllByText(
      "No agents available. Create your first agent to get started!",
    );
    await expect(emptyMessages.length).toBeGreaterThan(0);
    await expect(emptyMessages[0]).toBeInTheDocument();
  },
};

export const ResponsiveTest: Story = {
  args: {
    agents: sampleAgents,
    onEditSubmission: fn(),
    onDeleteSubmission: fn(),
  },
  parameters: {
    viewport: {
      defaultViewport: "mobile2",
    },
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);

    // In mobile view, cards should be visible instead of table
    // Check for at least one card
    const cards = canvas.getAllByTestId("agent-table-card");
    await expect(cards.length).toBe(3);

    // Table should be hidden
    const tables = canvasElement.querySelectorAll("table");
    await expect(tables.length).toBe(1);
  },
};
