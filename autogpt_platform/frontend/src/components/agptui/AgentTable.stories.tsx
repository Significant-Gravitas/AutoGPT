import type { Meta, StoryObj } from "@storybook/react";
import { AgentTable } from "./AgentTable";
import { within, expect, fn } from "@storybook/test";
import { StatusType } from "./Status";

const meta = {
  title: "Agpt Custom UI/marketing/Agent Table",
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
    onEditSubmission: fn(() => {
      console.log("Edit submission");
    }),
    onDeleteSubmission: fn(() => {
      console.log(`Delete submission for agent `);
    }),
  },
};

export const LongContent: Story = {
  args: {
    ...Default.args,
    agents: [
      {
        ...sampleAgents[0],
        agentName:
          "Super Advanced Artificial Intelligence Code Generator and Optimizer with Machine Learning Capabilities",
        description:
          "This is an extremely advanced artificial intelligence code generator that can write clean, efficient, and optimized code in multiple programming languages. It utilizes state-of-the-art machine learning algorithms to understand requirements and generate appropriate solutions while following best practices and design patterns. The agent can handle complex programming tasks, debug existing code, and suggest improvements to enhance performance and readability.",
      },
      {
        ...sampleAgents[1],
        agentName:
          "Super Advanced Artificial Intelligence Code Generator and Optimizer with Machine Learning Capabilities",
        description:
          "A sophisticated data analysis tool capable of processing petabytes of structured and unstructured data to extract meaningful insights and patterns. This agent leverages advanced statistical methods, machine learning techniques, and data visualization capabilities to transform raw data into actionable business intelligence. It can handle time series analysis, predictive modeling, anomaly detection, and generate comprehensive reports with minimal human intervention.",
      },
      {
        ...sampleAgents[2],
        agentName:
          "Super Advanced Artificial Intelligence Code Generator and Optimizer with Machine Learning Capabilities",
        description:
          "This specialized UI/UX design assistant creates beautiful, accessible, and intuitive user interfaces for web and mobile applications. By combining principles of human-centered design with modern aesthetic sensibilities, the agent produces wireframes, mockups, and interactive prototypes that enhance user engagement and satisfaction. It follows design systems, ensures consistent branding, and optimizes layouts for various screen sizes while maintaining accessibility standards.",
      },
    ],
  },
};

export const ManyAgents: Story = {
  args: {
    ...Default.args,
    agents: Array(20)
      .fill(null)
      .map((_, index) => ({
        ...sampleAgents[index % 3],
        id: 100 + index,
        agent_id: `${100 + index}`,
        agentName: `Test Agent ${index + 1}`,
      })),
  },
};

export const EmptyTableTest: Story = {
  args: {
    ...Default.args,
    agents: [],
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

export const TestingInteractions: Story = {
  args: {
    ...Default.args,
    agents: sampleAgents,
  },
  play: async ({ canvasElement, args }) => {
    const canvas = within(canvasElement);

    const checkboxes = canvas.getAllByTestId("dropdown-button");
    await expect(checkboxes.length).toBeGreaterThan(0);
  },
};
