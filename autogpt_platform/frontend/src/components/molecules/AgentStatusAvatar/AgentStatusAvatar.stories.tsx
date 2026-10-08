import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { AUTOPILOT_AVATAR_URL } from "../AutopilotAvatar/helpers";
import { AgentStatusAvatar } from "./AgentStatusAvatar";
import type { AgentStatus } from "./helpers";

const meta = {
  title: "Molecules/AgentStatusAvatar",
  component: AgentStatusAvatar,
  tags: ["autodocs"],
  parameters: { layout: "centered", a11y: { test: "error" } },
  args: {
    name: "Otto",
    src: AUTOPILOT_AVATAR_URL,
    status: "idle",
    size: "lg",
  },
} satisfies Meta<typeof AgentStatusAvatar>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Idle: Story = {};
export const Thinking: Story = { args: { status: "thinking" } };
export const Working: Story = { args: { status: "working" } };
export const Waiting: Story = { args: { status: "waiting" } };
export const Done: Story = { args: { status: "done" } };
export const WithoutImage: Story = { args: { src: undefined } };

const STATUSES: AgentStatus[] = [
  "idle",
  "thinking",
  "working",
  "waiting",
  "done",
];

export const AllStates: Story = {
  render: (args) => (
    <div className="flex items-center gap-4">
      {STATUSES.map((status) => (
        <AgentStatusAvatar key={status} {...args} status={status} />
      ))}
    </div>
  ),
};
