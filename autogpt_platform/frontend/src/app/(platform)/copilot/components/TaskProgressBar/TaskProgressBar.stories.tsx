import { TooltipProvider } from "@/components/ui/tooltip";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { ChatInput } from "../ChatInput/ChatInput";
import { TaskProgressBar } from "./TaskProgressBar";
import type { TodoItem } from "./helpers";

const SAMPLE: TodoItem[] = [
  { content: "Read the uploaded spreadsheet", status: "completed" },
  { content: "Clean and normalize the rows", status: "completed" },
  {
    content: "Generate the summary chart",
    status: "in_progress",
    activeForm: "Generating the summary chart…",
  },
  { content: "Write the report", status: "pending" },
  { content: "Attach the report to the chat", status: "pending" },
];

const meta: Meta<typeof TaskProgressBar> = {
  title: "Copilot/TaskProgressBar",
  component: TaskProgressBar,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Compact, collapsible task-list bar meant to sit directly above the chat input. Shows the current task + completion count when collapsed; expands to the full checklist. Presentational only — the backend wiring comes in a later step.",
      },
    },
  },
  argTypes: {
    isStreaming: { control: "boolean" },
    defaultExpanded: { control: "boolean" },
  },
  decorators: [
    (Story) => (
      // Mirrors ChatContainer: the bar sits just above the chat input inside the
      // #fafafa chat column. ChatInput is shown for placement reference only.
      <TooltipProvider>
        <div className="mx-auto flex w-full max-w-3xl flex-col bg-[#fafafa] px-3 pb-2 pt-2">
          <div className="relative z-10">
            <Story />
          </div>
          <ChatInput
            onSend={() => {}}
            placeholder="What else can I help with?"
            hasSession
          />
        </div>
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Collapsed: Story = {
  name: "Collapsed — Streaming",
  args: {
    todos: SAMPLE,
    isStreaming: true,
    defaultExpanded: false,
  },
};

export const Expanded: Story = {
  name: "Expanded — Streaming",
  args: {
    todos: SAMPLE,
    isStreaming: true,
    defaultExpanded: true,
  },
};

export const Idle: Story = {
  name: "Idle — In-progress shown as pending",
  args: {
    todos: SAMPLE,
    isStreaming: false,
    defaultExpanded: true,
  },
  parameters: {
    docs: {
      description: {
        story:
          "When not streaming, the in-progress task renders as pending (no active spinner/bold) so an idle agent doesn't look like it's still working.",
      },
    },
  },
};

export const JustStarted: Story = {
  name: "Just started — Nothing completed",
  args: {
    todos: [
      {
        content: "Analyze the goal",
        status: "in_progress",
        activeForm: "Analyzing the goal…",
      },
      { content: "Break it into steps", status: "pending" },
      { content: "Execute each step", status: "pending" },
    ],
    isStreaming: true,
    defaultExpanded: false,
  },
};

export const AllComplete: Story = {
  name: "All tasks complete",
  args: {
    todos: SAMPLE.map((t) => ({ ...t, status: "completed" as const })),
    isStreaming: false,
    defaultExpanded: false,
  },
};

export const SingleTask: Story = {
  name: "Single task",
  args: {
    todos: [
      {
        content: "Draft the email",
        status: "in_progress",
        activeForm: "Drafting the email…",
      },
    ],
    isStreaming: true,
    defaultExpanded: true,
  },
};

export const LongList: Story = {
  name: "Long list — Scrolls when expanded",
  args: {
    todos: Array.from({ length: 12 }, (_, i) => ({
      content: `Step ${i + 1}: do the thing number ${i + 1} with a fairly long description that may truncate`,
      status: i < 4 ? "completed" : i === 4 ? "in_progress" : "pending",
      activeForm: i === 4 ? "Working on step 5…" : undefined,
    })) as TodoItem[],
    isStreaming: true,
    defaultExpanded: true,
  },
};

export const Empty: Story = {
  name: "Empty — Renders nothing",
  args: {
    todos: [],
    isStreaming: false,
  },
  parameters: {
    docs: {
      description: {
        story:
          "With no tasks the component returns null so the above-input area stays clean until Autopilot populates a list.",
      },
    },
  },
};
