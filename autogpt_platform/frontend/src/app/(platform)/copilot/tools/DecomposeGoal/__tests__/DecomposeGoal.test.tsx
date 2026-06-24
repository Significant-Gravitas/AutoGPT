import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DecomposeGoalTool } from "../DecomposeGoal";
import type { TaskDecompositionOutput } from "../helpers";

const mockOnSend = vi.fn();
vi.mock(
  "../../../components/CopilotChatActionsProvider/useCopilotChatActions",
  () => ({
    useCopilotChatActions: () => ({ onSend: mockOnSend }),
  }),
);

const STEPS = [
  {
    step_id: "step_1",
    description: "Accept a topic from the user",
    action: "add_input",
    block_name: null,
    status: "pending",
  },
  {
    step_id: "step_2",
    description: "Summarize the topic with AI",
    action: "add_block",
    block_name: "AI Text Generator",
    status: "pending",
  },
  {
    step_id: "step_3",
    description: "Hand the result back to the user",
    action: "connect_blocks",
    block_name: null,
    status: "pending",
  },
];

const DECOMPOSITION: TaskDecompositionOutput = {
  type: "task_decomposition",
  message: "Here's the plan (3 steps):",
  goal: "Build a news summarizer",
  steps: STEPS,
  step_count: 3,
  session_id: "test-session-1",
};

function makePart(
  state: string,
  output?: unknown,
): {
  type: string;
  toolCallId: string;
  toolName: string;
  state: string;
  input?: unknown;
  output?: unknown;
} {
  return {
    type: "tool-decompose_goal",
    toolCallId: "call_1",
    toolName: "decompose_goal",
    state,
    output,
  };
}

describe("DecomposeGoalTool", () => {
  afterEach(() => {
    cleanup();
    mockOnSend.mockClear();
  });

  it("renders analyzing animation during input-streaming", () => {
    render(<DecomposeGoalTool part={makePart("input-streaming") as any} />);
    expect(screen.getByText(/A/)).toBeDefined();
  });

  it("renders error card when state is output-error", () => {
    render(<DecomposeGoalTool part={makePart("output-error") as any} />);
    expect(screen.getByText(/Failed to analyze the goal/i)).toBeDefined();
    expect(screen.getByText("Try again")).toBeDefined();
  });

  it("sends retry message when Try again is clicked on error", () => {
    render(<DecomposeGoalTool part={makePart("output-error") as any} />);
    fireEvent.click(screen.getByText("Try again"));
    expect(mockOnSend).toHaveBeenCalledWith(
      "Please try decomposing the goal again.",
    );
  });

  it("renders error card for error output object", () => {
    const errorOutput = {
      type: "error",
      error: "missing_steps",
      message: "Please provide at least one step.",
    };
    render(
      <DecomposeGoalTool
        part={makePart("output-available", errorOutput) as any}
      />,
    );
    expect(screen.getByText("Please provide at least one step.")).toBeDefined();
  });

  it("renders the build plan accordion with steps as a read-only list", () => {
    render(
      <DecomposeGoalTool
        part={makePart("output-available", DECOMPOSITION) as any}
      />,
    );
    expect(screen.getByText(/Build Plan — 3 steps/)).toBeDefined();
    expect(screen.getByText("Build a news summarizer")).toBeDefined();
    expect(screen.getByText(/Here's the plan/)).toBeDefined();
    expect(screen.getByText(/1\. Accept a topic from the user/)).toBeDefined();
    expect(screen.getByText(/2\. Summarize the topic with AI/)).toBeDefined();
    expect(
      screen.getByText(/3\. Hand the result back to the user/),
    ).toBeDefined();
  });

  it("renders block name badges for steps that have them", () => {
    render(
      <DecomposeGoalTool
        part={makePart("output-available", DECOMPOSITION) as any}
      />,
    );
    expect(screen.getByText("AI Text Generator")).toBeDefined();
  });

  it("does not render approve, modify, or edit controls", () => {
    render(
      <DecomposeGoalTool
        part={makePart("output-available", DECOMPOSITION) as any}
      />,
    );
    expect(screen.queryByText("Modify")).toBeNull();
    expect(screen.queryByText("Approve")).toBeNull();
    expect(screen.queryByText(/Starting in/)).toBeNull();
    expect(screen.queryByPlaceholderText("Step description")).toBeNull();
    expect(screen.queryByLabelText("Remove step")).toBeNull();
    expect(screen.queryByLabelText("Insert step here")).toBeNull();
  });

  it("does not call onSend when the plan card renders", () => {
    render(
      <DecomposeGoalTool
        part={makePart("output-available", DECOMPOSITION) as any}
      />,
    );
    expect(mockOnSend).not.toHaveBeenCalled();
  });

  it("renders nothing pending when output is not yet available", () => {
    const { container } = render(
      <DecomposeGoalTool part={makePart("input-available") as any} />,
    );
    expect(container.querySelector(".py-2")).toBeDefined();
  });
});
