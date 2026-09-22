import React from "react";
import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TaskProgressBar } from "../TaskProgressBar";
import type { TodoItem } from "../helpers";

vi.mock("framer-motion", () => {
  const MOTION_PROPS = [
    "initial",
    "animate",
    "exit",
    "transition",
    "layout",
    "layoutId",
    "whileHover",
    "whileTap",
    "style",
  ];
  function makeMotion(Tag: string) {
    return React.forwardRef(function MotionComponent(
      props: Record<string, unknown>,
      ref: React.Ref<unknown>,
    ) {
      const rest: Record<string, unknown> = {};
      let children: React.ReactNode = null;
      for (const [key, value] of Object.entries(props)) {
        if (key === "children") children = value as React.ReactNode;
        else if (!MOTION_PROPS.includes(key)) rest[key] = value;
      }
      return React.createElement(Tag, { ref, ...rest }, children);
    });
  }
  return {
    AnimatePresence: ({ children }: { children: React.ReactNode }) => (
      <>{children}</>
    ),
    useReducedMotion: () => false,
    motion: {
      div: makeMotion("div"),
      span: makeMotion("span"),
    },
  };
});

function todo(
  content: string,
  status: TodoItem["status"],
  activeForm?: string,
): TodoItem {
  return { content, status, activeForm };
}

afterEach(cleanup);

describe("TaskProgressBar", () => {
  it("renders nothing when there are no todos", () => {
    const { container } = render(<TaskProgressBar todos={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the current task and step count when collapsed", () => {
    render(
      <TaskProgressBar
        todos={[
          todo("Step 1", "completed"),
          todo("Step 2", "in_progress", "Working on step 2"),
          todo("Step 3", "pending"),
        ]}
        isStreaming
      />,
    );
    expect(screen.getByText("Working on step 2")).toBeDefined();
    expect(screen.getByText("2/3")).toBeDefined();
  });

  it('shows "All tasks complete" when every task is done', () => {
    render(
      <TaskProgressBar
        todos={[todo("Step 1", "completed"), todo("Step 2", "completed")]}
      />,
    );
    expect(screen.getByText("All tasks complete")).toBeDefined();
    expect(screen.getByText("2/2")).toBeDefined();
  });

  it("expands to reveal every task row when the header is clicked", () => {
    render(
      <TaskProgressBar
        todos={[
          todo("First task", "completed"),
          todo("Second task", "pending"),
        ]}
        defaultExpanded={false}
      />,
    );
    const header = screen.getByRole("button", { expanded: false });
    fireEvent.click(header);
    expect(screen.getByRole("button", { expanded: true })).toBeDefined();
    expect(screen.getByText("First task")).toBeDefined();
    expect(screen.getByText("Second task")).toBeDefined();
  });

  it("marks an interrupted in-progress task as stopped when not streaming", () => {
    render(
      <TaskProgressBar
        todos={[todo("Halted step", "in_progress", "Running step")]}
        isStreaming={false}
        defaultExpanded
      />,
    );
    expect(screen.getAllByLabelText("stopped").length).toBeGreaterThan(0);
    // A stopped task shows its plain content, not the running activeForm.
    expect(screen.getByText("Halted step")).toBeDefined();
  });

  it("shows a spinner for the in-progress task while streaming", () => {
    render(
      <TaskProgressBar
        todos={[todo("Running step", "in_progress", "Running step")]}
        isStreaming
        defaultExpanded
      />,
    );
    expect(screen.getAllByLabelText("in progress").length).toBeGreaterThan(0);
  });
});
