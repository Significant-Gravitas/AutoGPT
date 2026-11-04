import type { ToolUIPart } from "ai";
import { describe, expect, it } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { CreateTaskTool } from "../CreateTask";
import { isCreatedTaskPart } from "../helpers";

function part(state: string, output?: unknown) {
  return {
    type: "tool-create_task",
    toolCallId: "call-1",
    state,
    input: { title: "Launch pricing page" },
    output,
  } as unknown as ToolUIPart;
}

const created = {
  type: "task_updated",
  message: "Created T-7.",
  task: { id: "task-7", number: 7, title: "Launch pricing page" },
};

describe("CreateTaskTool", () => {
  it("shows a pill with the title and a link to the task", () => {
    render(<CreateTaskTool part={part("output-available", created)} />);

    expect(screen.getByText("Launch pricing page")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Open task Launch pricing page" }),
    ).toHaveProperty("href", expect.stringContaining("/tasks?task=task-7"));
    expect(screen.queryByText(/T-7|task_updated/)).toBeNull();
  });

  it("leaves running and failed calls in the chain", () => {
    expect(isCreatedTaskPart(part("output-available", created))).toBe(true);
    expect(isCreatedTaskPart(part("input-available"))).toBe(false);
    expect(
      isCreatedTaskPart(
        part("output-available", { type: "error", message: "nope" }),
      ),
    ).toBe(false);
  });
});
