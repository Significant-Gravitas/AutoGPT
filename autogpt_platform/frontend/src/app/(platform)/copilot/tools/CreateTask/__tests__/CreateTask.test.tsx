import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { ToolResult } from "../../../components/ToolChain/ToolResult";
import type { ChainRow } from "../../../components/ToolChain/helpers";

function createTaskRow(output: unknown): ChainRow {
  return {
    key: "call-1",
    category: "platform",
    text: "Created task",
    state: "done",
    tool: "create_task",
    input: { title: "Launch pricing page" },
    output,
  } as unknown as ChainRow;
}

const CREATED = {
  type: "task_created",
  message: "Created the task.",
  task: {
    id: "task-7",
    title: "Launch pricing page",
    status: "TODO",
    owner_expert_id: null,
    created_by: "OTTO",
    session_ids: [],
    updated_at: "2026-11-10T09:00:00Z",
  },
};

describe("create_task result card", () => {
  it("shows the created task's title with a link that opens it", () => {
    render(<ToolResult row={createTaskRow(CREATED)} />);

    expect(screen.getByText("Launch pricing page")).toBeDefined();
    expect(screen.getByText(/Created task:/)).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Open task Launch pricing page" })
        .getAttribute("href"),
    ).toBe("/tasks?task=task-7");
    expect(screen.queryByText(/task_created/)).toBeNull();
  });

  it("reads a result that arrives as JSON text", () => {
    render(<ToolResult row={createTaskRow(JSON.stringify(CREATED))} />);

    expect(
      screen.getByRole("link", { name: "Open task Launch pricing page" }),
    ).toBeDefined();
  });

  it("shows no task card for an error result", () => {
    render(
      <ToolResult
        row={createTaskRow({ type: "error", message: "Could not create it." })}
      />,
    );

    expect(screen.queryByRole("link", { name: /Open task/ })).toBeNull();
    expect(screen.getByText(/Could not create it\./)).toBeDefined();
  });
});
