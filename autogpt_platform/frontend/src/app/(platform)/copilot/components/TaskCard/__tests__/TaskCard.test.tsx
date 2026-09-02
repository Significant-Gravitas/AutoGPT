import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { TaskCard } from "../TaskCard";
import { getTaskCardMetadata, type TaskCardMetadata } from "../helpers";

const baseMeta: TaskCardMetadata = {
  taskId: "task-1",
  executionId: "exec-1",
  graphId: "graph-1",
  libraryAgentId: "lib-1",
  graphName: "Weekly Report",
  status: "DONE",
};

describe("TaskCard", () => {
  it("renders the agent name, done chip and preview", () => {
    render(<TaskCard metadata={baseMeta} preview="Sent the weekly report." />);

    expect(screen.getByText("Weekly Report")).toBeDefined();
    expect(screen.getByText("Task done")).toBeDefined();
    expect(screen.getByText("Sent the weekly report.")).toBeDefined();
  });

  it("shows a failed chip for a failed task", () => {
    render(
      <TaskCard metadata={{ ...baseMeta, status: "FAILED" }} preview="" />,
    );

    expect(screen.getByText("Task failed")).toBeDefined();
    expect(screen.queryByText("Task done")).toBeNull();
  });

  it("opens the output sheet on the run link", () => {
    render(<TaskCard metadata={baseMeta} preview="" />);

    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(
      screen.getByRole("link", { name: "Open run details" }),
    ).toBeDefined();
  });
});

describe("getTaskCardMetadata", () => {
  const payload = {
    kind: "delegated_task",
    task_id: "task-1",
    execution_id: "exec-1",
    graph_id: "graph-1",
    library_agent_id: "lib-1",
    graph_name: "Weekly Report",
    status: "DONE",
  };

  it("reads a task-outcome payload", () => {
    expect(getTaskCardMetadata(payload)).toEqual(baseMeta);
  });

  it("ignores a run-post payload so WorkCard still handles it", () => {
    expect(getTaskCardMetadata({ ...payload, kind: "expert_run" })).toBeNull();
  });

  it("ignores a payload missing the ids the card links to", () => {
    expect(getTaskCardMetadata({ ...payload, execution_id: null })).toBeNull();
    expect(getTaskCardMetadata({ ...payload, task_id: undefined })).toBeNull();
  });

  it("never reports an unreadable status as success", () => {
    expect(getTaskCardMetadata({ ...payload, status: "wat" })?.status).toBe(
      "FAILED",
    );
  });
});
