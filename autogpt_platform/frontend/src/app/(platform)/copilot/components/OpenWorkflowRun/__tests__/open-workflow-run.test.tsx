import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { useWorkAreaStore } from "../../../workArea";
import { OpenWorkflowRun } from "../OpenWorkflowRun";

beforeEach(() => {
  useWorkAreaStore.getState().reset();
});

describe("See run under a workflow card", () => {
  it("opens the run in a workflow tab, in front", async () => {
    render(
      <OpenWorkflowRun
        graphID="graph-1"
        executionID="run-1"
        title="Morning digest"
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "See run" }));

    const { tabs, activeKey } = useWorkAreaStore.getState();
    expect(tabs).toEqual([
      {
        key: "workflow:graph-1",
        kind: "workflow",
        graphID: "graph-1",
        graphVersion: null,
        executionID: "run-1",
        title: "Morning digest",
      },
    ]);
    expect(activeKey).toBe("workflow:graph-1");
  });

  it("moves an open workflow tab to the run asked for", async () => {
    useWorkAreaStore.getState().openWorkflowRun({
      graphID: "graph-1",
      executionID: "run-1",
      title: "Morning digest",
    });
    render(
      <OpenWorkflowRun
        graphID="graph-1"
        executionID="run-2"
        title="Morning digest"
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: "See run" }));

    const { tabs } = useWorkAreaStore.getState();
    expect(tabs).toHaveLength(1);
    expect(tabs[0]).toMatchObject({ executionID: "run-2" });
  });
});
