import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import type { SitrepItemData } from "@/app/(platform)/library/types";
import { RecentWorkflowRuns } from "../RecentWorkflowRuns";

function makeRun(overrides: Partial<SitrepItemData> = {}): SitrepItemData {
  return {
    id: "run-1",
    agentID: "agent-1",
    agentName: "Test Agent",
    status: "running",
    priority: "running",
    message: "Doing work…",
    ...overrides,
  };
}

describe("RecentWorkflowRuns", () => {
  test("renders nothing when there are no runs", () => {
    const { container } = render(<RecentWorkflowRuns runs={[]} />);
    expect(container.innerHTML).toBe("");
  });

  test("renders run names and messages", () => {
    render(
      <RecentWorkflowRuns
        runs={[
          makeRun({
            id: "1",
            agentName: "Alpha Bot",
            message: "Running task A",
          }),
          makeRun({
            id: "2",
            agentName: "Beta Bot",
            message: "Running task B",
          }),
        ]}
      />,
    );

    expect(screen.getByText("Alpha Bot")).toBeDefined();
    expect(screen.getByText("Running task A")).toBeDefined();
    expect(screen.getByText("Beta Bot")).toBeDefined();
    expect(screen.getByText("Running task B")).toBeDefined();
  });

  test("names the section and links to the library", () => {
    render(<RecentWorkflowRuns runs={[makeRun()]} />);

    expect(screen.getByText("Recent workflow runs")).toBeDefined();
    expect(
      screen.getByRole("link", { name: /View all/ }).getAttribute("href"),
    ).toBe("/library");
  });

  test("shows a Completed badge for a successful run", () => {
    render(
      <RecentWorkflowRuns
        runs={[makeRun({ priority: "success", status: "idle" })]}
      />,
    );

    expect(screen.getByText("Completed")).toBeDefined();
  });

  test("links Ask to the copilot with a prompt about the run", () => {
    render(
      <RecentWorkflowRuns
        runs={[
          makeRun({
            agentName: "Error Agent",
            status: "error",
            priority: "error",
          }),
        ]}
      />,
    );

    const href = screen.getByRole("link", { name: /Ask/ }).getAttribute("href");
    expect(href).toBe(
      `/copilot?autosubmit=true#prompt=${encodeURIComponent(
        "What happened with Error Agent? It has an error — can you check?",
      )}`,
    );
  });

  test("asks for a summary of a completed run", () => {
    render(
      <RecentWorkflowRuns
        runs={[
          makeRun({
            agentName: "Done Agent",
            priority: "success",
            status: "idle",
          }),
        ]}
      />,
    );

    const href = screen.getByRole("link", { name: /Ask/ }).getAttribute("href");
    expect(decodeURIComponent(href ?? "")).toContain(
      "Done Agent just finished a run — can you summarize what it did?",
    );
  });

  test("links See to the agent's library page", () => {
    render(<RecentWorkflowRuns runs={[makeRun({ agentID: "agent-xyz" })]} />);

    expect(screen.getByRole("link", { name: /See/ }).getAttribute("href")).toBe(
      "/library/agents/agent-xyz",
    );
  });
});
