import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { DelegationDock } from "../DelegationDock";
import { getDockLine } from "../helpers";

function delegation(id: string, output: Record<string, unknown>): UIMessage {
  return {
    id: `m-${id}`,
    role: "assistant",
    parts: [
      {
        type: "tool-delegate_to_expert",
        state: "output-available",
        toolCallId: id,
        input: { expert_id: "exp-alex", prompt: "Draft the PRD" },
        output,
      },
    ],
  } as unknown as UIMessage;
}

describe("DelegationDock", () => {
  beforeEach(() => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, isOpen: false, activeTab: "files" },
    }));
  });
  afterEach(cleanup);

  it("renders nothing without hand-offs", () => {
    render(<DelegationDock messages={[]} />);
    expect(screen.queryByTestId("delegation-dock")).toBeNull();
  });

  it("says how many experts are working and opens the Work tab", () => {
    render(
      <DelegationDock
        messages={[
          delegation("c1", { status: "running" }),
          delegation("c2", { status: "queued" }),
        ]}
      />,
    );
    const dock = screen.getByTestId("delegation-dock");
    expect(dock.textContent).toContain("1 expert working · 1 queued");

    fireEvent.click(dock);
    const panel = useCopilotUIStore.getState().artifactPanel;
    expect(panel.isOpen).toBe(true);
    expect(panel.activeTab).toBe("work");
  });

  it("yields the slot to a running task list", () => {
    render(
      <DelegationDock
        messages={[delegation("c1", { status: "running" })]}
        hasActiveTaskList
      />,
    );
    expect(screen.queryByTestId("delegation-dock")).toBeNull();
  });

  it("stays quiet once every hand-off has reported back", () => {
    render(
      <DelegationDock messages={[delegation("c1", { status: "completed" })]} />,
    );
    expect(screen.queryByTestId("delegation-dock")).toBeNull();
  });
});

describe("getDockLine", () => {
  const base = {
    toolCallId: "c1",
    tool: "delegate_to_expert" as const,
    expertId: "exp-alex",
    expert: null,
    prompt: null,
    subSessionId: "sub-1",
    link: null,
    elapsedSeconds: null,
    response: null,
    error: null,
    files: [],
    reviewId: null,
  };

  it("counts a teammate's question as needing you", () => {
    const line = getDockLine([{ ...base, status: "running" }], {
      c1: "needs-input",
    });
    expect(line).toEqual({ text: "1 expert needs you", tone: "waiting" });
  });

  it("puts approvals first and keeps the working count", () => {
    const line = getDockLine(
      [
        { ...base, toolCallId: "c1", status: "proposed" },
        { ...base, toolCallId: "c2", status: "running" },
      ],
      {},
    );
    expect(line?.text).toBe(
      "1 hand-off waiting for your approval · 1 expert working",
    );
  });

  it("trusts the live status over the frozen one", () => {
    expect(
      getDockLine([{ ...base, status: "running" }], { c1: "completed" }),
    ).toBeNull();
  });
});
