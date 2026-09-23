import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";

import type { AgentExecutionWithInfo } from "../../helpers";
import { ActivityItem } from "../ActivityItem";

function makeExecution(
  overrides: Partial<AgentExecutionWithInfo> = {},
): AgentExecutionWithInfo {
  return {
    id: "exec-1",
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    status: AgentExecutionStatus.COMPLETED,
    started_at: "2026-07-21T12:00:00Z",
    ended_at: "2026-07-21T12:01:00Z",
    agent_name: "Weather Agent",
    agent_description: "Reports the weather",
    library_agent_id: "lib-agent-1",
    ...overrides,
  } as unknown as AgentExecutionWithInfo;
}

describe("ActivityItem", () => {
  it("renders a linked item with a nav indicator in the new layout", () => {
    render(<ActivityItem execution={makeExecution()} newLayout />);

    const link = screen.getByRole("button");
    expect(link.getAttribute("href")).toContain("/library/agents/lib-agent-1");
    expect(link.getAttribute("href")).toContain("activeTab=runs");
    expect(screen.getByText("Weather Agent")).toBeDefined();
  });

  it("routes review executions to the reviews tab", () => {
    render(
      <ActivityItem
        execution={makeExecution({ status: AgentExecutionStatus.REVIEW })}
        newLayout
      />,
    );

    expect(screen.getByRole("button").getAttribute("href")).toContain(
      "activeTab=reviews",
    );
  });

  it("renders a running execution in the classic layout as a link", () => {
    render(
      <ActivityItem
        execution={makeExecution({
          status: AgentExecutionStatus.RUNNING,
          ended_at: null,
        })}
      />,
    );

    expect(screen.getByRole("button")).toBeDefined();
    expect(screen.getByText("Weather Agent")).toBeDefined();
  });

  it("renders a non-linked item when the execution has no library agent", () => {
    render(
      <ActivityItem
        execution={makeExecution({ library_agent_id: undefined })}
      />,
    );

    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByText("Weather Agent")).toBeDefined();
  });
});
