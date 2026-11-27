import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { act, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";

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
  it("renders a linked item with a nav indicator", () => {
    render(<ActivityItem execution={makeExecution()} />);

    const link = screen.getByRole("link");
    expect(screen.queryByRole("button")).toBeNull();
    expect(link.getAttribute("href")).toContain("/library/agents/lib-agent-1");
    expect(link.getAttribute("href")).toContain("activeTab=runs");
    expect(screen.getByText("Weather Agent")).toBeDefined();
  });

  it("routes review executions to the reviews tab", () => {
    render(
      <ActivityItem
        execution={makeExecution({ status: AgentExecutionStatus.REVIEW })}
      />,
    );

    expect(screen.getByRole("link").getAttribute("href")).toContain(
      "activeTab=reviews",
    );
  });

  it("renders a running execution as a link", () => {
    render(
      <ActivityItem
        execution={makeExecution({
          status: AgentExecutionStatus.RUNNING,
          ended_at: null,
        })}
      />,
    );

    expect(screen.getByRole("link")).toBeDefined();
    expect(screen.getByText("Weather Agent")).toBeDefined();
  });

  it("counts a running execution's duration up each second", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-21T12:00:10Z"));
    try {
      render(
        <ActivityItem
          execution={makeExecution({
            status: AgentExecutionStatus.RUNNING,
            ended_at: null,
          })}
        />,
      );
      expect(screen.getByText(/, 10s running$/)).toBeDefined();
      act(() => {
        vi.advanceTimersByTime(6_000);
      });
      expect(screen.getByText(/, 16s running$/)).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });

  it.each([
    [AgentExecutionStatus.RUNNING, /, 16s running$/],
    [AgentExecutionStatus.QUEUED, /, 16s queued$/],
  ])("counts a %s run up to now, past a stale ended_at", (status, line) => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-21T12:00:16Z"));
    try {
      render(
        <ActivityItem
          execution={makeExecution({
            status,
            ended_at: new Date("2026-07-21T12:00:03Z"),
          })}
        />,
      );
      expect(screen.getByText(line)).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows a few seconds, not Unknown, when a running execution started after the floored now", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-21T12:00:00.400Z"));
    try {
      render(
        <ActivityItem
          execution={makeExecution({
            status: AgentExecutionStatus.RUNNING,
            started_at: new Date("2026-07-21T12:00:00.400Z"),
            ended_at: null,
          })}
        />,
      );
      expect(screen.queryByText(/Unknown/)).toBeNull();
      expect(screen.getByText(/, a few seconds running$/)).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });

  it("says a completed run with a failed step completed with errors", () => {
    render(
      <ActivityItem
        execution={makeExecution({ stats: { node_error_count: 1 } })}
      />,
    );

    expect(screen.getByText(/^Completed with errors /)).toBeDefined();
  });

  it("says a clean completed run completed", () => {
    render(
      <ActivityItem
        execution={makeExecution({ stats: { node_error_count: 0 } })}
      />,
    );

    expect(screen.getByText(/^Completed /)).toBeDefined();
    expect(screen.queryByText(/with errors/)).toBeNull();
  });

  it.each([
    [AgentExecutionStatus.QUEUED, /^Started .* queued$/],
    [AgentExecutionStatus.FAILED, /^Failed /],
    [AgentExecutionStatus.TERMINATED, /^Stopped /],
  ])("says what a %s run did", (status, line) => {
    render(<ActivityItem execution={makeExecution({ status })} />);

    expect(screen.getByText(line)).toBeDefined();
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
