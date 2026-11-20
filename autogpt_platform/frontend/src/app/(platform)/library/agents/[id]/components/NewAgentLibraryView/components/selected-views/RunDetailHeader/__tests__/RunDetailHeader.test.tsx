import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { act, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";

import { RunDetailHeader } from "../RunDetailHeader";

const AGENT = {
  id: "lib-1",
  name: "Weather report",
} as unknown as LibraryAgent;

function makeRun(overrides: Partial<GraphExecution> = {}): GraphExecution {
  return {
    id: "run-1",
    graph_version: 1,
    status: AgentExecutionStatus.RUNNING,
    started_at: "2026-07-21T12:00:00Z",
    ended_at: null,
    stats: null,
    ...overrides,
  } as unknown as GraphExecution;
}

describe("RunDetailHeader", () => {
  it("counts a running run's duration up each second", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-21T12:00:10Z"));
    try {
      render(<RunDetailHeader agent={AGENT} run={makeRun()} />);
      expect(screen.getByText("Duration: 10 seconds")).toBeDefined();

      act(() => {
        vi.advanceTimersByTime(6_000);
      });

      expect(screen.getByText("Duration: 16 seconds")).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows no duration for a finished run with no stats.duration or ended_at", () => {
    render(
      <RunDetailHeader
        agent={AGENT}
        run={makeRun({ status: AgentExecutionStatus.TERMINATED })}
      />,
    );

    expect(screen.getByText("Weather report")).toBeDefined();
    expect(screen.queryByText(/^Duration:/)).toBeNull();
  });
});
