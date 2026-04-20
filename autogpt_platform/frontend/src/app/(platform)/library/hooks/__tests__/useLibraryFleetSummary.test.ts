import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useLibraryFleetSummary } from "../useLibraryFleetSummary";

const mockUseGetV1ListAllExecutions = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1ListAllExecutions: (
    ...args: Parameters<typeof mockUseGetV1ListAllExecutions>
  ) => mockUseGetV1ListAllExecutions(...args),
  getGetV1ListAllExecutionsQueryKey: () => ["list-all-executions"],
}));

vi.mock("@/hooks/useExecutionEvents", () => ({
  useExecutionEvents: vi.fn(),
}));

const mockInvalidateQueries = vi.fn();
vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>(
    "@tanstack/react-query",
  );
  return {
    ...actual,
    useQueryClient: () => ({ invalidateQueries: mockInvalidateQueries }),
  };
});

function makeAgent(overrides: Partial<LibraryAgent> = {}): LibraryAgent {
  return {
    graph_id: "graph-1",
    has_external_trigger: false,
    is_scheduled: false,
    recommended_schedule_cron: null,
    ...(overrides as object),
  } as LibraryAgent;
}

function makeExec(
  overrides: Partial<Record<keyof GraphExecutionMeta, unknown>>,
): GraphExecutionMeta {
  return {
    id: "exec-1",
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: null,
    preset_id: null,
    status: AgentExecutionStatus.COMPLETED,
    started_at: null,
    ended_at: null,
    stats: null,
    ...overrides,
  } as GraphExecutionMeta;
}

function withExecutions(executions: GraphExecutionMeta[] | undefined) {
  mockUseGetV1ListAllExecutions.mockReturnValue({
    data: executions,
    isSuccess: executions !== undefined,
  });
}

describe("useLibraryFleetSummary — monthly spend", () => {
  const FIXED_NOW = new Date("2026-04-15T12:00:00.000Z");

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    vi.setSystemTime(FIXED_NOW);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns undefined while executions query has not succeeded", () => {
    withExecutions(undefined);
    const { result } = renderHook(() => useLibraryFleetSummary([]));
    expect(result.current).toBeUndefined();
  });

  it("sums costs of executions started this month", () => {
    withExecutions([
      makeExec({
        id: "a",
        started_at: "2026-04-02T10:00:00.000Z",
        stats: { cost: 250 } as GraphExecutionMeta["stats"],
      }),
      makeExec({
        id: "b",
        started_at: "2026-04-10T10:00:00.000Z",
        stats: { cost: 75 } as GraphExecutionMeta["stats"],
      }),
    ]);

    const { result } = renderHook(() => useLibraryFleetSummary([makeAgent()]));

    expect(result.current?.monthlySpend).toBe(325);
  });

  it("excludes executions started before the current month", () => {
    withExecutions([
      makeExec({
        id: "prev-month",
        started_at: "2026-02-15T00:00:00.000Z",
        stats: { cost: 9999 } as GraphExecutionMeta["stats"],
      }),
      makeExec({
        id: "this-month",
        started_at: "2026-04-10T00:00:00.000Z",
        stats: { cost: 100 } as GraphExecutionMeta["stats"],
      }),
    ]);

    const { result } = renderHook(() => useLibraryFleetSummary([makeAgent()]));

    expect(result.current?.monthlySpend).toBe(100);
  });

  it("ignores executions with missing started_at, invalid dates, or non-numeric cost", () => {
    withExecutions([
      makeExec({ id: "no-started", started_at: null }),
      makeExec({
        id: "invalid-date",
        started_at: "not-a-date",
        stats: { cost: 50 } as GraphExecutionMeta["stats"],
      }),
      makeExec({
        id: "no-stats",
        started_at: "2026-04-05T00:00:00.000Z",
        stats: null,
      }),
      makeExec({
        id: "zero-cost",
        started_at: "2026-04-05T00:00:00.000Z",
        stats: { cost: 0 } as GraphExecutionMeta["stats"],
      }),
      makeExec({
        id: "string-cost",
        started_at: "2026-04-05T00:00:00.000Z",
        stats: { cost: "10" } as unknown as GraphExecutionMeta["stats"],
      }),
      makeExec({
        id: "valid",
        started_at: "2026-04-05T00:00:00.000Z",
        stats: { cost: 42 } as GraphExecutionMeta["stats"],
      }),
    ]);

    const { result } = renderHook(() => useLibraryFleetSummary([makeAgent()]));

    // Zero contributes but is additively neutral; string cost is dropped.
    expect(result.current?.monthlySpend).toBe(42);
  });

  it("uses UTC month boundary so late-UTC-month-end executions land in the right month", () => {
    // "Now" is 2026-04-01T00:30:00Z — 30 min into April UTC.
    vi.setSystemTime(new Date("2026-04-01T00:30:00.000Z"));

    withExecutions([
      // Execution at 2026-03-31T23:59:00Z — last minute of March UTC.
      makeExec({
        id: "march-end",
        started_at: "2026-03-31T23:59:00.000Z",
        stats: { cost: 999 } as GraphExecutionMeta["stats"],
      }),
      // Execution at 2026-04-01T00:15:00Z — first 15 minutes of April UTC.
      makeExec({
        id: "april-start",
        started_at: "2026-04-01T00:15:00.000Z",
        stats: { cost: 5 } as GraphExecutionMeta["stats"],
      }),
    ]);

    const { result } = renderHook(() => useLibraryFleetSummary([makeAgent()]));

    // Only the April execution should count, regardless of the browser's
    // local timezone (which would have classified the March one differently
    // under the old local-time boundary).
    expect(result.current?.monthlySpend).toBe(5);
  });

  it("accepts Date instances for started_at", () => {
    withExecutions([
      makeExec({
        id: "date-instance",
        started_at: new Date("2026-04-08T00:00:00.000Z"),
        stats: { cost: 17 } as GraphExecutionMeta["stats"],
      }),
    ]);

    const { result } = renderHook(() => useLibraryFleetSummary([makeAgent()]));

    expect(result.current?.monthlySpend).toBe(17);
  });
});

describe("useLibraryFleetSummary — agent bucketing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-15T12:00:00.000Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("buckets agents by execution state and trigger config", () => {
    withExecutions([
      makeExec({
        id: "running",
        graph_id: "g-running",
        status: AgentExecutionStatus.RUNNING,
      }),
      makeExec({
        id: "failed",
        graph_id: "g-failed",
        status: AgentExecutionStatus.FAILED,
        ended_at: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
      }),
      makeExec({
        id: "completed",
        graph_id: "g-completed",
        status: AgentExecutionStatus.COMPLETED,
        ended_at: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
      }),
    ]);

    const agents: LibraryAgent[] = [
      makeAgent({ graph_id: "g-running" }),
      makeAgent({ graph_id: "g-failed" }),
      makeAgent({ graph_id: "g-completed" }),
      makeAgent({ graph_id: "g-listening", has_external_trigger: true }),
      makeAgent({ graph_id: "g-scheduled", is_scheduled: true }),
      makeAgent({ graph_id: "g-idle" }),
    ];

    const { result } = renderHook(() => useLibraryFleetSummary(agents));

    expect(result.current).toMatchObject({
      running: 1,
      error: 1,
      completed: 1,
      listening: 1,
      scheduled: 1,
      idle: 2,
    });
  });
});
