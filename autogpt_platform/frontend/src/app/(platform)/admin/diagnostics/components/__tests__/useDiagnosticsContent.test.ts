import { describe, expect, it, vi } from "vitest";

import { renderHook } from "@testing-library/react";

// Mock the three generated API hooks this hook composes. The hook under
// test is intentionally thin — it forwards loading/error from the three
// children and coalesces them. Stubbing at the module boundary lets us
// exercise the combine-logic without spinning up MSW.
const mockExecutions = vi.fn();
const mockAgents = vi.fn();
const mockSchedules = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  useGetV2GetExecutionDiagnostics: (...args: unknown[]) =>
    mockExecutions(...args),
  useGetV2GetAgentDiagnostics: (...args: unknown[]) => mockAgents(...args),
  useGetV2GetScheduleDiagnostics: (...args: unknown[]) =>
    mockSchedules(...args),
}));

import { useDiagnosticsContent } from "../useDiagnosticsContent";

function stub(overrides: Record<string, unknown> = {}) {
  return {
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
    ...overrides,
  };
}

describe("useDiagnosticsContent", () => {
  it("is loading when any of the child queries is loading", () => {
    mockExecutions.mockReturnValue(stub({ isLoading: true }));
    mockAgents.mockReturnValue(stub());
    mockSchedules.mockReturnValue(stub());

    const { result } = renderHook(() => useDiagnosticsContent());
    expect(result.current.isLoading).toBe(true);
    expect(result.current.isError).toBe(false);
  });

  it("is in error state when any of the child queries errored", () => {
    const err = new Error("schedule boom");
    mockExecutions.mockReturnValue(stub());
    mockAgents.mockReturnValue(stub());
    mockSchedules.mockReturnValue(stub({ isError: true, error: err }));

    const { result } = renderHook(() => useDiagnosticsContent());
    expect(result.current.isError).toBe(true);
    expect(result.current.error).toBe(err);
  });

  it("unwraps each response's data field into a dedicated return key", () => {
    mockExecutions.mockReturnValue(
      stub({ data: { data: { running_executions: 7 } } }),
    );
    mockAgents.mockReturnValue(stub({ data: { data: { total_agents: 3 } } }));
    mockSchedules.mockReturnValue(
      stub({ data: { data: { user_schedules: 9 } } }),
    );

    const { result } = renderHook(() => useDiagnosticsContent());
    expect(result.current.executionData).toEqual({ running_executions: 7 });
    expect(result.current.agentData).toEqual({ total_agents: 3 });
    expect(result.current.scheduleData).toEqual({ user_schedules: 9 });
  });

  it("refresh() invokes refetch on all three child queries", () => {
    const refetchEx = vi.fn();
    const refetchAg = vi.fn();
    const refetchSc = vi.fn();
    mockExecutions.mockReturnValue(stub({ refetch: refetchEx }));
    mockAgents.mockReturnValue(stub({ refetch: refetchAg }));
    mockSchedules.mockReturnValue(stub({ refetch: refetchSc }));

    const { result } = renderHook(() => useDiagnosticsContent());
    result.current.refresh();

    expect(refetchEx).toHaveBeenCalledTimes(1);
    expect(refetchAg).toHaveBeenCalledTimes(1);
    expect(refetchSc).toHaveBeenCalledTimes(1);
  });
});
