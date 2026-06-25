import { beforeEach, describe, expect, it, vi } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";

// Mock withRoleAccess to bypass server-side auth
vi.mock("@/lib/withRoleAccess", () => ({
  withRoleAccess: () =>
    Promise.resolve((Component: React.ComponentType) =>
      Promise.resolve(Component),
    ),
}));

// `vi.hoisted` so the shared stubs are initialised before the module
// factory below is evaluated. Individual tests reassign the return
// values; reset in `beforeEach` so tests don't bleed state into each
// other.
const api = vi.hoisted(() => {
  const defaultQuery = () => ({
    data: undefined as unknown,
    isLoading: false,
    isError: false,
    error: null as unknown,
    refetch: () => {},
  });
  const defaultMutation = () => ({
    mutateAsync: async () => {},
    isPending: false,
  });
  return {
    useGetV2GetExecutionDiagnostics: vi.fn(defaultQuery),
    useGetV2GetAgentDiagnostics: vi.fn(defaultQuery),
    useGetV2GetScheduleDiagnostics: vi.fn(defaultQuery),
    useGetV2ListRunningExecutions: vi.fn(defaultQuery),
    useGetV2ListOrphanedExecutions: vi.fn(defaultQuery),
    useGetV2ListFailedExecutions: vi.fn(defaultQuery),
    useGetV2ListLongRunningExecutions: vi.fn(defaultQuery),
    useGetV2ListStuckQueuedExecutions: vi.fn(defaultQuery),
    useGetV2ListInvalidExecutions: vi.fn(defaultQuery),
    usePostV2StopSingleExecution: vi.fn(defaultMutation),
    usePostV2StopMultipleExecutions: vi.fn(defaultMutation),
    usePostV2StopAllLongRunningExecutions: vi.fn(defaultMutation),
    usePostV2CleanupOrphanedExecutions: vi.fn(defaultMutation),
    usePostV2CleanupAllOrphanedExecutions: vi.fn(defaultMutation),
    usePostV2CleanupAllStuckQueuedExecutions: vi.fn(defaultMutation),
    usePostV2RequeueStuckExecution: vi.fn(defaultMutation),
    usePostV2RequeueMultipleStuckExecutions: vi.fn(defaultMutation),
    usePostV2RequeueAllStuckQueuedExecutions: vi.fn(defaultMutation),
    useGetV2ListAllUserSchedules: vi.fn(defaultQuery),
    useGetV2ListOrphanedSchedules: vi.fn(defaultQuery),
    usePostV2CleanupOrphanedSchedules: vi.fn(defaultMutation),
    defaultQuery,
    defaultMutation,
  };
});

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  useGetV2GetExecutionDiagnostics: api.useGetV2GetExecutionDiagnostics,
  useGetV2GetAgentDiagnostics: api.useGetV2GetAgentDiagnostics,
  useGetV2GetScheduleDiagnostics: api.useGetV2GetScheduleDiagnostics,
  useGetV2ListRunningExecutions: api.useGetV2ListRunningExecutions,
  useGetV2ListOrphanedExecutions: api.useGetV2ListOrphanedExecutions,
  useGetV2ListFailedExecutions: api.useGetV2ListFailedExecutions,
  useGetV2ListLongRunningExecutions: api.useGetV2ListLongRunningExecutions,
  useGetV2ListStuckQueuedExecutions: api.useGetV2ListStuckQueuedExecutions,
  useGetV2ListInvalidExecutions: api.useGetV2ListInvalidExecutions,
  usePostV2StopSingleExecution: api.usePostV2StopSingleExecution,
  usePostV2StopMultipleExecutions: api.usePostV2StopMultipleExecutions,
  usePostV2StopAllLongRunningExecutions:
    api.usePostV2StopAllLongRunningExecutions,
  usePostV2CleanupOrphanedExecutions: api.usePostV2CleanupOrphanedExecutions,
  usePostV2CleanupAllOrphanedExecutions:
    api.usePostV2CleanupAllOrphanedExecutions,
  usePostV2CleanupAllStuckQueuedExecutions:
    api.usePostV2CleanupAllStuckQueuedExecutions,
  usePostV2RequeueStuckExecution: api.usePostV2RequeueStuckExecution,
  usePostV2RequeueMultipleStuckExecutions:
    api.usePostV2RequeueMultipleStuckExecutions,
  usePostV2RequeueAllStuckQueuedExecutions:
    api.usePostV2RequeueAllStuckQueuedExecutions,
  useGetV2ListAllUserSchedules: api.useGetV2ListAllUserSchedules,
  useGetV2ListOrphanedSchedules: api.useGetV2ListOrphanedSchedules,
  usePostV2CleanupOrphanedSchedules: api.usePostV2CleanupOrphanedSchedules,
}));

// Import the inner component directly since the page is async/server
import { DiagnosticsContent } from "../components/DiagnosticsContent";

function query<T>(data: T) {
  return {
    data: { data },
    isLoading: false,
    isError: false,
    error: null,
    refetch: () => {},
  };
}

function emptyExecutionData() {
  // All-zeros snapshot that still populates every numeric field the UI
  // reads — lets the happy-path render without tripping on undefined.
  return {
    timestamp: "2026-04-22T12:00:00Z",
    running_executions: 0,
    queued_executions_db: 0,
    queued_executions_rabbitmq: 0,
    cancel_queue_depth: 0,
    stuck_queued_1h: 0,
    stuck_running_24h: 0,
    orphaned_running: 0,
    orphaned_queued: 0,
    failed_count_24h: 0,
    failed_count_1h: 0,
    failure_rate_24h: 0,
    completed_24h: 0,
    completed_1h: 0,
    throughput_per_hour: 0,
    invalid_queued_with_start: 0,
    invalid_running_without_start: 0,
    oldest_running_hours: null,
  };
}

describe("AdminDiagnosticsPage", () => {
  beforeEach(() => {
    for (const fn of Object.values(api)) {
      if (typeof fn === "function" && "mockImplementation" in fn) {
        (fn as ReturnType<typeof vi.fn>).mockImplementation(api.defaultQuery);
      }
    }
  });

  it("renders DiagnosticsContent in loading state", () => {
    api.useGetV2GetExecutionDiagnostics.mockImplementation(() => ({
      ...api.defaultQuery(),
      isLoading: true,
    }));
    render(<DiagnosticsContent />);
    expect(screen.getByText("Loading diagnostics...")).toBeDefined();
  });

  it("renders the ErrorCard when a diagnostics query fails", () => {
    api.useGetV2GetExecutionDiagnostics.mockImplementation(() => ({
      ...api.defaultQuery(),
      isError: true,
      error: { status: 500, message: "boom" },
    }));
    render(<DiagnosticsContent />);
    // ErrorCard is the shared molecule; it renders "Something went
    // wrong" as its header. Assert on that so this test doesn't break
    // if the card adopts proper a11y roles later.
    expect(screen.getByText(/something went wrong/i)).toBeDefined();
  });

  it("renders the dashboard with zeroed execution data (no alert cards)", () => {
    api.useGetV2GetExecutionDiagnostics.mockImplementation(() =>
      query(emptyExecutionData()),
    );
    api.useGetV2GetAgentDiagnostics.mockImplementation(() =>
      query({ total_agents: 0 }),
    );
    api.useGetV2GetScheduleDiagnostics.mockImplementation(() =>
      query({
        user_schedules: 0,
        total_orphaned: 0,
        orphaned_deleted_graph: 0,
        orphaned_no_library_access: 0,
        total_runs_next_hour: 0,
        schedules_next_hour: 0,
      }),
    );

    render(<DiagnosticsContent />);

    // Dashboard header is the canonical "we rendered something" signal.
    expect(screen.getByText("System Diagnostics")).toBeDefined();
    expect(screen.getByRole("button", { name: /refresh/i })).toBeDefined();
    // ``Click to view →`` text appears only inside an alert card; its
    // absence confirms the conditional branches stayed off. (The
    // card titles themselves also appear in the always-rendered legend,
    // so asserting on them would produce false positives.)
    expect(screen.queryByText(/Click to view/i)).toBeNull();
  });

  it("renders alert cards when there are critical issues", () => {
    api.useGetV2GetExecutionDiagnostics.mockImplementation(() =>
      query({
        ...emptyExecutionData(),
        orphaned_running: 3,
        orphaned_queued: 2,
        failed_count_24h: 11,
        failed_count_1h: 2,
        failure_rate_24h: 0.45,
        stuck_running_24h: 4,
        oldest_running_hours: 36,
        invalid_queued_with_start: 1,
        invalid_running_without_start: 1,
      }),
    );
    api.useGetV2GetScheduleDiagnostics.mockImplementation(() =>
      query({
        user_schedules: 10,
        total_orphaned: 2,
        orphaned_deleted_graph: 1,
        orphaned_no_library_access: 1,
        total_runs_next_hour: 0,
        schedules_next_hour: 0,
      }),
    );

    render(<DiagnosticsContent />);

    expect(screen.getByText("Orphaned Executions")).toBeDefined();
    expect(screen.getByText("Failed Executions (24h)")).toBeDefined();
    expect(screen.getByText("Long-Running Executions")).toBeDefined();
    expect(screen.getByText("Orphaned Schedules")).toBeDefined();
    expect(screen.getByText("Invalid States (Data Corruption)")).toBeDefined();
  });
});
