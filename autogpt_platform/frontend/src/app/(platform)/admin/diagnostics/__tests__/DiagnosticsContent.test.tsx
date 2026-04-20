import {
  render,
  screen,
  cleanup,
  fireEvent,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DiagnosticsContent } from "../components/DiagnosticsContent";

// Mock the generated API hooks directly so useDiagnosticsContent code is exercised
const mockExecQuery = vi.fn();
const mockAgentQuery = vi.fn();
const mockScheduleQuery = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  useGetV2GetExecutionDiagnostics: () => mockExecQuery(),
  useGetV2GetAgentDiagnostics: () => mockAgentQuery(),
  useGetV2GetScheduleDiagnostics: () => mockScheduleQuery(),
  useGetV2ListRunningExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListOrphanedExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListFailedExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListLongRunningExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListStuckQueuedExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListInvalidExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  usePostV2StopSingleExecution: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2StopMultipleExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2StopAllLongRunningExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2CleanupOrphanedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2CleanupAllOrphanedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2CleanupAllStuckQueuedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2RequeueStuckExecution: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2RequeueMultipleStuckExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2RequeueAllStuckQueuedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  useGetV2ListAllUserSchedules: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListOrphanedSchedules: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  usePostV2CleanupOrphanedSchedules: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

afterEach(() => {
  cleanup();
  mockExecQuery.mockReset();
  mockAgentQuery.mockReset();
  mockScheduleQuery.mockReset();
});

const executionData = {
  running_executions: 10,
  queued_executions_db: 5,
  queued_executions_rabbitmq: 3,
  cancel_queue_depth: 0,
  orphaned_running: 2,
  orphaned_queued: 1,
  failed_count_1h: 5,
  failed_count_24h: 20,
  failure_rate_24h: 0.83,
  stuck_running_24h: 3,
  stuck_running_1h: 5,
  oldest_running_hours: 26.5,
  stuck_queued_1h: 2,
  queued_never_started: 1,
  invalid_queued_with_start: 1,
  invalid_running_without_start: 1,
  completed_1h: 50,
  completed_24h: 1200,
  throughput_per_hour: 50.0,
  timestamp: "2026-04-17T00:00:00Z",
};

const agentData = {
  agents_with_active_executions: 7,
  timestamp: "2026-04-17T00:00:00Z",
};

const scheduleData = {
  total_schedules: 15,
  user_schedules: 10,
  system_schedules: 5,
  orphaned_deleted_graph: 2,
  orphaned_no_library_access: 1,
  orphaned_invalid_credentials: 0,
  orphaned_validation_failed: 0,
  total_orphaned: 3,
  schedules_next_hour: 4,
  schedules_next_24h: 8,
  total_runs_next_hour: 12,
  total_runs_next_24h: 48,
  timestamp: "2026-04-17T00:00:00Z",
};

function setupLoadedMocks() {
  mockExecQuery.mockReturnValue({
    data: { data: executionData },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  mockAgentQuery.mockReturnValue({
    data: { data: agentData },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  mockScheduleQuery.mockReturnValue({
    data: { data: scheduleData },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
}

function setupLoadingMocks() {
  mockExecQuery.mockReturnValue({
    data: undefined,
    isLoading: true,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  mockAgentQuery.mockReturnValue({
    data: undefined,
    isLoading: true,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  mockScheduleQuery.mockReturnValue({
    data: undefined,
    isLoading: true,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
}

function setupErrorMocks() {
  mockExecQuery.mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: true,
    error: { status: 500, message: "Server error" },
    refetch: vi.fn(),
  });
  mockAgentQuery.mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
  mockScheduleQuery.mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
  });
}

describe("DiagnosticsContent", () => {
  it("shows loading state", () => {
    setupLoadingMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("Loading diagnostics...")).toBeDefined();
  });

  it("shows error state with retry", () => {
    setupErrorMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("Try Again")).toBeDefined();
  });

  it("renders system diagnostics heading with data", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("System Diagnostics")).toBeDefined();
    expect(screen.getByText("Refresh")).toBeDefined();
  });

  it("renders execution queue status cards", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("Execution Queue Status")).toBeDefined();
    expect(screen.getByText("Running Executions")).toBeDefined();
    expect(screen.getByText("Queued in Database")).toBeDefined();
    expect(screen.getByText("Queued in RabbitMQ")).toBeDefined();
  });

  it("renders throughput metrics", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("System Throughput")).toBeDefined();
    expect(screen.getByText("Completed (24h)")).toBeDefined();
    expect(screen.getByText("Throughput Rate")).toBeDefined();
    expect(screen.getByText("50.0")).toBeDefined();
  });

  it("renders schedule summary card", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("User Schedules")).toBeDefined();
    expect(screen.getByText("Upcoming Runs (1h)")).toBeDefined();
    expect(screen.getByText("Upcoming Runs (24h)")).toBeDefined();
  });

  it("renders alert cards for critical issues", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("Orphaned Executions")).toBeDefined();
    expect(screen.getByText("Failed Executions (24h)")).toBeDefined();
    expect(screen.getByText("Long-Running Executions")).toBeDefined();
    expect(screen.getByText("Orphaned Schedules")).toBeDefined();
    expect(screen.getByText("Invalid States (Data Corruption)")).toBeDefined();
  });

  it("hides alert cards when counts are zero", () => {
    mockExecQuery.mockReturnValue({
      data: {
        data: {
          ...executionData,
          orphaned_running: 0,
          orphaned_queued: 0,
          failed_count_24h: 0,
          stuck_running_24h: 0,
          invalid_queued_with_start: 0,
          invalid_running_without_start: 0,
        },
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockAgentQuery.mockReturnValue({
      data: { data: agentData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockScheduleQuery.mockReturnValue({
      data: { data: { ...scheduleData, total_orphaned: 0 } },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.queryByText("Orphaned Executions")).toBeNull();
    expect(screen.queryByText("Failed Executions (24h)")).toBeNull();
    expect(screen.queryByText("Long-Running Executions")).toBeNull();
    expect(screen.queryByText("Orphaned Schedules")).toBeNull();
    expect(screen.queryByText("Invalid States (Data Corruption)")).toBeNull();
  });

  it("renders diagnostic information section", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("Diagnostic Information")).toBeDefined();
    expect(screen.getByText("Throughput Metrics:")).toBeDefined();
    expect(screen.getByText("Queue Health:")).toBeDefined();
  });

  it("shows no data message when execution data is null", () => {
    mockExecQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockAgentQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockScheduleQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<DiagnosticsContent />);
    const noDataMessages = screen.getAllByText("No data available");
    expect(noDataMessages.length).toBeGreaterThanOrEqual(1);
  });

  it("shows RabbitMQ error state when depth is -1", () => {
    mockExecQuery.mockReturnValue({
      data: {
        data: { ...executionData, queued_executions_rabbitmq: -1 },
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockAgentQuery.mockReturnValue({
      data: { data: agentData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockScheduleQuery.mockReturnValue({
      data: { data: scheduleData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<DiagnosticsContent />);
    const errorTexts = screen.getAllByText("Error");
    expect(errorTexts.length).toBeGreaterThanOrEqual(1);
  });

  it("renders completed 24h and 1h values", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("1200")).toBeDefined();
    expect(screen.getByText("50 in last hour")).toBeDefined();
  });

  it("renders schedule metric values", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText("12")).toBeDefined();
    expect(screen.getByText("48")).toBeDefined();
  });

  it("renders oldest running hours in alert card", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText(/oldest:.*26h/)).toBeDefined();
  });

  it("renders cancel queue depth error when -1", () => {
    mockExecQuery.mockReturnValue({
      data: {
        data: { ...executionData, cancel_queue_depth: -1 },
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockAgentQuery.mockReturnValue({
      data: { data: agentData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    mockScheduleQuery.mockReturnValue({
      data: { data: scheduleData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<DiagnosticsContent />);
    const errorTexts = screen.getAllByText("Error");
    expect(errorTexts.length).toBeGreaterThanOrEqual(1);
  });

  it("renders stuck queued count in queue status card", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText(/2 stuck/)).toBeDefined();
  });

  it("renders schedule orphaned count in card", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText(/3 orphaned/)).toBeDefined();
  });

  it("clicking orphaned alert card does not crash", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    fireEvent.click(screen.getByText("Orphaned Executions"));
  });

  it("clicking failed alert card does not crash", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    fireEvent.click(screen.getByText("Failed Executions (24h)"));
  });

  it("clicking long-running alert card does not crash", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    fireEvent.click(screen.getByText("Long-Running Executions"));
  });

  it("clicking orphaned schedules alert card does not crash", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    fireEvent.click(screen.getByText("Orphaned Schedules"));
  });

  it("clicking invalid states alert card does not crash", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    fireEvent.click(screen.getByText("Invalid States (Data Corruption)"));
  });

  it("renders orphan detail text in schedule alert", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText(/2 deleted graph/)).toBeDefined();
    expect(screen.getByText(/1 no access/)).toBeDefined();
  });

  it("renders failure rate in failed alert card", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText(/0.8\/hr rate/)).toBeDefined();
  });

  it("renders click to view text on alert cards", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    const clickTexts = screen.getAllByText(/Click to view/);
    expect(clickTexts.length).toBeGreaterThanOrEqual(3);
  });

  it("renders schedule next hour count", () => {
    setupLoadedMocks();
    render(<DiagnosticsContent />);
    expect(screen.getByText(/from 4 schedules/)).toBeDefined();
  });

  it("clicking Refresh button calls all refetch functions", () => {
    const refetchExec = vi.fn();
    const refetchAgent = vi.fn();
    const refetchSchedule = vi.fn();
    mockExecQuery.mockReturnValue({
      data: { data: executionData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchExec,
    });
    mockAgentQuery.mockReturnValue({
      data: { data: agentData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchAgent,
    });
    mockScheduleQuery.mockReturnValue({
      data: { data: scheduleData },
      isLoading: false,
      isError: false,
      error: null,
      refetch: refetchSchedule,
    });
    render(<DiagnosticsContent />);
    fireEvent.click(screen.getByText("Refresh"));
    expect(refetchExec).toHaveBeenCalled();
    expect(refetchAgent).toHaveBeenCalled();
    expect(refetchSchedule).toHaveBeenCalled();
  });
});
