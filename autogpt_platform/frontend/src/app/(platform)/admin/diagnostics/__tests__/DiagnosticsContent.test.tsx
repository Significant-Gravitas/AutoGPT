import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DiagnosticsContent } from "../components/DiagnosticsContent";

const mockUseDiagnosticsContent = vi.fn();

vi.mock("../components/useDiagnosticsContent", () => ({
  useDiagnosticsContent: () => mockUseDiagnosticsContent(),
}));

vi.mock("../components/ExecutionsTable", () => ({
  ExecutionsTable: () => (
    <div data-testid="executions-table">ExecutionsTable</div>
  ),
}));

vi.mock("../components/SchedulesTable", () => ({
  SchedulesTable: () => <div data-testid="schedules-table">SchedulesTable</div>,
}));

afterEach(() => {
  cleanup();
  mockUseDiagnosticsContent.mockReset();
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

describe("DiagnosticsContent", () => {
  it("shows loading state", () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData: undefined,
      agentData: undefined,
      scheduleData: undefined,
      isLoading: true,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("Loading diagnostics...")).toBeDefined();
  });

  it("shows error state with retry", () => {
    const refresh = vi.fn();
    mockUseDiagnosticsContent.mockReturnValue({
      executionData: undefined,
      agentData: undefined,
      scheduleData: undefined,
      isLoading: false,
      isError: true,
      error: { status: 500, message: "Server error" },
      refresh,
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("Try Again")).toBeDefined();
  });

  it("renders system diagnostics heading with data", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("System Diagnostics")).toBeDefined();
    expect(screen.getByText("Refresh")).toBeDefined();
  });

  it("renders execution queue status cards", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("Execution Queue Status")).toBeDefined();
    expect(screen.getByText("Running Executions")).toBeDefined();
    expect(screen.getByText("Queued in Database")).toBeDefined();
    expect(screen.getByText("Queued in RabbitMQ")).toBeDefined();
  });

  it("renders throughput metrics", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("System Throughput")).toBeDefined();
    expect(screen.getByText("Completed (24h)")).toBeDefined();
    expect(screen.getByText("Throughput Rate")).toBeDefined();
    expect(screen.getByText("50.0")).toBeDefined();
  });

  it("renders schedule summary card", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("User Schedules")).toBeDefined();
    expect(screen.getByText("Upcoming Runs (1h)")).toBeDefined();
    expect(screen.getByText("Upcoming Runs (24h)")).toBeDefined();
  });

  it("renders alert cards for critical issues", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("Orphaned Executions")).toBeDefined();
    expect(screen.getByText("Failed Executions (24h)")).toBeDefined();
    expect(screen.getByText("Long-Running Executions")).toBeDefined();
    expect(screen.getByText("Orphaned Schedules")).toBeDefined();
    expect(screen.getByText("Invalid States (Data Corruption)")).toBeDefined();
  });

  it("hides alert cards when counts are zero", async () => {
    const noIssuesData = {
      ...executionData,
      orphaned_running: 0,
      orphaned_queued: 0,
      failed_count_24h: 0,
      stuck_running_24h: 0,
      invalid_queued_with_start: 0,
      invalid_running_without_start: 0,
    };
    const noOrphanSchedules = {
      ...scheduleData,
      total_orphaned: 0,
    };
    mockUseDiagnosticsContent.mockReturnValue({
      executionData: noIssuesData,
      agentData,
      scheduleData: noOrphanSchedules,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.queryByText("Orphaned Executions")).toBeNull();
    expect(screen.queryByText("Failed Executions (24h)")).toBeNull();
    expect(screen.queryByText("Long-Running Executions")).toBeNull();
    expect(screen.queryByText("Orphaned Schedules")).toBeNull();
    expect(screen.queryByText("Invalid States (Data Corruption)")).toBeNull();
  });

  it("renders diagnostic information section", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByText("Diagnostic Information")).toBeDefined();
    expect(screen.getByText("Throughput Metrics:")).toBeDefined();
    expect(screen.getByText("Queue Health:")).toBeDefined();
  });

  it("renders tables", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    expect(screen.getByTestId("executions-table")).toBeDefined();
    expect(screen.getByTestId("schedules-table")).toBeDefined();
  });

  it("shows no data message when execution data is null", async () => {
    mockUseDiagnosticsContent.mockReturnValue({
      executionData: undefined,
      agentData: undefined,
      scheduleData: undefined,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    const noDataMessages = screen.getAllByText("No data available");
    expect(noDataMessages.length).toBeGreaterThanOrEqual(1);
  });

  it("shows RabbitMQ error state when depth is -1", async () => {
    const errorData = {
      ...executionData,
      queued_executions_rabbitmq: -1,
    };
    mockUseDiagnosticsContent.mockReturnValue({
      executionData: errorData,
      agentData,
      scheduleData,
      isLoading: false,
      isError: false,
      error: undefined,
      refresh: vi.fn(),
    });
    render(<DiagnosticsContent />);
    const errorTexts = screen.getAllByText("Error");
    expect(errorTexts.length).toBeGreaterThanOrEqual(1);
  });
});
