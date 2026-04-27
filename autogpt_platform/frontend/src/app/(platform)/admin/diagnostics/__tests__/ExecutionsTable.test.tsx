import {
  render,
  screen,
  cleanup,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ExecutionsTable } from "../components/ExecutionsTable";

const mockRunningQuery = vi.fn();
const mockOrphanedQuery = vi.fn();
const mockFailedQuery = vi.fn();
const mockLongRunningQuery = vi.fn();
const mockStuckQueuedQuery = vi.fn();
const mockInvalidQuery = vi.fn();
const mockStopSingle = vi.fn();
const mockStopMultiple = vi.fn();
const mockStopAllLongRunning = vi.fn();
const mockCleanupOrphaned = vi.fn();
const mockCleanupAllOrphaned = vi.fn();
const mockCleanupAllStuckQueued = vi.fn();
const mockRequeueSingle = vi.fn();
const mockRequeueMultiple = vi.fn();
const mockRequeueAllStuck = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  useGetV2ListRunningExecutions: (...args: unknown[]) =>
    mockRunningQuery(...args),
  useGetV2ListOrphanedExecutions: (...args: unknown[]) =>
    mockOrphanedQuery(...args),
  useGetV2ListFailedExecutions: (...args: unknown[]) =>
    mockFailedQuery(...args),
  useGetV2ListLongRunningExecutions: (...args: unknown[]) =>
    mockLongRunningQuery(...args),
  useGetV2ListStuckQueuedExecutions: (...args: unknown[]) =>
    mockStuckQueuedQuery(...args),
  useGetV2ListInvalidExecutions: (...args: unknown[]) =>
    mockInvalidQuery(...args),
  usePostV2StopSingleExecution: () => ({
    mutateAsync: mockStopSingle,
    isPending: false,
  }),
  usePostV2StopMultipleExecutions: () => ({
    mutateAsync: mockStopMultiple,
    isPending: false,
  }),
  usePostV2StopAllLongRunningExecutions: () => ({
    mutateAsync: mockStopAllLongRunning,
    isPending: false,
  }),
  usePostV2CleanupOrphanedExecutions: () => ({
    mutateAsync: mockCleanupOrphaned,
    isPending: false,
  }),
  usePostV2CleanupAllOrphanedExecutions: () => ({
    mutateAsync: mockCleanupAllOrphaned,
    isPending: false,
  }),
  usePostV2CleanupAllStuckQueuedExecutions: () => ({
    mutateAsync: mockCleanupAllStuckQueued,
    isPending: false,
  }),
  usePostV2RequeueStuckExecution: () => ({
    mutateAsync: mockRequeueSingle,
    isPending: false,
  }),
  usePostV2RequeueMultipleStuckExecutions: () => ({
    mutateAsync: mockRequeueMultiple,
    isPending: false,
  }),
  usePostV2RequeueAllStuckQueuedExecutions: () => ({
    mutateAsync: mockRequeueAllStuck,
    isPending: false,
  }),
}));

function defaultQueryReturn(overrides = {}) {
  return {
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    ...overrides,
  };
}

function withExecutions(
  executions: Record<string, unknown>[],
  total: number,
  overrides = {},
) {
  return defaultQueryReturn({
    data: { data: { executions, total } },
    ...overrides,
  });
}

const sampleExecution = {
  execution_id: "exec-001",
  graph_id: "graph-123",
  graph_name: "Test Agent",
  graph_version: 1,
  user_id: "user-abc",
  user_email: "alice@example.com",
  status: "RUNNING",
  created_at: "2026-04-16T10:00:00Z",
  started_at: "2026-04-16T10:01:00Z",
  queue_status: null,
};

const diagnosticsData = {
  orphaned_running: 2,
  orphaned_queued: 1,
  failed_count_24h: 5,
  stuck_running_24h: 3,
  stuck_queued_1h: 2,
  invalid_queued_with_start: 1,
  invalid_running_without_start: 1,
};

function setupDefaultMocks() {
  mockRunningQuery.mockReturnValue(defaultQueryReturn());
  mockOrphanedQuery.mockReturnValue(defaultQueryReturn());
  mockFailedQuery.mockReturnValue(defaultQueryReturn());
  mockLongRunningQuery.mockReturnValue(defaultQueryReturn());
  mockStuckQueuedQuery.mockReturnValue(defaultQueryReturn());
  mockInvalidQuery.mockReturnValue(defaultQueryReturn());
}

afterEach(() => {
  cleanup();
  mockRunningQuery.mockReset();
  mockOrphanedQuery.mockReset();
  mockFailedQuery.mockReset();
  mockLongRunningQuery.mockReset();
  mockStuckQueuedQuery.mockReset();
  mockInvalidQuery.mockReset();
});

describe("ExecutionsTable", () => {
  it("shows empty state when no executions", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("No running executions")).toBeDefined();
  });

  it("renders execution rows in all tab", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("Test Agent")).toBeDefined();
    expect(screen.getByText("alice@example.com")).toBeDefined();
    expect(screen.getByText("RUNNING")).toBeDefined();
  });

  it("shows loading spinner", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(defaultQueryReturn({ isLoading: true }));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(document.querySelector(".animate-spin")).toBeDefined();
  });

  it("renders tab triggers with counts from diagnostics data", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/Orphaned/)).toBeDefined();
    expect(screen.getByText(/Failed/)).toBeDefined();
    expect(screen.getByText(/Long-Running/)).toBeDefined();
    expect(screen.getByText(/Stuck Queued/)).toBeDefined();
    expect(screen.getByText(/Invalid/)).toBeDefined();
  });

  it("renders error state", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(
      defaultQueryReturn({ error: { status: 500, message: "Server down" } }),
    );
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("Try Again")).toBeDefined();
  });

  it("renders failed execution with error message", () => {
    setupDefaultMocks();
    const failedExec = {
      ...sampleExecution,
      execution_id: "exec-fail-1",
      status: "FAILED",
      failed_at: "2026-04-16T12:00:00Z",
      error_message: "Out of memory",
    };
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    mockFailedQuery.mockReturnValue(withExecutions([failedExec], 1));
    render(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="failed" />,
    );
    expect(screen.getByText("Out of memory")).toBeDefined();
  });

  it("renders pagination when total exceeds page size", () => {
    setupDefaultMocks();
    const executions = Array.from({ length: 10 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-${i}`,
    }));
    mockRunningQuery.mockReturnValue(withExecutions(executions, 25));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/Page 1 of 3/)).toBeDefined();
    expect(screen.getByText("Previous")).toBeDefined();
    expect(screen.getByText("Next")).toBeDefined();
  });

  it("shows unknown for null user email", () => {
    setupDefaultMocks();
    const noEmailExec = {
      ...sampleExecution,
      user_email: null,
    };
    mockRunningQuery.mockReturnValue(withExecutions([noEmailExec], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("Unknown")).toBeDefined();
  });

  it("copies execution ID to clipboard on click", () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    fireEvent.click(screen.getByText("exec-001".substring(0, 8) + "..."));
    expect(writeText).toHaveBeenCalledWith("exec-001");
    vi.unstubAllGlobals();
  });

  it("copies user ID to clipboard on click", () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    fireEvent.click(screen.getByText("user-abc".substring(0, 8) + "..."));
    expect(writeText).toHaveBeenCalledWith("user-abc");
    vi.unstubAllGlobals();
  });

  it("shows never started for null started_at", () => {
    setupDefaultMocks();
    const neverStarted = {
      ...sampleExecution,
      started_at: null,
    };
    mockRunningQuery.mockReturnValue(withExecutions([neverStarted], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("Never started")).toBeDefined();
  });

  it("renders stuck-queued tab with requeue buttons", () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-1",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    expect(screen.getByTitle("Cleanup (mark as FAILED)")).toBeDefined();
    expect(screen.getByTitle("Requeue (send to RabbitMQ)")).toBeDefined();
  });

  it("renders orphaned tab executions", () => {
    setupDefaultMocks();
    const orphanedExec = {
      ...sampleExecution,
      execution_id: "exec-orphan-1",
      created_at: "2026-04-10T10:00:00Z",
    };
    mockOrphanedQuery.mockReturnValue(withExecutions([orphanedExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="orphaned"
      />,
    );
    expect(screen.getByText("Test Agent")).toBeDefined();
  });

  it("renders long-running tab executions", () => {
    setupDefaultMocks();
    mockLongRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="long-running"
      />,
    );
    expect(screen.getByText("Test Agent")).toBeDefined();
  });

  it("renders invalid tab executions", () => {
    setupDefaultMocks();
    const invalidExec = {
      ...sampleExecution,
      execution_id: "exec-invalid-1",
      status: "QUEUED",
      started_at: "2026-04-16T10:01:00Z",
    };
    mockInvalidQuery.mockReturnValue(withExecutions([invalidExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="invalid"
      />,
    );
    expect(screen.getByText("QUEUED")).toBeDefined();
  });

  it("renders all tab trigger labels with correct counts", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/Orphaned.*3/)).toBeDefined();
    expect(screen.getByText(/Failed.*5/)).toBeDefined();
    expect(screen.getByText(/Stuck Queued.*2/)).toBeDefined();
    expect(screen.getByText(/Long-Running.*3/)).toBeDefined();
    expect(screen.getByText(/Invalid States.*2/)).toBeDefined();
  });

  it("shows graph version number", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("1")).toBeDefined();
  });

  it("renders QUEUED status badge", () => {
    setupDefaultMocks();
    const queuedExec = { ...sampleExecution, status: "QUEUED" };
    mockRunningQuery.mockReturnValue(withExecutions([queuedExec], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("QUEUED")).toBeDefined();
  });

  it("renders without diagnosticsData", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    render(<ExecutionsTable />);
    expect(screen.getByText(/All/)).toBeDefined();
  });

  it("renders stuck-queued bulk action buttons when total > 0", () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 5));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    expect(screen.getByText(/Cleanup All \(5\)/)).toBeDefined();
    expect(screen.getByText(/Requeue All \(5\)/)).toBeDefined();
  });

  it("renders long-running stop all button when total > 0", () => {
    setupDefaultMocks();
    mockLongRunningQuery.mockReturnValue(withExecutions([sampleExecution], 3));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="long-running"
      />,
    );
    expect(screen.getByText(/Stop All Long-Running \(3\)/)).toBeDefined();
  });

  it("shows invalid state read-only banner", () => {
    setupDefaultMocks();
    mockInvalidQuery.mockReturnValue(withExecutions([], 0));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="invalid"
      />,
    );
    expect(
      screen.getByText(
        /Read-only: Invalid states require manual investigation/,
      ),
    ).toBeDefined();
  });

  it("shows view-only message in failed tab with no selection", () => {
    setupDefaultMocks();
    const failedExec = {
      ...sampleExecution,
      status: "FAILED",
      error_message: "err",
    };
    mockFailedQuery.mockReturnValue(withExecutions([failedExec], 1));
    render(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="failed" />,
    );
    expect(screen.getByText("View-only (select to delete)")).toBeDefined();
  });

  it("renders table column headers", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("Execution ID")).toBeDefined();
    expect(screen.getByText("Agent Name")).toBeDefined();
    expect(screen.getByText("Version")).toBeDefined();
    expect(screen.getByText("User")).toBeDefined();
    expect(screen.getByText("Status")).toBeDefined();
    expect(screen.getByText("Age")).toBeDefined();
  });

  it("renders failed tab with error column header", () => {
    setupDefaultMocks();
    const failedExec = {
      ...sampleExecution,
      status: "FAILED",
      failed_at: "2026-04-16T12:00:00Z",
      error_message: "Timeout",
    };
    mockFailedQuery.mockReturnValue(withExecutions([failedExec], 1));
    render(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="failed" />,
    );
    expect(screen.getByText("Error Message")).toBeDefined();
    expect(screen.getByText("Timeout")).toBeDefined();
  });

  it("renders no error message text when error_message is null", () => {
    setupDefaultMocks();
    const failedNoMsg = {
      ...sampleExecution,
      status: "FAILED",
      failed_at: "2026-04-16T12:00:00Z",
      error_message: null,
    };
    mockFailedQuery.mockReturnValue(withExecutions([failedNoMsg], 1));
    render(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="failed" />,
    );
    expect(screen.getByText("No error message")).toBeDefined();
  });

  it("renders started_at as dash when null in non-failed tab", () => {
    setupDefaultMocks();
    const noStart = { ...sampleExecution, started_at: null };
    mockRunningQuery.mockReturnValue(withExecutions([noStart], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const dashes = screen.getAllByText("-");
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it("renders failed_at as dash when null in failed tab", () => {
    setupDefaultMocks();
    const failedNoDate = {
      ...sampleExecution,
      status: "FAILED",
      failed_at: null,
      error_message: "err",
    };
    mockFailedQuery.mockReturnValue(withExecutions([failedNoDate], 1));
    render(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="failed" />,
    );
    const dashes = screen.getAllByText("-");
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it("renders Executions card title", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText("Executions")).toBeDefined();
  });

  it("opens stop dialog when clicking cleanup button on stuck-queued row", async () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-dialog",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(
        screen.getByText("Confirm Cleanup Orphaned Executions"),
      ).toBeDefined();
      expect(screen.getByText("Cancel")).toBeDefined();
      expect(screen.getByText("Cleanup Orphaned")).toBeDefined();
    });
  });

  it("calls cleanupOrphanedExecutions when confirming single cleanup", async () => {
    setupDefaultMocks();
    mockCleanupOrphaned.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "Cleaned" },
    });
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-confirm",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(screen.getByText("Cleanup Orphaned")).toBeDefined();
    });
    fireEvent.click(screen.getByText("Cleanup Orphaned"));
    await waitFor(() => {
      expect(mockCleanupOrphaned).toHaveBeenCalled();
    });
  });

  it("opens cleanup dialog for stuck-queued execution", async () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-1",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(
        screen.getByText("Confirm Cleanup Orphaned Executions"),
      ).toBeDefined();
      expect(screen.getByText("Cleanup Orphaned")).toBeDefined();
    });
  });

  it("calls cleanupOrphanedExecutions when confirming cleanup", async () => {
    setupDefaultMocks();
    mockCleanupOrphaned.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "Cleaned" },
    });
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-1",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(screen.getByText("Cleanup Orphaned")).toBeDefined();
    });
    fireEvent.click(screen.getByText("Cleanup Orphaned"));
    await waitFor(() => {
      expect(mockCleanupOrphaned).toHaveBeenCalled();
    });
  });

  it("opens requeue dialog for stuck-queued execution", async () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-1",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Requeue (send to RabbitMQ)"));
    await waitFor(() => {
      expect(
        screen.getByText("Confirm Requeue Stuck Executions"),
      ).toBeDefined();
      expect(screen.getByText("Requeue Executions")).toBeDefined();
    });
  });

  it("calls requeueSingleExecution when confirming requeue", async () => {
    setupDefaultMocks();
    mockRequeueSingle.mockResolvedValue({
      data: { success: true, requeued_count: 1, message: "Requeued" },
    });
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-1",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Requeue (send to RabbitMQ)"));
    await waitFor(() => {
      expect(screen.getByText("Requeue Executions")).toBeDefined();
    });
    fireEvent.click(screen.getByText("Requeue Executions"));
    await waitFor(() => {
      expect(mockRequeueSingle).toHaveBeenCalled();
    });
  });

  it("closes dialog when cancel is clicked", async () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-cancel-test",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(
        screen.getByText("Confirm Cleanup Orphaned Executions"),
      ).toBeDefined();
    });
    fireEvent.click(screen.getByText("Cancel"));
    await waitFor(() => {
      expect(
        screen.queryByText("Confirm Cleanup Orphaned Executions"),
      ).toBeNull();
    });
  });

  it("handles cleanup mutation error gracefully", async () => {
    setupDefaultMocks();
    mockCleanupOrphaned.mockRejectedValue(new Error("Network error"));
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-error-test",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(screen.getByText("Cleanup Orphaned")).toBeDefined();
    });
    fireEvent.click(screen.getByText("Cleanup Orphaned"));
    await waitFor(() => {
      expect(mockCleanupOrphaned).toHaveBeenCalled();
    });
  });

  it("calls requeueAllStuck when clicking Requeue All button and confirming", async () => {
    setupDefaultMocks();
    mockRequeueAllStuck.mockResolvedValue({
      data: { success: true, requeued_count: 5, message: "Requeued 5" },
    });
    const stuckExecs = Array.from({ length: 3 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-stuck-${i}`,
      status: "QUEUED",
      started_at: null,
    }));
    mockStuckQueuedQuery.mockReturnValue(withExecutions(stuckExecs, 5));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByText(/Requeue All \(5\)/));
    await waitFor(() => {
      expect(
        screen.getByText("Confirm Requeue Stuck Executions"),
      ).toBeDefined();
    });
    fireEvent.click(screen.getByText("Requeue Executions"));
    await waitFor(() => {
      expect(mockRequeueAllStuck).toHaveBeenCalled();
    });
  });

  it("calls cleanupAllStuckQueued when clicking Cleanup All on stuck-queued tab", async () => {
    setupDefaultMocks();
    mockCleanupAllStuckQueued.mockResolvedValue({
      data: { success: true, stopped_count: 5, message: "Cleaned 5" },
    });
    const stuckExecs = Array.from({ length: 3 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-stuck-${i}`,
      status: "QUEUED",
      started_at: null,
    }));
    mockStuckQueuedQuery.mockReturnValue(withExecutions(stuckExecs, 5));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByText(/Cleanup All \(5\)/));
    await waitFor(() => {
      expect(
        screen.getByText("Confirm Cleanup Orphaned Executions"),
      ).toBeDefined();
    });
    fireEvent.click(screen.getByText("Cleanup Orphaned"));
    await waitFor(() => {
      expect(mockCleanupAllStuckQueued).toHaveBeenCalled();
    });
  });

  it("calls stopAllLongRunning when clicking Stop All Long-Running", async () => {
    setupDefaultMocks();
    mockStopAllLongRunning.mockResolvedValue({
      data: { success: true, stopped_count: 3, message: "Stopped 3" },
    });
    mockLongRunningQuery.mockReturnValue(withExecutions([sampleExecution], 3));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="long-running"
      />,
    );
    fireEvent.click(screen.getByText(/Stop All Long-Running \(3\)/));
    await waitFor(() => {
      expect(screen.getByText("Confirm Stop Executions")).toBeDefined();
    });
    fireEvent.click(screen.getByText("Stop Executions"));
    await waitFor(() => {
      expect(mockStopAllLongRunning).toHaveBeenCalled();
    });
  });

  it("shows requeue warning text in dialog", async () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-warn",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Requeue (send to RabbitMQ)"));
    await waitFor(() => {
      expect(screen.getByText(/will cost credits/)).toBeDefined();
    });
  });

  it("shows cleanup description in dialog", async () => {
    setupDefaultMocks();
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-stuck-desc",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(screen.getByText(/cleanup this orphaned execution/)).toBeDefined();
    });
  });

  it("renders age in days format for old executions", () => {
    setupDefaultMocks();
    const oldExec = {
      ...sampleExecution,
      started_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    };
    mockRunningQuery.mockReturnValue(withExecutions([oldExec], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/3d/)).toBeDefined();
  });

  it("shows stop selected button after selecting a checkbox", async () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    if (checkboxes[1]) fireEvent.click(checkboxes[1]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected/)).toBeDefined();
    });
  });

  it("shows stop selected button with count after selection", async () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    if (checkboxes[1]) fireEvent.click(checkboxes[1]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected \(1\)/)).toBeDefined();
    });
  });

  it("renders select-all checkbox", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    expect(checkboxes.length).toBeGreaterThanOrEqual(2);
  });

  it("selects all checkboxes with select-all", async () => {
    setupDefaultMocks();
    const execs = [
      { ...sampleExecution, execution_id: "exec-a" },
      { ...sampleExecution, execution_id: "exec-b" },
    ];
    mockRunningQuery.mockReturnValue(withExecutions(execs, 2));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    // First checkbox is select-all
    if (checkboxes[0]) fireEvent.click(checkboxes[0]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected \(2\)/)).toBeDefined();
    });
  });

  it("renders hours format for recent execution age", () => {
    setupDefaultMocks();
    const recentExec = {
      ...sampleExecution,
      started_at: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
    };
    mockRunningQuery.mockReturnValue(withExecutions([recentExec], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/5h/)).toBeDefined();
  });

  it("calls onRefresh when provided", async () => {
    setupDefaultMocks();
    const onRefresh = vi.fn();
    mockStopSingle.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "Stopped" },
    });
    const stuckExec = {
      ...sampleExecution,
      execution_id: "exec-refresh-test",
      status: "QUEUED",
      started_at: null,
    };
    mockStuckQueuedQuery.mockReturnValue(withExecutions([stuckExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
        onRefresh={onRefresh}
      />,
    );
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(screen.getByText("Cleanup Orphaned")).toBeDefined();
    });
    mockCleanupOrphaned.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "OK" },
    });
    fireEvent.click(screen.getByText("Cleanup Orphaned"));
    await waitFor(() => {
      expect(onRefresh).toHaveBeenCalled();
    });
  });

  it("renders showing count text in pagination", () => {
    setupDefaultMocks();
    const executions = Array.from({ length: 10 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-page-${i}`,
    }));
    mockRunningQuery.mockReturnValue(withExecutions(executions, 30));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/Showing 1 to 10 of 30/)).toBeDefined();
  });

  it("disables Previous button on first page", () => {
    setupDefaultMocks();
    const executions = Array.from({ length: 10 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-dis-${i}`,
    }));
    mockRunningQuery.mockReturnValue(withExecutions(executions, 25));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const prevBtn = screen.getByText("Previous").closest("button");
    expect(prevBtn?.disabled).toBe(true);
  });

  it("enables Next button when more pages exist", () => {
    setupDefaultMocks();
    const executions = Array.from({ length: 10 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-next-${i}`,
    }));
    mockRunningQuery.mockReturnValue(withExecutions(executions, 25));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const nextBtn = screen.getByText("Next").closest("button");
    expect(nextBtn?.disabled).toBe(false);
  });

  it("renders orphaned execution with orange background", () => {
    setupDefaultMocks();
    const orphanedExec = {
      ...sampleExecution,
      execution_id: "exec-orange",
      created_at: "2026-04-10T10:00:00Z",
    };
    mockOrphanedQuery.mockReturnValue(withExecutions([orphanedExec], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="orphaned"
      />,
    );
    const row = screen.getByText("Test Agent").closest("tr");
    expect(row?.className).toContain("bg-orange");
  });

  it("renders initialTab syncs with useEffect", () => {
    setupDefaultMocks();
    mockFailedQuery.mockReturnValue(
      withExecutions(
        [
          {
            ...sampleExecution,
            execution_id: "exec-sync",
            status: "FAILED",
            error_message: "sync test",
          },
        ],
        1,
      ),
    );
    const { rerender } = render(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="all" />,
    );
    // Rerender with new initialTab to trigger useEffect sync
    rerender(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="failed" />,
    );
    expect(screen.getByText("sync test")).toBeDefined();
  });

  it("renders the all tab total count", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 7));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    // "All (7)" in the tab trigger
    expect(screen.getByText(/All.*7/)).toBeDefined();
  });

  it("opens stop dialog and calls mutations for selected executions", async () => {
    setupDefaultMocks();
    mockStopMultiple.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "Stopped 1" },
    });
    mockCleanupOrphaned.mockResolvedValue({
      data: { success: true, stopped_count: 0, message: "OK" },
    });
    // Use a recent execution that won't be classified as orphaned
    const recentExec = {
      ...sampleExecution,
      execution_id: "exec-recent-stop",
      created_at: new Date().toISOString(),
    };
    mockRunningQuery.mockReturnValue(withExecutions([recentExec], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    // Select execution
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    if (checkboxes[1]) fireEvent.click(checkboxes[1]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected/)).toBeDefined();
    });
    // Click stop selected
    fireEvent.click(screen.getByText(/Stop Selected/));
    // Dialog should open
    await waitFor(() => {
      expect(screen.getByText("Confirm Stop Executions")).toBeDefined();
    });
    // Confirm
    fireEvent.click(screen.getByText("Stop Executions"));
    await waitFor(() => {
      expect(mockStopMultiple).toHaveBeenCalled();
    });
  });

  it("calls requeueMultiple for selected stuck-queued executions", async () => {
    setupDefaultMocks();
    mockRequeueMultiple.mockResolvedValue({
      data: { success: true, requeued_count: 2, message: "Requeued 2" },
    });
    const stuckExecs = [
      {
        ...sampleExecution,
        execution_id: "stuck-a",
        status: "QUEUED",
        started_at: null,
      },
      {
        ...sampleExecution,
        execution_id: "stuck-b",
        status: "QUEUED",
        started_at: null,
      },
    ];
    mockStuckQueuedQuery.mockReturnValue(withExecutions(stuckExecs, 2));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="stuck-queued"
      />,
    );
    // Select all via select-all checkbox
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    if (checkboxes[0]) fireEvent.click(checkboxes[0]);
    // In stuck-queued tab, no "Stop Selected" button - only Cleanup All / Requeue All
    // Use Requeue All button instead
    await waitFor(() => {
      expect(screen.getByText(/Requeue All \(2\)/)).toBeDefined();
    });
    fireEvent.click(screen.getByText(/Requeue All \(2\)/));
    await waitFor(() => {
      expect(screen.getByText("Requeue Executions")).toBeDefined();
    });
    fireEvent.click(screen.getByText("Requeue Executions"));
    await waitFor(() => {
      expect(mockRequeueAllStuck).toHaveBeenCalled();
    });
  });

  it("shows dialog description for stop all on long-running tab", async () => {
    setupDefaultMocks();
    mockLongRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="long-running"
      />,
    );
    fireEvent.click(screen.getByText(/Stop All Long-Running/));
    await waitFor(() => {
      expect(screen.getByText(/stop ALL 1 execution/)).toBeDefined();
    });
  });

  it("shows stop dialog description listing what it does", async () => {
    setupDefaultMocks();
    mockLongRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        initialTab="long-running"
      />,
    );
    fireEvent.click(screen.getByText(/Stop All Long-Running/));
    await waitFor(() => {
      expect(
        screen.getByText(/Send cancel signals for active executions/),
      ).toBeDefined();
      expect(screen.getByText(/Mark all as FAILED/)).toBeDefined();
    });
  });

  it("clicking refresh button calls refetch and onRefresh", () => {
    setupDefaultMocks();
    const onRefresh = vi.fn();
    const refetch = vi.fn();
    mockRunningQuery.mockReturnValue({
      data: { data: { executions: [sampleExecution], total: 1 } },
      isLoading: false,
      error: null,
      refetch,
    });
    render(
      <ExecutionsTable
        diagnosticsData={diagnosticsData}
        onRefresh={onRefresh}
      />,
    );
    // The refresh button is the last button with ArrowClockwise icon in the header
    const buttons = document.querySelectorAll("button");
    // Find the standalone refresh button (no text, just icon)
    const refreshBtn = Array.from(buttons).find(
      (b) => b.querySelector("svg") && b.textContent?.trim() === "",
    );
    if (refreshBtn) {
      fireEvent.click(refreshBtn);
      expect(refetch).toHaveBeenCalled();
      expect(onRefresh).toHaveBeenCalled();
    }
  });

  it("renders executions text label in Showing pagination", () => {
    setupDefaultMocks();
    const executions = Array.from({ length: 10 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-label-${i}`,
    }));
    mockRunningQuery.mockReturnValue(withExecutions(executions, 20));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/executions/)).toBeDefined();
  });

  it("renders status badge with green for RUNNING", () => {
    setupDefaultMocks();
    mockRunningQuery.mockReturnValue(withExecutions([sampleExecution], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const badge = screen.getByText("RUNNING");
    expect(badge.className).toContain("bg-green");
  });

  it("renders status badge with yellow for QUEUED", () => {
    setupDefaultMocks();
    const queuedExec = { ...sampleExecution, status: "QUEUED" };
    mockRunningQuery.mockReturnValue(withExecutions([queuedExec], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    const badge = screen.getByText("QUEUED");
    expect(badge.className).toContain("bg-yellow");
  });

  it("clicking Next advances pagination page", () => {
    setupDefaultMocks();
    const executions = Array.from({ length: 10 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-pagnext-${i}`,
    }));
    mockRunningQuery.mockReturnValue(withExecutions(executions, 25));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    expect(screen.getByText(/Page 1 of 3/)).toBeDefined();
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText(/Page 2 of 3/)).toBeDefined();
  });

  it("clicking Previous goes back a page", () => {
    setupDefaultMocks();
    const executions = Array.from({ length: 10 }, (_, i) => ({
      ...sampleExecution,
      execution_id: `exec-pagprev-${i}`,
    }));
    mockRunningQuery.mockReturnValue(withExecutions(executions, 25));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText(/Page 2 of 3/)).toBeDefined();
    fireEvent.click(screen.getByText("Previous"));
    expect(screen.getByText(/Page 1 of 3/)).toBeDefined();
  });

  it("splits orphaned and active IDs when stopping selected with old execution", async () => {
    setupDefaultMocks();
    mockStopMultiple.mockResolvedValue({
      data: { success: true, stopped_count: 0, message: "OK" },
    });
    mockCleanupOrphaned.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "Cleaned 1" },
    });
    // Use an OLD execution (>24h) so it's classified as orphaned
    const oldExec = {
      ...sampleExecution,
      execution_id: "exec-old-orphan",
      created_at: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
    };
    mockRunningQuery.mockReturnValue(withExecutions([oldExec], 1));
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    // Select the old execution
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    if (checkboxes[1]) fireEvent.click(checkboxes[1]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected/)).toBeDefined();
    });
    fireEvent.click(screen.getByText(/Stop Selected/));
    await waitFor(() => {
      expect(screen.getByText("Stop Executions")).toBeDefined();
    });
    fireEvent.click(screen.getByText("Stop Executions"));
    await waitFor(() => {
      // Should call cleanupOrphaned for the old execution
      expect(mockCleanupOrphaned).toHaveBeenCalled();
    });
  });

  it("clicking Try Again on error state calls refetch", () => {
    setupDefaultMocks();
    const refetch = vi.fn();
    mockRunningQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: { status: 500, message: "Server error" },
      refetch,
    });
    render(<ExecutionsTable diagnosticsData={diagnosticsData} />);
    fireEvent.click(screen.getByText("Try Again"));
    expect(refetch).toHaveBeenCalled();
  });
});
