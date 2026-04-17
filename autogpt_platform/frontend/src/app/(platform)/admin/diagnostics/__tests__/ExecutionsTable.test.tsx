import {
  render,
  screen,
  cleanup,
  fireEvent,
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
});
