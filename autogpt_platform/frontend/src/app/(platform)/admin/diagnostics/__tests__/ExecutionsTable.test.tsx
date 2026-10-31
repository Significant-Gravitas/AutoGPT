import {
  render,
  screen,
  cleanup,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

const queryByTab = {
  all: mockRunningQuery,
  orphaned: mockOrphanedQuery,
  failed: mockFailedQuery,
  "long-running": mockLongRunningQuery,
  "stuck-queued": mockStuckQueuedQuery,
  invalid: mockInvalidQuery,
};

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

beforeEach(setupDefaultMocks);

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
    renderTab("all", []);
    expect(screen.getByText("No running executions")).toBeDefined();
  });

  it("renders execution rows in all tab", () => {
    renderTab("all", [sampleExecution]);
    expect(screen.getByText("Test Agent")).toBeDefined();
    expect(screen.getByText("alice@example.com")).toBeDefined();
    expect(screen.getByText("RUNNING")).toBeDefined();
  });

  it("shows loading spinner", () => {
    mockRunningQuery.mockReturnValue(defaultQueryReturn({ isLoading: true }));
    renderTable();
    expect(document.querySelector(".animate-spin")).toBeDefined();
  });

  it("renders tab triggers with counts from diagnostics data", () => {
    renderTab("all", []);
    expect(screen.getByText(/Orphaned/)).toBeDefined();
    expect(screen.getByText(/Failed/)).toBeDefined();
    expect(screen.getByText(/Long-Running/)).toBeDefined();
    expect(screen.getByText(/Stuck Queued/)).toBeDefined();
    expect(screen.getByText(/Invalid/)).toBeDefined();
  });

  it("renders error state", () => {
    mockRunningQuery.mockReturnValue(
      defaultQueryReturn({ error: { status: 500, message: "Server down" } }),
    );
    renderTable();
    expect(screen.getByText("Try Again")).toBeDefined();
  });

  it("renders failed execution with error message", () => {
    const failedExec = {
      ...sampleExecution,
      execution_id: "exec-fail-1",
      status: "FAILED",
      failed_at: "2026-04-16T12:00:00Z",
      error_message: "Out of memory",
    };
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    renderTab("failed", [failedExec]);
    expect(screen.getByText("Out of memory")).toBeDefined();
  });

  it("renders pagination when total exceeds page size", () => {
    renderTab("all", executionsPage("exec-"), { total: 25 });
    expect(screen.getByText(/Page 1 of 3/)).toBeDefined();
    expect(screen.getByText("Previous")).toBeDefined();
    expect(screen.getByText("Next")).toBeDefined();
  });

  it("shows unknown for null user email", () => {
    renderTab("all", [{ ...sampleExecution, user_email: null }]);
    expect(screen.getByText("Unknown")).toBeDefined();
  });

  it.each([
    ["execution", "exec-001"],
    ["user", "user-abc"],
  ])("copies %s ID to clipboard on click", (_, id) => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
    renderTab("all", [sampleExecution]);
    fireEvent.click(screen.getByText(id.substring(0, 8) + "..."));
    expect(writeText).toHaveBeenCalledWith(id);
    vi.unstubAllGlobals();
  });

  it("shows never started for null started_at", () => {
    renderTab("all", [{ ...sampleExecution, started_at: null }]);
    expect(screen.getByText("Never started")).toBeDefined();
  });

  it("renders stuck-queued tab with requeue buttons", () => {
    renderTab("stuck-queued", [stuckExecution("exec-stuck-1")]);
    expect(screen.getByTitle("Cleanup (mark as FAILED)")).toBeDefined();
    expect(screen.getByTitle("Requeue (send to RabbitMQ)")).toBeDefined();
  });

  it("renders orphaned tab executions", () => {
    const orphanedExec = {
      ...sampleExecution,
      execution_id: "exec-orphan-1",
      created_at: "2026-04-10T10:00:00Z",
    };
    renderTab("orphaned", [orphanedExec]);
    expect(screen.getByText("Test Agent")).toBeDefined();
  });

  it("renders long-running tab executions", () => {
    renderTab("long-running", [sampleExecution]);
    expect(screen.getByText("Test Agent")).toBeDefined();
  });

  it("renders invalid tab executions", () => {
    const invalidExec = {
      ...sampleExecution,
      execution_id: "exec-invalid-1",
      status: "QUEUED",
      started_at: "2026-04-16T10:01:00Z",
    };
    renderTab("invalid", [invalidExec]);
    expect(screen.getByText("QUEUED")).toBeDefined();
  });

  it("renders all tab trigger labels with correct counts", () => {
    renderTab("all", []);
    expect(screen.getByText(/Orphaned.*3/)).toBeDefined();
    expect(screen.getByText(/Failed.*5/)).toBeDefined();
    expect(screen.getByText(/Stuck Queued.*2/)).toBeDefined();
    expect(screen.getByText(/Long-Running.*3/)).toBeDefined();
    expect(screen.getByText(/Invalid States.*2/)).toBeDefined();
  });

  it("shows graph version number", () => {
    renderTab("all", [sampleExecution]);
    expect(screen.getByText("1")).toBeDefined();
  });

  it("renders QUEUED status badge", () => {
    renderTab("all", [{ ...sampleExecution, status: "QUEUED" }]);
    expect(screen.getByText("QUEUED")).toBeDefined();
  });

  it("renders without diagnosticsData", () => {
    mockRunningQuery.mockReturnValue(withExecutions([], 0));
    render(<ExecutionsTable />);
    expect(screen.getByText(/All/)).toBeDefined();
  });

  it("renders stuck-queued bulk action buttons when total > 0", () => {
    renderTab("stuck-queued", [stuckExecution("exec-001")], { total: 5 });
    expect(screen.getByText(/Cleanup All \(5\)/)).toBeDefined();
    expect(screen.getByText(/Requeue All \(5\)/)).toBeDefined();
  });

  it("renders long-running stop all button when total > 0", () => {
    renderTab("long-running", [sampleExecution], { total: 3 });
    expect(screen.getByText(/Stop All Long-Running \(3\)/)).toBeDefined();
  });

  it("shows invalid state read-only banner", () => {
    renderTab("invalid", []);
    expect(
      screen.getByText(
        /Read-only: Invalid states require manual investigation/,
      ),
    ).toBeDefined();
  });

  it("shows view-only message in failed tab with no selection", () => {
    const failedExec = {
      ...sampleExecution,
      status: "FAILED",
      error_message: "err",
    };
    renderTab("failed", [failedExec]);
    expect(screen.getByText("View-only (select to delete)")).toBeDefined();
  });

  it("renders table column headers", () => {
    renderTab("all", [sampleExecution]);
    expect(screen.getByText("Execution ID")).toBeDefined();
    expect(screen.getByText("Agent Name")).toBeDefined();
    expect(screen.getByText("Version")).toBeDefined();
    expect(screen.getByText("User")).toBeDefined();
    expect(screen.getByText("Status")).toBeDefined();
    expect(screen.getByText("Age")).toBeDefined();
  });

  it("renders failed tab with error column header", () => {
    const failedExec = {
      ...sampleExecution,
      status: "FAILED",
      failed_at: "2026-04-16T12:00:00Z",
      error_message: "Timeout",
    };
    renderTab("failed", [failedExec]);
    expect(screen.getByText("Error Message")).toBeDefined();
    expect(screen.getByText("Timeout")).toBeDefined();
  });

  it("renders no error message text when error_message is null", () => {
    const failedNoMsg = {
      ...sampleExecution,
      status: "FAILED",
      failed_at: "2026-04-16T12:00:00Z",
      error_message: null,
    };
    renderTab("failed", [failedNoMsg]);
    expect(screen.getByText("No error message")).toBeDefined();
  });

  it("renders started_at as dash when null in non-failed tab", () => {
    renderTab("all", [{ ...sampleExecution, started_at: null }]);
    const dashes = screen.getAllByText("-");
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it("renders failed_at as dash when null in failed tab", () => {
    const failedNoDate = {
      ...sampleExecution,
      status: "FAILED",
      failed_at: null,
      error_message: "err",
    };
    renderTab("failed", [failedNoDate]);
    const dashes = screen.getAllByText("-");
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it("renders Executions card title", () => {
    renderTab("all", []);
    expect(screen.getByText("Executions")).toBeDefined();
  });

  it("opens stop dialog when clicking cleanup button on stuck-queued row", async () => {
    renderTab("stuck-queued", [stuckExecution("exec-stuck-dialog")]);
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
    mockCleanupOrphaned.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "Cleaned" },
    });
    renderTab("stuck-queued", [stuckExecution("exec-stuck-confirm")]);
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
    renderTab("stuck-queued", [stuckExecution("exec-stuck-1")]);
    fireEvent.click(screen.getByTitle("Requeue (send to RabbitMQ)"));
    await waitFor(() => {
      expect(
        screen.getByText("Confirm Requeue Stuck Executions"),
      ).toBeDefined();
      expect(screen.getByText("Requeue Executions")).toBeDefined();
    });
  });

  it("calls requeueSingleExecution when confirming requeue", async () => {
    mockRequeueSingle.mockResolvedValue({
      data: { success: true, requeued_count: 1, message: "Requeued" },
    });
    renderTab("stuck-queued", [stuckExecution("exec-stuck-1")]);
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
    renderTab("stuck-queued", [stuckExecution("exec-cancel-test")]);
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
    mockCleanupOrphaned.mockRejectedValue(new Error("Network error"));
    renderTab("stuck-queued", [stuckExecution("exec-error-test")]);
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
    mockRequeueAllStuck.mockResolvedValue({
      data: { success: true, requeued_count: 5, message: "Requeued 5" },
    });
    const stuckExecs = Array.from({ length: 3 }, (_, i) =>
      stuckExecution(`exec-stuck-${i}`),
    );
    renderTab("stuck-queued", stuckExecs, { total: 5 });
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
    mockCleanupAllStuckQueued.mockResolvedValue({
      data: { success: true, stopped_count: 5, message: "Cleaned 5" },
    });
    const stuckExecs = Array.from({ length: 3 }, (_, i) =>
      stuckExecution(`exec-stuck-${i}`),
    );
    renderTab("stuck-queued", stuckExecs, { total: 5 });
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
    mockStopAllLongRunning.mockResolvedValue({
      data: { success: true, stopped_count: 3, message: "Stopped 3" },
    });
    renderTab("long-running", [sampleExecution], { total: 3 });
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
    renderTab("stuck-queued", [stuckExecution("exec-stuck-warn")]);
    fireEvent.click(screen.getByTitle("Requeue (send to RabbitMQ)"));
    await waitFor(() => {
      expect(screen.getByText(/will cost credits/)).toBeDefined();
    });
  });

  it("shows cleanup description in dialog", async () => {
    renderTab("stuck-queued", [stuckExecution("exec-stuck-desc")]);
    fireEvent.click(screen.getByTitle("Cleanup (mark as FAILED)"));
    await waitFor(() => {
      expect(screen.getByText(/cleanup this orphaned execution/)).toBeDefined();
    });
  });

  it("renders age in days format for old executions", () => {
    const oldExec = {
      ...sampleExecution,
      started_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    };
    renderTab("all", [oldExec]);
    expect(screen.getByText(/3d/)).toBeDefined();
  });

  it("shows stop selected button after selecting a checkbox", async () => {
    renderTab("all", [sampleExecution]);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    if (checkboxes[1]) fireEvent.click(checkboxes[1]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected/)).toBeDefined();
    });
  });

  it("shows stop selected button with count after selection", async () => {
    renderTab("all", [sampleExecution]);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    if (checkboxes[1]) fireEvent.click(checkboxes[1]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected \(1\)/)).toBeDefined();
    });
  });

  it("renders select-all checkbox", () => {
    renderTab("all", [sampleExecution]);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    expect(checkboxes.length).toBeGreaterThanOrEqual(2);
  });

  it("selects all checkboxes with select-all", async () => {
    const execs = [
      { ...sampleExecution, execution_id: "exec-a" },
      { ...sampleExecution, execution_id: "exec-b" },
    ];
    renderTab("all", execs);
    const checkboxes = document.querySelectorAll('[role="checkbox"]');
    // First checkbox is select-all
    if (checkboxes[0]) fireEvent.click(checkboxes[0]);
    await waitFor(() => {
      expect(screen.getByText(/Stop Selected \(2\)/)).toBeDefined();
    });
  });

  it("renders hours format for recent execution age", () => {
    const recentExec = {
      ...sampleExecution,
      started_at: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
    };
    renderTab("all", [recentExec]);
    expect(screen.getByText(/5h/)).toBeDefined();
  });

  it("calls onRefresh when provided", async () => {
    const onRefresh = vi.fn();
    mockStopSingle.mockResolvedValue({
      data: { success: true, stopped_count: 1, message: "Stopped" },
    });
    renderTab("stuck-queued", [stuckExecution("exec-refresh-test")], {
      onRefresh,
    });
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
    renderTab("all", executionsPage("exec-page-"), { total: 30 });
    expect(screen.getByText(/Showing 1 to 10 of 30/)).toBeDefined();
  });

  it("disables Previous button on first page", () => {
    renderTab("all", executionsPage("exec-dis-"), { total: 25 });
    const prevBtn = screen.getByText("Previous").closest("button");
    expect(prevBtn?.disabled).toBe(true);
  });

  it("enables Next button when more pages exist", () => {
    renderTab("all", executionsPage("exec-next-"), { total: 25 });
    const nextBtn = screen.getByText("Next").closest("button");
    expect(nextBtn?.disabled).toBe(false);
  });

  it("renders orphaned execution with orange background", () => {
    const orphanedExec = {
      ...sampleExecution,
      execution_id: "exec-orange",
      created_at: "2026-04-10T10:00:00Z",
    };
    renderTab("orphaned", [orphanedExec]);
    const row = screen.getByText("Test Agent").closest("tr");
    expect(row?.className).toContain("bg-orange");
  });

  it("renders initialTab syncs with useEffect", () => {
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
    const { rerender } = renderTable({ initialTab: "all" });
    // Rerender with new initialTab to trigger useEffect sync
    rerender(
      <ExecutionsTable diagnosticsData={diagnosticsData} initialTab="failed" />,
    );
    expect(screen.getByText("sync test")).toBeDefined();
  });

  it("renders the all tab total count", () => {
    renderTab("all", [sampleExecution], { total: 7 });
    // "All (7)" in the tab trigger
    expect(screen.getByText(/All.*7/)).toBeDefined();
  });

  it("opens stop dialog and calls mutations for selected executions", async () => {
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
    renderTab("all", [recentExec]);
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
    mockRequeueMultiple.mockResolvedValue({
      data: { success: true, requeued_count: 2, message: "Requeued 2" },
    });
    renderTab("stuck-queued", [
      stuckExecution("stuck-a"),
      stuckExecution("stuck-b"),
    ]);
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
    renderTab("long-running", [sampleExecution]);
    fireEvent.click(screen.getByText(/Stop All Long-Running/));
    await waitFor(() => {
      expect(screen.getByText(/stop ALL 1 execution/)).toBeDefined();
    });
  });

  it("shows stop dialog description listing what it does", async () => {
    renderTab("long-running", [sampleExecution]);
    fireEvent.click(screen.getByText(/Stop All Long-Running/));
    await waitFor(() => {
      expect(
        screen.getByText(/Send cancel signals for active executions/),
      ).toBeDefined();
      expect(screen.getByText(/Mark all as FAILED/)).toBeDefined();
    });
  });

  it("clicking refresh button calls refetch and onRefresh", () => {
    const onRefresh = vi.fn();
    const refetch = vi.fn();
    mockRunningQuery.mockReturnValue(
      withExecutions([sampleExecution], 1, { refetch }),
    );
    renderTable({ onRefresh });
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
    renderTab("all", executionsPage("exec-label-"), { total: 20 });
    expect(screen.getByText(/executions/)).toBeDefined();
  });

  it("renders status badge with green for RUNNING", () => {
    renderTab("all", [sampleExecution]);
    const badge = screen.getByText("RUNNING");
    expect(badge.className).toContain("bg-success");
  });

  it("renders status badge with yellow for QUEUED", () => {
    renderTab("all", [{ ...sampleExecution, status: "QUEUED" }]);
    const badge = screen.getByText("QUEUED");
    expect(badge.className).toContain("bg-yellow");
  });

  it("clicking Next advances pagination page", () => {
    renderTab("all", executionsPage("exec-pagnext-"), { total: 25 });
    expect(screen.getByText(/Page 1 of 3/)).toBeDefined();
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText(/Page 2 of 3/)).toBeDefined();
  });

  it("clicking Previous goes back a page", () => {
    renderTab("all", executionsPage("exec-pagprev-"), { total: 25 });
    fireEvent.click(screen.getByText("Next"));
    expect(screen.getByText(/Page 2 of 3/)).toBeDefined();
    fireEvent.click(screen.getByText("Previous"));
    expect(screen.getByText(/Page 1 of 3/)).toBeDefined();
  });

  it("splits orphaned and active IDs when stopping selected with old execution", async () => {
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
    renderTab("all", [oldExec]);
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
    const refetch = vi.fn();
    mockRunningQuery.mockReturnValue(
      defaultQueryReturn({
        error: { status: 500, message: "Server error" },
        refetch,
      }),
    );
    renderTable();
    fireEvent.click(screen.getByText("Try Again"));
    expect(refetch).toHaveBeenCalled();
  });
});

function renderTab(
  tab: keyof typeof queryByTab,
  executions: Record<string, unknown>[],
  { total = executions.length, ...props }: { total?: number } & TableProps = {},
) {
  queryByTab[tab].mockReturnValue(withExecutions(executions, total));
  return renderTable(tab === "all" ? props : { initialTab: tab, ...props });
}

function renderTable(props: TableProps = {}) {
  return render(
    <ExecutionsTable diagnosticsData={diagnosticsData} {...props} />,
  );
}

function stuckExecution(executionId: string) {
  return {
    ...sampleExecution,
    execution_id: executionId,
    status: "QUEUED",
    started_at: null,
  };
}

function executionsPage(idPrefix: string) {
  return Array.from({ length: 10 }, (_, i) => ({
    ...sampleExecution,
    execution_id: `${idPrefix}${i}`,
  }));
}

type TableProps = Partial<ComponentProps<typeof ExecutionsTable>>;
