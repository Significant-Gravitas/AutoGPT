import {
  getDeleteV1DeleteExecutionScheduleMockHandler,
  getDeleteV1DeleteExecutionScheduleMockHandler422,
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import FollowupsPage from "../page";

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    useToast: () => ({ toast: toastMock }),
  };
});

function makeFollowup(
  overrides: Partial<CopilotTurnJobInfo>,
): CopilotTurnJobInfo {
  const runAt = new Date(Date.now() + 60 * 60 * 1000);
  return {
    id: "sched-1",
    name: "copilot-followup",
    user_id: "user-1",
    session_id: "session-abcdef0123",
    message: "Check the build status and report back",
    cron: null,
    run_at: runAt,
    next_run_time: runAt.toISOString(),
    kind: "copilot_turn",
    timezone: "UTC",
    cap_retry_count: 0,
    ...overrides,
  };
}

function makeGraphSchedule(
  overrides: Partial<GraphExecutionJobInfo>,
): GraphExecutionJobInfo {
  return {
    id: "graph-sched-1",
    name: "Daily summary",
    user_id: "user-1",
    graph_id: "graph-abc",
    graph_version: 1,
    agent_name: "Daily summary agent",
    cron: "0 9 * * *",
    input_data: {},
    next_run_time: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
    kind: "graph",
    timezone: "UTC",
    ...overrides,
  };
}

// Default zero-graph-schedules handler so tests that only care about
// followups don't have to mock the graph-schedules endpoint too.  The
// page now fetches BOTH on mount (unified Scheduled view).
function defaultGraphSchedulesHandler() {
  return getGetV1ListExecutionSchedulesForAUserMockHandler([]);
}

describe("FollowupsPage", () => {
  beforeEach(() => {
    toastMock.mockClear();
  });

  afterEach(() => {
    server.resetHandlers();
  });

  test("renders empty state when no follow-ups exist", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([]),
      defaultGraphSchedulesHandler(),
    );

    render(<FollowupsPage />);

    expect(await screen.findByTestId("followups-empty")).toBeDefined();
    expect(screen.queryByTestId("followups-list")).toBeNull();
  });

  test("renders one row per follow-up returned by the API", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([
        makeFollowup({ id: "f1", message: "First follow-up message" }),
        makeFollowup({
          id: "f2",
          message: "Second follow-up message",
          session_id: "session-zzzzzzzzzz",
        }),
      ]),
      defaultGraphSchedulesHandler(),
    );

    render(<FollowupsPage />);

    const rows = await screen.findAllByTestId("followup-row");
    expect(rows).toHaveLength(2);
    expect(rows[0].getAttribute("data-followup-id")).toBe("f1");
    expect(rows[1].getAttribute("data-followup-id")).toBe("f2");
    expect(screen.getByText("First follow-up message")).toBeDefined();
    expect(screen.getByText("Second follow-up message")).toBeDefined();
  });

  test("session link points to /copilot with the session id query param", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([
        makeFollowup({ id: "f1", session_id: "session-abcdef0123" }),
      ]),
      defaultGraphSchedulesHandler(),
    );

    render(<FollowupsPage />);

    const row = await screen.findByTestId("followup-row");
    const link = within(row).getByTestId("followup-open-session");
    expect(link.getAttribute("href")).toBe(
      "/copilot?sessionId=session-abcdef0123",
    );
  });

  test("renders a fresh-chat sentinel row (session_id=null) without a session link", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([
        makeFollowup({ id: "f1", session_id: null as unknown as string }),
      ]),
      defaultGraphSchedulesHandler(),
    );

    render(<FollowupsPage />);

    const row = await screen.findByTestId("followup-row");
    // No clickable session link when there is no target session yet.
    expect(within(row).queryByTestId("followup-open-session")).toBeNull();
    // Renders the "New chat" pill instead so the user can tell the
    // follow-up will spawn a fresh conversation at fire time.
    expect(within(row).getByText("New chat")).toBeDefined();
    // Delete still works on the sentinel row.
    expect(within(row).getByTestId("followup-delete-button")).toBeDefined();
  });

  test("Cancel button opens the confirmation dialog and calls the delete API", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([makeFollowup({ id: "f1" })]),
      defaultGraphSchedulesHandler(),
      getDeleteV1DeleteExecutionScheduleMockHandler(),
    );

    render(<FollowupsPage />);

    const cancelButton = await screen.findByTestId("followup-delete-button");
    fireEvent.click(cancelButton);

    const confirmButton = await screen.findByTestId("followup-confirm-delete");
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Follow-up deleted" }),
      );
    });
  });

  test("unified page renders graph schedule rows alongside copilot followups", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([
        makeFollowup({ id: "f1", message: "First follow-up message" }),
      ]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({ id: "g1", agent_name: "Nightly cleanup" }),
      ]),
    );

    render(<FollowupsPage />);

    // Followup row still renders.
    expect(await screen.findByText("First follow-up message")).toBeDefined();
    // Graph-kind row renders with its agent name + the "Agent run" badge.
    expect(screen.getByText("Nightly cleanup")).toBeDefined();
    const graphRow = screen.getByTestId("schedule-row");
    expect(graphRow.getAttribute("data-schedule-kind")).toBe("graph");
    expect(
      within(graphRow).getByTestId("schedule-kind-badge").textContent,
    ).toBe("Agent run");
  });

  test("shows a destructive toast when the delete API fails", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([makeFollowup({ id: "f1" })]),
      defaultGraphSchedulesHandler(),
      getDeleteV1DeleteExecutionScheduleMockHandler422(),
    );

    render(<FollowupsPage />);

    const cancelButton = await screen.findByTestId("followup-delete-button");
    fireEvent.click(cancelButton);

    const confirmButton = await screen.findByTestId("followup-confirm-delete");
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to delete follow-up",
          variant: "destructive",
        }),
      );
    });
  });

  test("graph row: cron is humanized when present, falls back to 'Runs once' when null", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({
          id: "g-cron",
          agent_name: "Daily summary",
          cron: "0 9 * * *",
        }),
        makeGraphSchedule({
          id: "g-once",
          agent_name: "One shot",
          // Backend returns null for one-shot graph schedules; the
          // generated type says ``string`` but we exercise the runtime
          // null-guard in `useGraphScheduleListItem` so cast through.
          cron: null as unknown as string,
        }),
      ]),
    );

    render(<FollowupsPage />);

    const rows = await screen.findAllByTestId("schedule-row");
    expect(rows).toHaveLength(2);
    // Recurring schedule humanizes the cron expression — the cron lib
    // turns ``0 9 * * *`` into a phrase starting with "At 9:00 AM" or
    // similar. We only assert the row does NOT render the raw cron
    // string or "Runs once".
    const cronRow = rows.find(
      (r) => r.getAttribute("data-schedule-id") === "g-cron",
    )!;
    expect(within(cronRow).queryByText("Runs once")).toBeNull();
    expect(within(cronRow).queryByText("0 9 * * *")).toBeNull();
    // One-shot (cron=null) renders the "Runs once" label.
    const onceRow = rows.find(
      (r) => r.getAttribute("data-schedule-id") === "g-once",
    )!;
    expect(within(onceRow).getByText("Runs once")).toBeDefined();
  });

  test("graph row: agentLabel falls back to schedule name then 'Scheduled agent' when agent_name missing", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({
          id: "g-fallback-name",
          agent_name: "" as unknown as string,
          name: "My schedule name",
        }),
        makeGraphSchedule({
          id: "g-fallback-default",
          agent_name: "" as unknown as string,
          name: "",
        }),
      ]),
    );

    render(<FollowupsPage />);

    // Second-tier fallback: schedule name renders when agent_name is empty.
    expect(await screen.findByText("My schedule name")).toBeDefined();
    // Final fallback: the literal "Scheduled agent" renders when BOTH
    // agent_name and name are empty.
    expect(screen.getByText("Scheduled agent")).toBeDefined();
  });

  test("graph row: clicking View opens the dialog with graph metadata", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({
          id: "g-view",
          agent_name: "Nightly cleanup",
          graph_id: "graph-xyz",
          graph_version: 7,
        }),
      ]),
    );

    render(<FollowupsPage />);

    const row = await screen.findByTestId("schedule-row");
    fireEvent.click(within(row).getByTestId("schedule-view-button"));

    // Dialog renders graph_id + version line.
    expect(await screen.findByText(/graph-xyz/)).toBeDefined();
    expect(screen.getByText(/v7/)).toBeDefined();
  });

  test("graph row: delete success path shows the success toast", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({ id: "g-del" }),
      ]),
      getDeleteV1DeleteExecutionScheduleMockHandler(),
    );

    render(<FollowupsPage />);

    const row = await screen.findByTestId("schedule-row");
    fireEvent.click(within(row).getByTestId("schedule-delete-button"));

    const confirmButton = await screen.findByTestId("schedule-confirm-delete");
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Schedule deleted" }),
      );
    });
  });

  test("graph row: delete error path shows the destructive toast", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({ id: "g-del-err" }),
      ]),
      getDeleteV1DeleteExecutionScheduleMockHandler422(),
    );

    render(<FollowupsPage />);

    const row = await screen.findByTestId("schedule-row");
    fireEvent.click(within(row).getByTestId("schedule-delete-button"));

    const confirmButton = await screen.findByTestId("schedule-confirm-delete");
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to delete schedule",
          variant: "destructive",
        }),
      );
    });
  });

  test("unified page sorts items by next_run_time ascending", async () => {
    const soon = new Date(Date.now() + 5 * 60 * 1000).toISOString(); // 5 min
    const later = new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(); // 2 h
    server.use(
      getListCopilotFollowupSchedulesMockHandler([
        makeFollowup({
          id: "f-later",
          message: "Later followup",
          next_run_time: later,
        }),
      ]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({
          id: "g-soon",
          agent_name: "Sooner agent",
          next_run_time: soon,
        }),
      ]),
    );

    render(<FollowupsPage />);

    const list = await screen.findByTestId("followups-list");
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    // Sooner graph schedule comes first; later followup second.
    expect(within(items[0]).getByText("Sooner agent")).toBeDefined();
    expect(within(items[1]).getByText("Later followup")).toBeDefined();
  });
});
