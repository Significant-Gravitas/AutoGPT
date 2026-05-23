import {
  getDeleteV1DeleteExecutionScheduleMockHandler,
  getDeleteV1DeleteExecutionScheduleMockHandler422,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
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

describe("FollowupsPage", () => {
  beforeEach(() => {
    toastMock.mockClear();
  });

  afterEach(() => {
    server.resetHandlers();
  });

  test("renders empty state when no follow-ups exist", async () => {
    server.use(getListCopilotFollowupSchedulesMockHandler([]));

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

  test("shows a destructive toast when the delete API fails", async () => {
    server.use(
      getListCopilotFollowupSchedulesMockHandler([makeFollowup({ id: "f1" })]),
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
});
