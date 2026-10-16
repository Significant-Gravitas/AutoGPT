import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { ArchiveSessionsRequest } from "@/app/api/__generated__/models/archiveSessionsRequest";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { toast } from "sonner";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TaskBoard } from "../TaskBoard";

const auth = vi.hoisted(() => ({ user: { id: "board-user-a" } }));
vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: () => auth }));

function session(id: string): SessionSummaryResponse {
  return {
    id,
    title: `Task ${id}`,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    chat_status: "idle",
    is_processing: false,
    last_message: "Finished the report",
    file_count: 2,
  };
}

function showBoard(sessions = [session("one")]) {
  server.use(
    getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
  );
  return render(
    <>
      <TaskBoard />
      <Toaster />
    </>,
  );
}

beforeEach(() => {
  auth.user = { id: "board-user-a" };
  window.localStorage.clear();
  server.use(
    getListExpertIdentitiesMockHandler([]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListCopilotFollowupSchedulesMockHandler([]),
  );
});

afterEach(() => toast.dismiss());

it("keeps custom columns per account and restores them on return", async () => {
  const user = userEvent.setup();
  const view = showBoard();
  await user.click(await screen.findByRole("button", { name: "Add column" }));
  await user.type(
    screen.getByRole("textbox", { name: "Column name" }),
    "Private plans{Enter}",
  );
  expect(screen.getByRole("region", { name: "Private plans" })).toBeDefined();
  auth.user = { id: "board-user-b" };
  view.rerender(
    <>
      <TaskBoard />
      <Toaster />
    </>,
  );
  await screen.findByRole("region", { name: "Done" });
  expect(screen.queryByRole("region", { name: "Private plans" })).toBeNull();
  auth.user = { id: "board-user-a" };
  view.rerender(
    <>
      <TaskBoard />
      <Toaster />
    </>,
  );
  await screen.findByRole("region", { name: "Done" });
  expect(screen.getByRole("region", { name: "Private plans" })).toBeDefined();
});

it("persists manual moves and lets new activity return a task to In progress", async () => {
  const user = userEvent.setup();
  const view = showBoard();
  await user.click(
    await screen.findByRole("button", { name: "Move Task one" }),
  );
  await user.click(screen.getByRole("menuitem", { name: "In progress" }));
  expect(
    within(screen.getByRole("region", { name: "In progress" })).getByText(
      "Task one",
    ),
  ).toBeDefined();
  expect(
    window.localStorage.getItem("expert-task-board:board-user-a"),
  ).toContain("in-progress");
  view.unmount();
  const restored = showBoard();
  expect(
    window.localStorage.getItem("expert-task-board:board-user-a"),
  ).toContain("in-progress");
  await screen.findByText("Task one");
  expect(
    await within(
      await screen.findByRole("region", { name: "In progress" }),
    ).findByText("Task one"),
  ).toBeDefined();
  await user.click(screen.getByRole("button", { name: "Move Task one" }));
  await user.click(screen.getByRole("menuitem", { name: "Done" }));
  expect(
    within(screen.getByRole("region", { name: "Done" })).getByText("Task one"),
  ).toBeDefined();
  restored.unmount();
  showBoard([{ ...session("one"), updated_at: new Date().toISOString() }]);
  expect(
    await within(
      await screen.findByRole("region", { name: "In progress" }),
    ).findByText("Task one"),
  ).toBeDefined();
});

it.each([false, true])(
  "batches Clear and allows Undo after a partial failure: %s",
  async (failSecond) => {
    const user = userEvent.setup();
    const calls: ArchiveSessionsRequest[] = [];
    server.use(
      http.post("*/api/chat/sessions/archive", async ({ request }) => {
        const body = (await request.json()) as ArchiveSessionsRequest;
        calls.push(body);
        if (
          body.session_ids.length > 100 ||
          (failSecond && body.archived && calls.length === 2)
        ) {
          return HttpResponse.json({ detail: "Failed" }, { status: 422 });
        }
        return HttpResponse.json({ updated: body.session_ids.length });
      }),
    );
    showBoard(Array.from({ length: 101 }, (_, i) => session(String(i))));
    await user.click(
      await screen.findByRole("button", { name: "Archive every chat in Done" }),
    );
    await waitFor(() =>
      expect(calls.filter((call) => call.archived)).toHaveLength(2),
    );
    expect(calls.map((call) => call.session_ids.length)).toEqual([100, 1]);
    await user.click(await screen.findByRole("button", { name: "Undo" }));
    await waitFor(() =>
      expect(
        calls
          .filter((call) => !call.archived)
          .flatMap((call) => call.session_ids),
      ).toHaveLength(failSecond ? 100 : 101),
    );
  },
);

it("shows scheduled follow-ups and the files and clock badges on their chat", async () => {
  server.use(
    getListCopilotFollowupSchedulesMockHandler([
      {
        id: "follow-up",
        user_id: "board-user-a",
        name: "Daily report",
        message: "Prepare the daily report",
        session_id: "one",
        next_run_time: "2026-12-01T12:00:00Z",
      },
    ]),
  );
  showBoard();
  expect(await screen.findByText("Prepare the daily report")).toBeDefined();
  expect(screen.getByLabelText("2 files")).toBeDefined();
  expect(screen.getByLabelText(/Follow-up scheduled for/)).toBeDefined();
  expect(
    screen
      .getByRole("link", { name: /Prepare the daily report/ })
      .getAttribute("href"),
  ).toBe("/home?sessionId=one");
});

it("removing a custom column returns its chats to their default column", async () => {
  const user = userEvent.setup();
  showBoard();
  await user.click(await screen.findByRole("button", { name: "Add column" }));
  await user.type(
    screen.getByRole("textbox", { name: "Column name" }),
    "Later{Enter}",
  );
  await user.click(screen.getByRole("button", { name: "Move Task one" }));
  await user.click(screen.getByRole("menuitem", { name: "Later" }));
  expect(
    within(screen.getByRole("region", { name: "Later" })).getByText("Task one"),
  ).toBeDefined();
  await user.click(
    screen.getByRole("button", { name: "Later column options" }),
  );
  await user.click(screen.getByRole("menuitem", { name: "Remove column" }));
  expect(screen.queryByRole("region", { name: "Later" })).toBeNull();
  expect(
    within(screen.getByRole("region", { name: "Done" })).getByText("Task one"),
  ).toBeDefined();
});
