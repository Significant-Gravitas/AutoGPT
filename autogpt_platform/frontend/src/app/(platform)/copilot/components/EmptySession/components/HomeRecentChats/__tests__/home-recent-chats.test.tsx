import userEvent from "@testing-library/user-event";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListTasksMockHandler200 } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { Task } from "@/app/api/__generated__/models/task";
import {
  taskTabKey,
  useWorkAreaStore,
} from "@/app/(platform)/copilot/workArea";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeSession, makeTask } from "../../../__tests__/taskFixtures";
import { HomeRecentRow } from "../HomeRecentRow";

/** A chat updated `minutesAgo` before the first, so the list's order is
 *  fixed. */
function chat(
  id: string,
  title: string,
  minutesAgo: number,
  extra: Partial<SessionSummaryResponse> = {},
) {
  const updatedAt = new Date(
    Date.parse("2026-11-10T09:00:00Z") - minutesAgo * 60_000,
  ).toISOString();
  return makeSession(id, title, null, { updated_at: updatedAt, ...extra });
}

function task(id: string, title: string, extra: Partial<Task>): Task {
  return { ...makeTask(id, title, "DONE"), ...extra };
}

function serve(sessions: SessionSummaryResponse[], tasks: Task[] = []) {
  server.use(
    getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
    getListTasksMockHandler200({ tasks }),
  );
}

async function findList() {
  const region = await screen.findByRole("region", { name: "Recent chats" });
  return within(region);
}

function hrefs(links: HTMLElement[]) {
  return links.map((link) => link.getAttribute("href"));
}

describe("Home recent chats", () => {
  it("lists only main chats, and opens a chat's task chats beneath it under their task titles", async () => {
    const user = userEvent.setup();
    serve(
      [
        chat("worker-1", "Done: You are working on the venue shortlist", 0, {
          parent_session_id: "main-1",
        }),
        chat("run-1", "You are working on the launch email", 1, {
          work_task_id: "task-email",
        }),
        chat("main-1", "Plan the offsite", 2),
        chat("main-2", "Weekly report", 3),
      ],
      [
        task("task-venue", "Shortlist three venues", {
          session_ids: ["main-1", "worker-1"],
          work_session_id: "worker-1",
        }),
        task("task-email", "Draft the launch email", {
          session_ids: ["main-1", "run-1"],
          work_session_id: "run-1",
        }),
      ],
    );
    render(<HomeRecentRow expertId={null} />);

    const list = await findList();
    expect(hrefs(await list.findAllByRole("link"))).toEqual([
      "/home?sessionId=main-1",
      "/home?sessionId=main-2",
    ]);
    expect(list.queryByText(/You are working on/)).toBeNull();

    const toggle = await list.findByRole("button", {
      name: "2 tasks from Plan the offsite",
    });
    expect(toggle.textContent).toContain("2");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(
      list.queryByRole("button", { name: /from Weekly report/ }),
    ).toBeNull();

    await user.click(toggle);

    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    const branch = document.getElementById(
      toggle.getAttribute("aria-controls") ?? "",
    );
    expect(branch).not.toBeNull();
    const children = within(branch as HTMLElement).getAllByRole("link");
    expect(hrefs(children)).toEqual([
      "/home?sessionId=worker-1",
      "/home?sessionId=run-1",
    ]);
    expect(children[0].textContent).toContain("Shortlist three venues");
    expect(children[1].textContent).toContain("Draft the launch email");
    expect(list.queryByText(/You are working on/)).toBeNull();

    await user.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
  });

  it("opens a task chat's task in the work area from the button beside its row", async () => {
    const user = userEvent.setup();
    useWorkAreaStore.getState().reset();
    serve(
      [
        chat("worker-1", "You are working on the venue shortlist", 0, {
          parent_session_id: "main-1",
          is_processing: true,
        }),
        chat("main-1", "Plan the offsite", 1, { is_processing: true }),
      ],
      [
        task("task-venue", "Shortlist three venues", {
          session_ids: ["main-1", "worker-1"],
          work_session_id: "worker-1",
        }),
      ],
    );
    render(<HomeRecentRow expertId={null} />);

    const list = await findList();
    await user.click(
      await list.findByRole("button", {
        name: "1 task from Plan the offsite",
      }),
    );
    const open = await list.findByRole("button", {
      name: "Open task details: Shortlist three venues",
    });
    expect(open.closest("a")).toBeNull();

    await user.click(open);

    const { tabs, activeKey } = useWorkAreaStore.getState();
    expect(activeKey).toBe(taskTabKey("task-venue"));
    expect(tabs).toContainEqual(
      expect.objectContaining({ kind: "task", taskID: "task-venue" }),
    );
    useWorkAreaStore.getState().reset();
  });

  it("marks working chats, main and task alike, with the sidebar's turning mark", async () => {
    const user = userEvent.setup();
    serve(
      [
        chat("worker-1", "You are working on the venue shortlist", 0, {
          parent_session_id: "main-1",
          is_processing: true,
        }),
        chat("main-1", "Plan the offsite", 1, { is_processing: true }),
        chat("main-2", "Weekly report", 2),
      ],
      [
        task("task-venue", "Shortlist three venues", {
          session_ids: ["main-1", "worker-1"],
          work_session_id: "worker-1",
        }),
      ],
    );
    render(<HomeRecentRow expertId={null} />);

    const list = await findList();
    await user.click(
      await list.findByRole("button", {
        name: "1 task from Plan the offsite",
      }),
    );
    const [main, child, done] = await list.findAllByRole("link");

    for (const link of [main, child]) {
      expect(link.textContent).toMatch(/^Working: /);
      expect(link.querySelector('[data-state="working"]')).not.toBeNull();
      expect(link.querySelector(".animate-spin")).toBeNull();
    }
    expect(done.textContent).toMatch(/^Done: Weekly report/);
    expect(done.querySelector('[data-state="done"]')).not.toBeNull();
  });

  it("leaves out task chats whose main chat is not in the list", async () => {
    serve(
      [
        chat("tab-run", "You are working on the board task", 0, {
          work_task_id: "task-board",
        }),
        chat("stray-worker", "You are working on a stray task", 1, {
          parent_session_id: "gone-chat",
        }),
        chat("main-1", "Plan the offsite", 2),
      ],
      [
        task("task-board", "Started from the Tasks tab", {
          session_ids: ["tab-run"],
          work_session_id: "tab-run",
        }),
      ],
    );
    render(<HomeRecentRow expertId={null} />);

    const list = await findList();
    expect(hrefs(await list.findAllByRole("link"))).toEqual([
      "/home?sessionId=main-1",
    ]);
    expect(list.queryByRole("button", { name: /tasks? from/ })).toBeNull();
    expect(list.queryByText("Started from the Tasks tab")).toBeNull();
  });

  it("lists every chat of the recipient, not just the newest six", async () => {
    const sessions = Array.from({ length: 9 }, (_, i) =>
      chat(`main-${i}`, `Chat number ${i}`, i),
    );
    serve(sessions);
    render(<HomeRecentRow expertId={null} />);

    const list = await findList();
    expect(await list.findAllByRole("link")).toHaveLength(9);
    expect(list.getByText("Chat number 8")).toBeDefined();
  });

  describe("scrolling to the end", () => {
    beforeEach(() => {
      vi.stubGlobal(
        "IntersectionObserver",
        class {
          constructor(
            private callback: (entries: { isIntersecting: boolean }[]) => void,
          ) {}
          observe() {
            this.callback([{ isIntersecting: true }]);
          }
          disconnect() {}
          unobserve() {}
        },
      );
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("loads the recipient's older chats", async () => {
      const newest = chat("main-new", "Newest chat", 0);
      const oldest = chat("main-old", "Oldest chat", 600);
      server.use(
        getGetV2ListSessionsMockHandler200(({ request }) => {
          const offset = Number(
            new URL(request.url).searchParams.get("offset"),
          );
          return offset === 0
            ? { sessions: [newest], total: 2 }
            : { sessions: [oldest], total: 2 };
        }),
        getListTasksMockHandler200({ tasks: [] }),
      );
      render(<HomeRecentRow expertId={null} />);

      const list = await findList();
      expect(await list.findByText("Oldest chat")).toBeDefined();
      expect(hrefs(list.getAllByRole("link"))).toEqual([
        "/home?sessionId=main-new",
        "/home?sessionId=main-old",
      ]);
    });
  });

  it("shows pinned chats in their own box beside the recent ones, each with a pin button", async () => {
    serve(
      [
        chat("pinned-1", "Quarterly plan", 0, { is_pinned: true }),
        chat("main-1", "Plan the offsite", 1),
      ],
      [],
    );
    render(<HomeRecentRow expertId={null} />);

    const pinned = within(
      await screen.findByRole("region", { name: "Pinned chats" }),
    );
    expect(hrefs(pinned.getAllByRole("link"))).toEqual([
      "/home?sessionId=pinned-1",
    ]);
    expect(
      pinned.getByRole("button", { name: "Unpin Quarterly plan" }),
    ).toBeDefined();

    const recent = within(screen.getByRole("region", { name: "Recent chats" }));
    expect(hrefs(recent.getAllByRole("link"))).toEqual([
      "/home?sessionId=main-1",
    ]);
    expect(
      recent.getByRole("button", { name: "Pin Plan the offsite" }),
    ).toBeDefined();
  });
});
