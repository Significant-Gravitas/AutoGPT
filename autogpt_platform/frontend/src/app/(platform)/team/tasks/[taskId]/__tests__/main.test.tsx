import {
  getAnswerTaskMockHandler,
  getCancelTaskMockHandler,
  getGetTaskMockHandler,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { getListWorkspaceFilesMockHandler } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TaskDetailPage from "../page";

const { expertsFlag, taskManagementFlag } = vi.hoisted(() => ({
  expertsFlag: { enabled: true },
  taskManagementFlag: { enabled: true },
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) => {
      if (flag === "hire-experts")
        return { enabled: expertsFlag.enabled, ready: true };
      if (flag === "expert-task-management")
        return { enabled: taskManagementFlag.enabled, ready: true };
      return actual.useFlagStatus(flag as never);
    },
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team/tasks/task-active",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ taskId: "task-active" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

function makeTask(overrides: Partial<DelegatedTask> = {}): DelegatedTask {
  return {
    id: "task-active",
    title: "Draft the weekly report",
    spec: "Run Weekly Report with:\n- week: current",
    status: "WORKING",
    acceptance: "PENDING",
    created_by_type: "USER",
    created_by_id: "user-1",
    owner: {
      id: "expert-maria",
      name: "Maria",
      avatar_url: null,
      role: "Marketing Strategist",
    },
    parent_task_id: null,
    root_task_id: "task-active",
    origin_session_id: "session-1",
    ancestor_expert_ids: ["expert-maria"],
    handoff_count: 0,
    revision_count: 0,
    spend_total: 250,
    outcome_summary: null,
    amendments: [],
    created_at: new Date("2026-08-30T09:00:00Z"),
    updated_at: new Date("2026-08-30T09:00:00Z"),
    runs: [],
    ...overrides,
  };
}

beforeEach(() => {
  expertsFlag.enabled = true;
  taskManagementFlag.enabled = true;
  server.use(getListWorkspaceFilesMockHandler({ files: [] }));
});

afterEach(() => {
  expertsFlag.enabled = true;
  taskManagementFlag.enabled = true;
});

describe("task detail page", () => {
  test("renders the spec, outcome and linked runs", async () => {
    const done = makeTask({
      status: "DONE",
      outcome_summary: "Posted to the blog and shared the link.",
      runs: [
        {
          execution_id: "run-1",
          graph_id: "graph-1",
          library_agent_id: "lib-1",
          agent_name: "Blog Publisher",
          status: "COMPLETED",
          started_at: null,
          ended_at: null,
          link: "/library/agents/lib-1?activeTab=runs&activeItem=run-1",
        },
      ],
    });
    server.use(getGetTaskMockHandler({ task: done, children: [] }));

    render(<TaskDetailPage />);

    expect(
      await screen.findByRole("heading", { name: "Draft the weekly report" }),
    ).toBeDefined();
    expect(screen.getByText(/Run Weekly Report with:/)).toBeDefined();
    // The outcome reads as the activity chain's last stop, not a panel of
    // its own — the review controls ride along with it.
    const activity = screen.getByRole("list", { name: "Activity" });
    expect(
      within(activity).getByText("Posted to the blog and shared the link."),
    ).toBeDefined();
    expect(
      within(activity).getByRole("button", { name: "Accept" }),
    ).toBeDefined();
    expect(screen.getByText("Blog Publisher")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Open run Blog Publisher" }),
    ).toBeDefined();
    // A finished task can't be cancelled.
    expect(screen.queryByRole("button", { name: "Cancel task" })).toBeNull();
  });

  test("renders the subtask tree with depth indent and links to each subtask", async () => {
    const child = makeTask({
      id: "task-child",
      title: "Write the launch copy",
      parent_task_id: "task-active",
      owner: {
        id: "expert-leo",
        name: "Leo",
        avatar_url: null,
        role: "Copywriter",
      },
    });
    const grandchild = makeTask({
      id: "task-grandchild",
      title: "Proofread the copy",
      status: "DONE",
      parent_task_id: "task-child",
      owner: null,
    });
    server.use(
      getGetTaskMockHandler({
        task: makeTask({
          amendments: [
            {
              at: new Date("2026-08-30T10:00:00Z"),
              by: "expert-maria",
              note: "Needs Leo's integrations.",
              kind: "handoff",
              from_expert_id: "expert-maria",
              to_expert_id: "expert-leo",
            },
          ],
        }),
        children: [child, grandchild],
      }),
    );

    render(<TaskDetailPage />);

    const tree = await screen.findByRole("list", { name: "Subtasks" });
    const rows = within(tree).getAllByRole("listitem");
    expect(rows).toHaveLength(2);

    const childLink = within(rows[0]).getByRole("link");
    expect(childLink.textContent).toContain("Write the launch copy");
    expect(childLink.getAttribute("href")).toBe("/team/tasks/task-child");
    expect(childLink.style.paddingLeft).toBe("12px");

    // A grandchild indents one level past its parent.
    const grandchildLink = within(rows[1]).getByRole("link");
    expect(grandchildLink.textContent).toContain("Proofread the copy");
    expect(grandchildLink.style.paddingLeft).toBe("32px");

    const activity = screen.getByRole("list", { name: "Activity" });
    // Feed-style entry: "<who> <did what>" header, quoted note beneath.
    expect(within(activity).getByText("handed this off")).toBeDefined();
    expect(within(activity).getByText("Maria")).toBeDefined();
    expect(
      within(activity).getByText("Needs Leo's integrations."),
    ).toBeDefined();
  });

  test("activity spells out the Autopilot hop and links back to its thread", async () => {
    server.use(getGetTaskMockHandler({ task: makeTask(), children: [] }));

    render(<TaskDetailPage />);

    // The user asked Autopilot, which routed the work to Maria — both hops
    // show rather than collapsing into "you asked Maria".
    const activity = await screen.findByRole("list", { name: "Activity" });
    const asked = within(activity).getByRole("link", {
      name: /You asked Autopilot/,
    });
    expect(asked.getAttribute("href")).toBe("/copilot?sessionId=session-1");
    expect(
      within(activity).getByRole("link", {
        name: /Autopilot delegated to Maria/,
      }),
    ).toBeDefined();
  });

  test("a hire's first task is not attributed to an Autopilot chat", async () => {
    server.use(
      getGetTaskMockHandler({
        task: makeTask({ created_by_type: "HIRE" }),
        children: [],
      }),
    );

    render(<TaskDetailPage />);

    const activity = await screen.findByRole("list", { name: "Activity" });
    expect(
      within(activity).getByText(/You hired Maria — this came with them/),
    ).toBeDefined();
    expect(within(activity).queryByText(/You asked Autopilot/)).toBeNull();
    expect(screen.getByText("First task")).toBeDefined();
  });

  test("a change request reads below the outcome it rejected", async () => {
    server.use(
      getGetTaskMockHandler({
        task: makeTask({
          status: "WORKING",
          outcome_summary: "Drafted the case study outline.",
          amendments: [
            {
              at: new Date("2026-08-30T11:00:00Z"),
              by: "user",
              note: "I want instagram also",
              kind: "revision",
            },
          ],
        }),
        children: [],
      }),
    );

    render(<TaskDetailPage />);

    const activity = await screen.findByRole("list", { name: "Activity" });
    const rows = within(activity).getAllByRole("listitem");
    const outcomeRow = rows.findIndex((row) =>
      row.textContent?.includes("completed this task"),
    );
    const revisionRow = rows.findIndex((row) =>
      row.textContent?.includes("requested changes"),
    );
    expect(outcomeRow).toBeGreaterThan(-1);
    expect(revisionRow).toBeGreaterThan(-1);
    expect(outcomeRow).toBeLessThan(revisionRow);
  });

  test("answers an escalation inline from the activity feed", async () => {
    const answerSpy = vi.fn(() => makeTask({ status: "WORKING" }));
    server.use(
      getGetTaskMockHandler({
        task: makeTask({
          status: "WAITING_USER",
          amendments: [
            {
              at: new Date("2026-08-30T10:00:00Z"),
              by: "expert-maria",
              note: "",
              kind: "escalation",
              question: "Where do your client accounts live?",
              options: ["Google Sheet", "HubSpot"],
              session_id: "session-2",
              target: "user",
            },
          ],
        }),
        children: [],
      }),
      getAnswerTaskMockHandler(answerSpy),
    );

    render(<TaskDetailPage />);

    const activity = await screen.findByRole("list", { name: "Activity" });
    expect(
      within(activity).getByText("Where do your client accounts live?"),
    ).toBeDefined();

    // One-click option answers without leaving the page.
    await userEvent.click(screen.getByRole("button", { name: "Google Sheet" }));
    await waitFor(() => expect(answerSpy).toHaveBeenCalled());
  });

  test("clamps a long activity quote behind Read more", async () => {
    server.use(
      getGetTaskMockHandler({
        task: makeTask({
          status: "DONE",
          outcome_summary: "All the process detail nobody asked for. ".repeat(
            12,
          ),
        }),
        children: [],
      }),
    );

    render(<TaskDetailPage />);

    const readMore = await screen.findByRole("button", { name: "Read more" });
    await userEvent.click(readMore);
    expect(screen.getByRole("button", { name: "Show less" })).toBeDefined();
  });

  test("shows the session's files and the runs' credentials as cards", async () => {
    server.use(
      getGetTaskMockHandler({
        task: makeTask({
          credentials: [
            {
              id: "cred-1",
              provider: "openai",
              title: "My OpenAI key",
            },
          ],
        }),
        children: [],
      }),
      getListWorkspaceFilesMockHandler({
        files: [
          {
            id: "file-1",
            name: "q3-recap.md",
            path: "/sessions/session-1/q3-recap.md",
            mime_type: "text/markdown",
            size_bytes: 2048,
            origin: "generated",
            created_at: "2026-08-30T09:05:00Z",
          },
        ],
      }),
    );

    render(<TaskDetailPage />);

    const files = await screen.findByRole("list", { name: "Task files" });
    const fileLink = within(files).getByRole("link", { name: /q3-recap.md/ });
    expect(fileLink.getAttribute("href")).toBe(
      "/api/proxy/api/workspace/files/file-1/download",
    );
    expect(within(files).getByText(/Generated · 2.0 KB/)).toBeDefined();

    const creds = screen.getByRole("list", { name: "Credentials used" });
    expect(within(creds).getByText("My OpenAI key")).toBeDefined();
    expect(within(creds).getByText("openai")).toBeDefined();
  });

  test("cancels an open task", async () => {
    const cancelSpy = vi.fn(() => ({
      task: makeTask({ status: "CANCELLED" }),
      children: [],
    }));
    server.use(
      getGetTaskMockHandler({ task: makeTask(), children: [] }),
      getCancelTaskMockHandler(cancelSpy),
    );

    render(<TaskDetailPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Cancel task" }),
    );

    await waitFor(() => expect(cancelSpy).toHaveBeenCalled());
  });

  test("the page is gone when the experts flag is off", () => {
    expertsFlag.enabled = false;
    server.use(getGetTaskMockHandler({ task: makeTask(), children: [] }));

    expect(() => render(<TaskDetailPage />)).toThrow("NEXT_NOT_FOUND");
  });

  test("the page is gone when task management is off, even with experts on", () => {
    taskManagementFlag.enabled = false;
    server.use(getGetTaskMockHandler({ task: makeTask(), children: [] }));

    expect(() => render(<TaskDetailPage />)).toThrow("NEXT_NOT_FOUND");
  });
});
