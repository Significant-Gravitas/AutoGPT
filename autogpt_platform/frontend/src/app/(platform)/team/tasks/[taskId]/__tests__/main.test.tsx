import {
  getCancelTaskMockHandler,
  getGetTaskMockHandler,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
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

const { expertsFlag } = vi.hoisted(() => ({ expertsFlag: { enabled: true } }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? { enabled: expertsFlag.enabled, ready: true }
        : actual.useFlagStatus(flag as never),
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
});

afterEach(() => {
  expertsFlag.enabled = true;
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
    expect(
      screen.getByText("Posted to the blog and shared the link."),
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
    expect(within(activity).getByText("Handed off")).toBeDefined();
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
});
