import { getGetExpertMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListTasksMockHandler } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

const { expertsFlag } = vi.hoisted(() => ({
  expertsFlag: { enabled: true },
}));

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
    useGetFlag: () => true,
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
  usePathname: () => "/team/expert-maria",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ expertId: "expert-maria" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "Maria is a senior marketing strategist.",
  skills: ["Content strategy"],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

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

const activeTask = makeTask();
const doneTask = makeTask({
  id: "task-done",
  title: "Published the Q3 recap",
  status: "DONE",
  spend_total: 40,
  outcome_summary: "Posted to the blog and shared the link.",
});

beforeEach(() => {
  expertsFlag.enabled = true;
  server.use(
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
  );
});

afterEach(() => {
  expertsFlag.enabled = true;
});

async function openTasksTab() {
  await userEvent.click(await screen.findByRole("tab", { name: /tasks/i }));
}

describe("Expert Tasks tab", () => {
  test("shows the expert's tasks in the same table as the team board", async () => {
    server.use(getListTasksMockHandler([activeTask, doneTask]));

    render(<ExpertDetailPage />);
    await openTasksTab();

    const table = await screen.findByRole("table", {
      name: "Delegated tasks",
    });
    expect(within(table).getByText("Draft the weekly report")).toBeDefined();
    expect(within(table).getByText("Working")).toBeDefined();
    expect(within(table).getByText("$2.50")).toBeDefined();
    expect(within(table).getByText("Published the Q3 recap")).toBeDefined();
    expect(within(table).getByText("Completed")).toBeDefined();
    expect(within(table).getByText("$0.40")).toBeDefined();
  });

  test("drops the owner column — every row would repeat the same expert", async () => {
    server.use(getListTasksMockHandler([activeTask]));

    render(<ExpertDetailPage />);
    await openTasksTab();

    await screen.findByRole("table", { name: "Delegated tasks" });
    expect(screen.queryByRole("columnheader", { name: /owner/i })).toBeNull();
  });

  test("shows an empty state when the expert has never been delegated to", async () => {
    server.use(getListTasksMockHandler([]));

    render(<ExpertDetailPage />);
    await openTasksTab();

    expect(
      await screen.findByText(
        /Nothing delegated yet. Ask this expert to do something/i,
      ),
    ).toBeDefined();
  });

  test("only queries the tasks the expert owns", async () => {
    const requested: (string | null)[] = [];
    server.use(
      http.get("/api/proxy/api/tasks", ({ request }) => {
        requested.push(new URL(request.url).searchParams.get("expert_id"));
        return HttpResponse.json([activeTask]);
      }),
    );

    render(<ExpertDetailPage />);
    await openTasksTab();

    await screen.findByText("Draft the weekly report");
    expect(requested).toContain("expert-maria");
  });

  test("each task row links to its detail page", async () => {
    server.use(getListTasksMockHandler([activeTask, doneTask]));

    render(<ExpertDetailPage />);
    await openTasksTab();

    const table = await screen.findByRole("table", {
      name: "Delegated tasks",
    });
    const rows = within(table).getAllByRole("row");
    const hrefs = rows.map((row) => row.getAttribute("href")).filter(Boolean);
    expect(hrefs).toContain("/team/tasks/task-active");
    expect(hrefs).toContain("/team/tasks/task-done");
  });

  test("the whole page, tab included, is gone when the experts flag is off", () => {
    expertsFlag.enabled = false;
    server.use(getListTasksMockHandler([activeTask]));

    expect(() => render(<ExpertDetailPage />)).toThrow("NEXT_NOT_FOUND");
  });
});
