import {
  getListExpertPodsMockHandler,
  getListExpertRunsMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListTasksMockHandler } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

let taskSpineEnabled = true;

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "task-spine"
        ? taskSpineEnabled
        : actual.useGetFlag(flag as never),
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? { enabled: true, ready: true }
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
  usePathname: () => "/team",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: vi.fn(),
}));

function makeExpert(over: Partial<Expert> = {}): Expert {
  return {
    id: "expert-maria",
    name: "Maria",
    avatar_url: null,
    role: "Marketing Strategist",
    bio: null,
    skills: [],
    tagline: "Grows your brand while you sleep",
    identity: "You are Maria.",
    voice_preferences: null,
    boundaries: null,
    protected_soul_rules: [],
    is_template: false,
    source_template_id: "template-maria",
    is_archived: false,
    workflows: [],
    ...over,
  } as Expert;
}

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

const expertTask = makeTask();
const autopilotTask = makeTask({
  id: "task-autopilot",
  title: "Summarise this week's runs",
  status: "DONE",
  owner: null,
  ancestor_expert_ids: [],
  outcome_summary: "Summary posted.",
});

beforeEach(() => {
  taskSpineEnabled = true;
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertPodsMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(),
    getListExpertsMockHandler([makeExpert()]),
    getListExpertRunsMockHandler([]),
    getListTasksMockHandler([expertTask, autopilotTask]),
  );
});

afterEach(() => {
  server.events.removeAllListeners("request:start");
});

describe("team board on the task spine", () => {
  test("All tasks shows delegated receipts split into Active and History", async () => {
    const user = userEvent.setup();
    render(<TeamPage />);

    await user.click(await screen.findByRole("tab", { name: "All tasks" }));

    const active = await screen.findByRole("list", { name: "Active tasks" });
    expect(within(active).getByText("Draft the weekly report")).toBeDefined();
    expect(within(active).getByText("Maria")).toBeDefined();

    const history = screen.getByRole("list", { name: "History tasks" });
    expect(
      within(history).getByText("Summarise this week's runs"),
    ).toBeDefined();
    // Autopilot work has no owning expert; the run-based board could never
    // show it because it only fans out per hired expert.
    expect(within(history).getByText("Autopilot")).toBeDefined();
  });

  test("spine board reads /api/tasks and leaves the per-expert runs fan-out alone", async () => {
    const user = userEvent.setup();
    render(<TeamPage />);
    const requested: string[] = [];
    server.events.on("request:start", ({ request }) => {
      requested.push(new URL(request.url).pathname);
    });

    await user.click(await screen.findByRole("tab", { name: "All tasks" }));
    await screen.findByRole("list", { name: "Active tasks" });

    expect(requested.some((path) => path.endsWith("/api/tasks"))).toBe(true);
    expect(requested.some((path) => path.includes("/runs"))).toBe(false);
  });

  test("flag off keeps the run-based board and never touches /api/tasks", async () => {
    taskSpineEnabled = false;
    const user = userEvent.setup();
    render(<TeamPage />);
    const requested: string[] = [];
    server.events.on("request:start", ({ request }) => {
      requested.push(new URL(request.url).pathname);
    });

    await user.click(await screen.findByRole("tab", { name: "All tasks" }));
    expect(
      await screen.findByText(
        "No completed work yet. Finished runs will show up here.",
      ),
    ).toBeDefined();

    expect(requested.some((path) => path.endsWith("/api/tasks"))).toBe(false);
  });
});
