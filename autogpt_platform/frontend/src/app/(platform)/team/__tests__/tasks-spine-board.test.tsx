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
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

// The board rides the experts flag. useGetFlag is what the AllTasksSection
// fork reads; useFlagStatus is what gates the page — mocking them separately
// models LaunchDarkly answering the page gate while useGetFlag still resolves
// its fail-closed default.
let expertsFlagEnabled = true;

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "hire-experts"
        ? expertsFlagEnabled
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
  expertsFlagEnabled = true;
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
  test("All tasks renders every receipt as a table row", async () => {
    const user = userEvent.setup();
    render(<TeamPage />);

    await user.click(await screen.findByRole("tab", { name: "All tasks" }));

    const table = await screen.findByRole("table", { name: "Delegated tasks" });
    const workingRow = within(table).getByRole("row", {
      name: /Draft the weekly report/,
    });
    expect(within(workingRow).getByText("Maria")).toBeDefined();
    expect(within(workingRow).getByText("Working")).toBeDefined();
    expect(within(workingRow).getByText("$2.50")).toBeDefined();

    // Autopilot work has no owning expert; the run-based board could never
    // show it because it only fans out per hired expert.
    const doneRow = within(table).getByRole("row", {
      name: /Summarise this week's runs/,
    });
    expect(within(doneRow).getByText("Autopilot")).toBeDefined();
  });

  test("status chips filter the table", async () => {
    const user = userEvent.setup();
    render(<TeamPage />);

    await user.click(await screen.findByRole("tab", { name: "All tasks" }));
    const table = await screen.findByRole("table", { name: "Delegated tasks" });

    await user.click(screen.getByRole("button", { name: /Active 1/ }));

    // Filtered-out rows collapse and go aria-hidden, so they vanish from the
    // accessibility tree while the matching row stays.
    await waitFor(() =>
      expect(
        within(table).queryByRole("row", {
          name: /Summarise this week's runs/,
        }),
      ).toBeNull(),
    );
    expect(
      within(table).getByRole("row", { name: /Draft the weekly report/ }),
    ).toBeDefined();
  });

  test("every row links to its own task page", async () => {
    const user = userEvent.setup();
    render(<TeamPage />);

    await user.click(await screen.findByRole("tab", { name: "All tasks" }));
    const table = await screen.findByRole("table", { name: "Delegated tasks" });

    expect(
      within(table)
        .getByRole("row", { name: /Summarise this week's runs/ })
        .getAttribute("href"),
    ).toBe("/team/tasks/task-autopilot");
    expect(
      within(table)
        .getByRole("row", { name: /Draft the weekly report/ })
        .getAttribute("href"),
    ).toBe("/team/tasks/task-active");
  });

  test("spine board reads /api/tasks and leaves the per-expert runs fan-out alone", async () => {
    const user = userEvent.setup();
    render(<TeamPage />);
    const requested: string[] = [];
    server.events.on("request:start", ({ request }) => {
      requested.push(new URL(request.url).pathname);
    });

    await user.click(await screen.findByRole("tab", { name: "All tasks" }));
    await screen.findByRole("table", { name: "Delegated tasks" });

    expect(requested.some((path) => path.endsWith("/api/tasks"))).toBe(true);
    expect(requested.some((path) => path.includes("/runs"))).toBe(false);
  });

  test("unresolved flag falls back to the run-based board and never touches /api/tasks", async () => {
    expertsFlagEnabled = false;
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
