import {
  getListExpertPodsMockHandler,
  getListExpertRunsMockHandler,
  getListExpertRunsMockHandler401,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { ExpertRun } from "@/app/api/__generated__/models/expertRun";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
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

function makeRun(over: Partial<ExpertRun> = {}): ExpertRun {
  return {
    execution_id: "exec-1",
    graph_id: "graph-1",
    agent_name: "Content Calendar",
    library_agent_id: "lib-1",
    status: "COMPLETED",
    output_type: "text",
    output_key: "output",
    needs_review: false,
    started_at: new Date("2026-08-20T10:00:00Z"),
    ended_at: new Date("2026-08-20T10:05:00Z"),
    link: null,
    ...over,
  } as ExpertRun;
}

const maria = makeExpert();
const lee = makeExpert({ id: "expert-lee", name: "Lee", role: "Researcher" });

beforeEach(() => {
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertPodsMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(),
    getListExpertRunsMockHandler([]),
  );
});

describe("TeamPage tabs", () => {
  test("opens on Team Overview and offers Pod board and All tasks", async () => {
    server.use(getListExpertsMockHandler([maria]));

    render(<TeamPage />);

    const overview = await screen.findByRole("tab", { name: "Team Overview" });
    expect(overview.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("tab", { name: "Pod board" })).toBeDefined();
    expect(screen.getByRole("tab", { name: "All tasks" })).toBeDefined();
  });

  test("Team Overview lists every expert without pod sections", async () => {
    const growth: ExpertPod = {
      id: "pod-growth",
      name: "Growth",
      created_at: new Date("2026-08-14T00:00:00Z"),
    };
    server.use(
      getListExpertsMockHandler([
        makeExpert({ pod_id: "pod-growth" } as Partial<Expert>),
        lee,
      ]),
      getListExpertPodsMockHandler([growth]),
    );

    render(<TeamPage />);

    expect(await screen.findByText("Maria")).toBeDefined();
    expect(screen.getByText("Lee")).toBeDefined();
    // Pod grouping belongs to the Pod board tab, not the overview grid.
    expect(screen.queryByRole("heading", { name: "Growth" })).toBeNull();
  });

  test("Pod board tells you how to start when there are no pods and no experts", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([]));

    render(<TeamPage />);
    await user.click(await screen.findByRole("tab", { name: "Pod board" }));

    expect(await screen.findByText("No pods yet")).toBeDefined();
  });

  test("Pod board still shows ungrouped experts when there are no pods", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([maria]));

    render(<TeamPage />);
    await user.click(await screen.findByRole("tab", { name: "Pod board" }));

    expect(
      await screen.findByRole("heading", { name: "Ungrouped" }),
    ).toBeDefined();
    expect(screen.getByText("Maria")).toBeDefined();
    expect(screen.queryByText("No pods yet")).toBeNull();
  });

  test("All tasks merges runs from every hired expert", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([maria, lee]),
      getListExpertRunsMockHandler(({ params }) =>
        params.expertId === "expert-maria"
          ? [makeRun({ started_at: new Date("2026-08-20T09:00:00Z") })]
          : [
              makeRun({
                execution_id: "exec-2",
                agent_name: "Market Scan",
                started_at: new Date("2026-08-20T11:00:00Z"),
              }),
            ],
      ),
    );

    render(<TeamPage />);
    await user.click(await screen.findByRole("tab", { name: "All tasks" }));

    const list = await screen.findByRole("list");
    const rows = within(list).getAllByRole("listitem");
    // Newest first: Lee's 11:00 run sorts above Maria's 09:00 run.
    expect(within(rows[0]).getByText("Market Scan")).toBeDefined();
    expect(within(rows[0]).getByText("Lee")).toBeDefined();
    expect(within(rows[1]).getByText("Content Calendar")).toBeDefined();
    expect(within(rows[1]).getByText("Maria")).toBeDefined();
  });

  test("All tasks filters down to what needs review", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([maria]),
      getListExpertRunsMockHandler([
        makeRun(),
        makeRun({
          execution_id: "exec-2",
          agent_name: "Draft Newsletter",
          needs_review: true,
        }),
      ]),
    );

    render(<TeamPage />);
    await user.click(await screen.findByRole("tab", { name: "All tasks" }));

    await user.click(
      await screen.findByRole("button", { name: /needs review/i }),
    );

    expect(await screen.findByText("Draft Newsletter")).toBeDefined();
    expect(screen.queryByText("Content Calendar")).toBeNull();
  });

  test("All tasks offers a retry when every expert's runs fail to load", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([maria]),
      getListExpertRunsMockHandler401(),
    );

    render(<TeamPage />);
    await user.click(await screen.findByRole("tab", { name: "All tasks" }));

    expect(await screen.findByText("Something went wrong")).toBeDefined();
  });

  test("All tasks points at hiring when the team is empty", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([]));

    render(<TeamPage />);
    await user.click(await screen.findByRole("tab", { name: "All tasks" }));

    expect(
      await screen.findByText(
        "Hire an expert and their finished work will show up here.",
      ),
    ).toBeDefined();
  });
});

describe("TeamRoster toolbar", () => {
  test("search narrows the roster to matching experts", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([maria, lee]));

    render(<TeamPage />);
    expect(await screen.findByText("Maria")).toBeDefined();

    await user.type(
      screen.getByRole("searchbox", { name: "Search experts" }),
      "resear",
    );

    // Matches Lee on role, not name.
    expect(screen.getByText("Lee")).toBeDefined();
    expect(screen.queryByText("Maria")).toBeNull();
    // Autopilot is pinned, but steps aside once the roster is narrowed.
    expect(screen.queryByText("Autopilot")).toBeNull();
  });

  test("says so when nothing matches the search", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([maria]));

    render(<TeamPage />);
    expect(await screen.findByText("Maria")).toBeDefined();

    await user.type(
      screen.getByRole("searchbox", { name: "Search experts" }),
      "nobody",
    );

    expect(
      screen.getByText("No experts match that search or filter."),
    ).toBeDefined();
  });

  test("the table view lists the same experts as rows", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([maria, lee]));

    render(<TeamPage />);
    expect(await screen.findByText("Maria")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Table view" }));

    const table = screen.getByRole("table");
    expect(within(table).getByText("Maria")).toBeDefined();
    expect(within(table).getByText("Lee")).toBeDefined();
    expect(
      within(table).getByRole("columnheader", { name: "Expert" }),
    ).toBeDefined();
    // Autopilot is not an expert, so it stays out of the table.
    expect(within(table).queryByText("Autopilot")).toBeNull();
  });

  test("the paused filter keeps only experts with paused schedules", async () => {
    const user = userEvent.setup();
    const pausedLee = makeExpert({
      id: "expert-lee",
      name: "Lee",
      schedules_paused_at: new Date("2026-08-20T10:00:00Z"),
    } as Partial<Expert>);
    server.use(getListExpertsMockHandler([maria, pausedLee]));

    render(<TeamPage />);
    expect(await screen.findByText("Maria")).toBeDefined();

    await user.click(screen.getByRole("combobox", { name: /filter/i }));
    await user.click(await screen.findByRole("option", { name: "Paused" }));

    expect(screen.getByText("Lee")).toBeDefined();
    expect(screen.queryByText("Maria")).toBeNull();
  });
});

describe("AutopilotCard", () => {
  test("surfaces the soonest run across the whole team", async () => {
    server.use(
      getListExpertsMockHandler([maria, lee]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        {
          id: "sched-late",
          name: "Weekly Report",
          user_id: "user-1",
          graph_id: "graph-1",
          graph_version: 1,
          cron: "0 9 * * 1",
          input_data: {},
          next_run_time: "2099-01-02T09:00:00Z",
          expert_id: "expert-maria",
        },
        {
          id: "sched-soon",
          name: "Content Calendar",
          user_id: "user-1",
          graph_id: "graph-2",
          graph_version: 1,
          cron: "0 7 * * *",
          input_data: {},
          next_run_time: "2099-01-01T07:00:00Z",
          expert_id: "expert-lee",
        },
      ]),
    );

    render(<TeamPage />);
    expect(await screen.findByText("Lee")).toBeDefined();

    const autopilot = within(screen.getByRole("region", { name: "Autopilot" }));
    // Lee's schedule fires first, so it wins over Maria's later one.
    expect(autopilot.getByText(/Content Calendar/).textContent).toContain(
      "Content Calendar",
    );
    expect(autopilot.queryByText(/Weekly Report/)).toBeNull();
  });

  test("says all clear when nothing is waiting on the user", async () => {
    server.use(getListExpertsMockHandler([maria]));

    render(<TeamPage />);
    expect(await screen.findByText("Maria")).toBeDefined();

    const autopilot = within(screen.getByRole("region", { name: "Autopilot" }));
    expect(autopilot.getByText("All clear")).toBeDefined();
    expect(autopilot.getByText("Nothing scheduled")).toBeDefined();
  });
});
