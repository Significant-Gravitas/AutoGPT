import {
  getListExpertCredentialsMockHandler,
  getListExpertPodsMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
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

/** Reads the value of one of a card's stat columns by its label. */
function getStatValue(card: HTMLElement, label: string) {
  const column = within(card).getByText(label).closest("div");
  return column?.querySelector("dd")?.textContent;
}

const maria = makeExpert();
const lee = makeExpert({ id: "expert-lee", name: "Lee", role: "Researcher" });

beforeEach(() => {
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertPodsMockHandler([]),
    getListExpertCredentialsMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(),
  );
});

describe("TeamPage tabs", () => {
  test("opens on Team Overview and offers Pod board", async () => {
    server.use(getListExpertsMockHandler([maria]));

    render(<TeamPage />);

    const overview = await screen.findByRole("tab", { name: "Team Overview" });
    expect(overview.getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("tab", { name: "Pod board" })).toBeDefined();
    expect(screen.queryByRole("tab", { name: "All tasks" })).toBeNull();
  });

  test("Team Overview lists every expert without pod sections", async () => {
    const growth: ExpertPod = {
      id: "pod-growth",
      name: "Growth",
      created_at: new Date("2026-08-14T00:00:00Z"),
    };
    server.use(
      getListExpertsMockHandler([makeExpert({ pod_id: "pod-growth" }), lee]),
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

  test("the paused filter keeps only experts with paused schedules", async () => {
    const user = userEvent.setup();
    const pausedLee = makeExpert({
      id: "expert-lee",
      name: "Lee",
      schedules_paused_at: new Date("2026-08-20T10:00:00Z"),
    });
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
  test("counts the whole team's skills, schedules and workflows", async () => {
    const scheduledMaria = makeExpert({
      ...maria,
      skills: ["Copywriting", "SEO"],
      workflows: [
        {
          id: "wf-1",
          store_listing_version_id: "slv-1",
          library_agent_id: null,
          graph_id: "graph-1",
          name: "Content Calendar",
          description: null,
          schedule_cron: null,
          schedule_id: null,
        },
      ],
    });
    // Lee shares Copywriting with Maria, so the team has three skills, not four.
    const skilledLee = makeExpert({
      ...lee,
      skills: ["Copywriting", "Analytics"],
    });
    server.use(
      getListExpertsMockHandler([scheduledMaria, skilledLee]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        {
          id: "sched-soon",
          name: "Content Calendar",
          user_id: "user-1",
          graph_id: "graph-1",
          graph_version: 1,
          cron: "0 7 * * *",
          input_data: {},
          next_run_time: "2099-01-01T07:00:00Z",
          expert_id: "expert-maria",
        },
      ]),
    );

    render(<TeamPage />);
    expect(await screen.findByText("Lee")).toBeDefined();

    const autopilot = screen.getByRole("region", { name: "Autopilot" });
    expect(getStatValue(autopilot, "Skills")).toBe("3");
    expect(getStatValue(autopilot, "Schedules")).toBe("1");
    expect(getStatValue(autopilot, "Workflows")).toBe("1");
  });

  test("zeroes out the totals for a team with nothing set up", async () => {
    server.use(getListExpertsMockHandler([maria]));

    render(<TeamPage />);
    expect(await screen.findByText("Maria")).toBeDefined();

    const autopilot = screen.getByRole("region", { name: "Autopilot" });
    expect(getStatValue(autopilot, "Skills")).toBe("0");
    expect(getStatValue(autopilot, "Schedules")).toBe("0");
    expect(getStatValue(autopilot, "Workflows")).toBe("0");
    expect(within(autopilot).getByText("Built in")).toBeDefined();
    expect(within(autopilot).queryByText("Budget")).toBeNull();
    expect(
      within(autopilot)
        .getByRole("link", { name: "Chat" })
        .getAttribute("href"),
    ).toBe("/copilot");
    expect(within(autopilot).queryByRole("link", { name: "Edit" })).toBeNull();
  });

  test("links its body to the Autopilot page", async () => {
    server.use(getListExpertsMockHandler([maria]));

    render(<TeamPage />);
    expect(await screen.findByText("Maria")).toBeDefined();

    const autopilot = screen.getByRole("region", { name: "Autopilot" });
    expect(
      within(autopilot)
        .getByRole("link", { name: "View Autopilot" })
        .getAttribute("href"),
    ).toBe("/team/autopilot");
  });
});
