import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import AutopilotPage from "../page";

const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
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
        ? setFlagStatusMock()
        : actual.useFlagStatus(flag as never),
  };
});

const { notFoundMock } = vi.hoisted(() => ({ notFoundMock: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team/autopilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

function makeExpert(over: Partial<Expert> = {}): Expert {
  return {
    id: "expert-maria",
    name: "Maria",
    avatar_url: null,
    role: "Marketing Strategist",
    bio: null,
    skills: [],
    tagline: null,
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

const maria = makeExpert({
  skills: ["Copywriting", "SEO"],
  workflows: [
    {
      id: "wf-1",
      store_listing_version_id: "slv-1",
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "Content Calendar",
      description: "Plans a week of posts",
      schedule_cron: "40 7 * * *",
      schedule_id: "sched-1",
    },
    {
      id: "wf-2",
      store_listing_version_id: "slv-2",
      library_agent_id: "lib-2",
      graph_id: "graph-2",
      name: "SEO Audit",
      description: null,
      schedule_cron: "0 9 * * 1",
      schedule_id: null,
    },
  ],
});

const lee = makeExpert({
  id: "expert-lee",
  name: "Lee",
  role: "Researcher",
  skills: ["Copywriting", "Analytics"],
});

const firedSam = makeExpert({
  id: "expert-sam",
  name: "Sam",
  role: "Fired",
  is_archived: true,
  skills: ["Ghostwriting"],
});

const mariaSchedule: GraphExecutionJobInfo = {
  id: "sched-1",
  name: "Content Calendar",
  agent_name: "Content Calendar",
  user_id: "user-1",
  graph_id: "graph-1",
  graph_version: 1,
  cron: "40 7 * * *",
  input_data: {},
  next_run_time: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
  expert_id: "expert-maria",
};

/** A schedule nobody on the team owns, which Autopilot must not claim. */
const straySchedule: GraphExecutionJobInfo = {
  ...mariaSchedule,
  id: "sched-stray",
  name: "Nobody's Job",
  agent_name: "Nobody's Job",
  graph_id: "graph-stray",
  expert_id: null,
};

/** Reads the value of one of the summary's stat columns by its label. */
function getStatValue(container: HTMLElement, label: string) {
  const column = within(container).getByText(label).closest("div");
  return column?.querySelector("dd")?.textContent;
}

beforeEach(() => {
  server.use(
    getListExpertsMockHandler([maria, lee, firedSam]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([
      mariaSchedule,
      straySchedule,
    ]),
  );
});

afterEach(() => {
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  notFoundMock.mockReset();
});

async function openTab(name: string) {
  await userEvent.click(await screen.findByRole("tab", { name }));
}

describe("AutopilotPage", () => {
  test("renders the Autopilot profile with a chat link and a way back", async () => {
    render(<AutopilotPage />);

    expect(
      await screen.findByRole("heading", { name: "Autopilot" }),
    ).toBeDefined();
    expect(screen.getByText("Head of AI")).toBeDefined();
    expect(screen.getByText("Built in")).toBeDefined();
    expect(screen.getByText("Identity")).toBeDefined();
    expect(screen.getByText("Works with")).toBeDefined();
    expect(screen.getByText("Boundaries")).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Chat" }).getAttribute("href"),
    ).toBe("/copilot");
    expect(
      screen.getByRole("link", { name: "Back to Team" }).getAttribute("href"),
    ).toBe("/team");
  });

  test("sums the hired team in the summary and links each expert", async () => {
    render(<AutopilotPage />);

    const summary = await screen.findByRole("complementary", {
      name: "Team at a glance",
    });
    // Sam is fired, so the roster is two experts.
    expect(getStatValue(summary, "Experts")).toBe("2");
    // Only Maria's schedule belongs to the team; the stray one is ignored.
    expect(getStatValue(summary, "Schedules")).toBe("1");
    // Copywriting is shared, so three skills — and Sam's Ghostwriting is out.
    expect(getStatValue(summary, "Skills")).toBe("3");
    expect(getStatValue(summary, "Workflows")).toBe("2");

    expect(
      within(summary).getByRole("link", { name: /Maria/ }).getAttribute("href"),
    ).toBe("/team/expert-maria");
    expect(
      within(summary).getByRole("link", { name: /Lee/ }).getAttribute("href"),
    ).toBe("/team/expert-lee");
    expect(within(summary).queryByText("Sam")).toBeNull();
  });

  test("lists the team's schedules", async () => {
    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });

    await openTab("Schedules");

    const list = screen.getByRole("list", { name: "Expert schedules" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(1);
    expect(within(list).getByText("Content Calendar")).toBeDefined();
    expect(within(list).queryByText("Nobody's Job")).toBeNull();
  });

  test("groups workflows by the expert who owns them", async () => {
    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });

    await openTab("Workflows");

    const group = screen.getByRole("region", { name: "Maria workflows" });
    expect(
      within(group).getByRole("link", { name: /Maria/ }).getAttribute("href"),
    ).toBe("/team/expert-maria");
    const rows = within(group).getAllByTestId("autopilot-workflow-row");
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText("Content Calendar")).toBeDefined();
    expect(within(rows[0]).getByText(/Every day at 07:40/)).toBeDefined();
    expect(within(rows[1]).getByText("SEO Audit")).toBeDefined();
    expect(within(rows[1]).getByText("Needs setup")).toBeDefined();
    // Lee has no workflows, so gets no group.
    expect(screen.queryByRole("region", { name: "Lee workflows" })).toBeNull();
  });

  test("shows the team's skills de-duplicated and sorted", async () => {
    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });

    await openTab("Skills");

    const panel = screen.getByRole("tabpanel");
    const skills = within(panel)
      .getAllByText(/^(Analytics|Copywriting|SEO|Ghostwriting)$/)
      .map((node) => node.textContent);
    expect(skills).toEqual(["Analytics", "Copywriting", "SEO"]);
  });

  test("tells you when the team has nothing set up yet", async () => {
    server.use(
      getListExpertsMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    );

    render(<AutopilotPage />);

    const summary = await screen.findByRole("complementary", {
      name: "Team at a glance",
    });
    expect(getStatValue(summary, "Experts")).toBe("0");
    expect(within(summary).getByText(/No experts hired yet/)).toBeDefined();

    await openTab("Workflows");
    expect(screen.getByText(/No workflows yet/)).toBeDefined();

    await openTab("Skills");
    expect(screen.getByText(/No skills listed yet/)).toBeDefined();

    await openTab("Schedules");
    expect(screen.getByText(/No schedules yet/)).toBeDefined();
  });

  test("offers a retry when the team fails to load", async () => {
    const user = userEvent.setup();
    let failures = 0;
    server.use(
      http.get("/api/proxy/api/experts", () => {
        failures += 1;
        return failures === 1
          ? HttpResponse.json({ detail: "boom" }, { status: 500 })
          : HttpResponse.json([maria]);
      }),
    );

    render(<AutopilotPage />);

    expect(
      await screen.findByText("We could not load your team."),
    ).toBeDefined();
    await user.click(screen.getByRole("button", { name: /try again/i }));

    expect(
      await screen.findByRole("heading", { name: "Autopilot" }),
    ).toBeDefined();
    expect(failures).toBe(2);
  });

  test("404s when experts are switched off", () => {
    setFlagStatusMock.mockReturnValue({ enabled: false, ready: true });

    expect(() => render(<AutopilotPage />)).toThrow("NEXT_NOT_FOUND");
    expect(notFoundMock).toHaveBeenCalled();
  });
});
