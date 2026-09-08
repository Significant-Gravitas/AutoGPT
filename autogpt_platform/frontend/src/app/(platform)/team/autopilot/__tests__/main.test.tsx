import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetLibraryAgentMockHandler200,
  getGetV2ListLibraryAgentsMockHandler200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
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

function makeLibraryAgent(over: Partial<LibraryAgent>): LibraryAgent {
  return {
    id: "lib-free",
    graph_id: "graph-free",
    graph_version: 1,
    name: "Inbox Triage",
    description: "Sorts the morning mail",
    creator_name: "You",
    image_url: null,
    marketplace_listing: null,
    execution_count: 3,
    ...over,
  } as unknown as LibraryAgent;
}

/** Maria owns this one, so it is hers and not Autopilot's. */
const mariasAgent = makeLibraryAgent({
  id: "lib-1",
  graph_id: "graph-1",
  name: "Content Calendar",
});

const freeAgent = makeLibraryAgent({});

const strayAgent = makeLibraryAgent({
  id: "lib-stray",
  graph_id: "graph-stray",
  name: "Nobody's Job",
  description: "",
});

function libraryResponse(agents: LibraryAgent[]): LibraryAgentResponse {
  return {
    agents,
    pagination: {
      total_items: agents.length,
      total_pages: 1,
      current_page: 1,
      page_size: 100,
    },
  } as LibraryAgentResponse;
}

const librarySkills = [
  { name: "SEO", description: "Rank pages", triggers: [] },
  { name: "Research", description: "Dig up sources", triggers: ["find"] },
  { name: "Analytics", description: "Read the numbers", triggers: [] },
  { name: "Bookkeeping", description: "Balance the books", triggers: [] },
];

beforeEach(() => {
  server.use(
    getListExpertsMockHandler([maria, lee, firedSam]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([
      mariaSchedule,
      straySchedule,
    ]),
    getGetV2ListLibraryAgentsMockHandler200(
      libraryResponse([mariasAgent, freeAgent, strayAgent]),
    ),
    getGetV2GetLibraryAgentMockHandler200(freeAgent),
    getListCopilotSkillsMockHandler200(librarySkills),
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

  test("lists the team's schedules", async () => {
    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });

    await openTab("Schedules");

    const list = screen.getByRole("list", { name: "Expert schedules" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(1);
    expect(within(list).getByText("Content Calendar")).toBeDefined();
    expect(within(list).queryByText("Nobody's Job")).toBeNull();
  });

  test("lists the library workflows no expert owns as Autopilot's", async () => {
    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });

    await openTab("Workflows");

    expect(screen.getByText("Autopilot's Workflows")).toBeDefined();
    const rows = await screen.findAllByTestId("expert-workflow-row");
    expect(
      rows.map((row) => within(row).getByText(/Job|Triage/).textContent),
    ).toEqual(["Inbox Triage", "Nobody's Job"]);
    // The stray schedule's graph gives its workflow a cadence.
    expect(within(rows[1]).getByText(/Every day at 07:40/)).toBeDefined();
    // Maria's Content Calendar is hers, not Autopilot's.
    expect(screen.queryByText("Content Calendar")).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Install workflow" }),
    ).toBeNull();
    expect(
      within(rows[0])
        .getByRole("link", { name: "Ask about this workflow" })
        .getAttribute("href"),
    ).toContain("/copilot?autosubmit=true");
  });

  test("keeps loading library pages so a second-page workflow still counts as Autopilot's", async () => {
    const pagesRequested: string[] = [];
    server.use(
      getGetV2ListLibraryAgentsMockHandler200(({ request }) => {
        const page = new URL(request.url).searchParams.get("page") ?? "1";
        pagesRequested.push(page);
        const agents = page === "2" ? [strayAgent] : [mariasAgent, freeAgent];
        return {
          ...libraryResponse(agents),
          pagination: {
            total_items: 3,
            total_pages: 2,
            current_page: Number(page),
            page_size: 2,
          },
        } as LibraryAgentResponse;
      }),
    );

    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });
    await openTab("Workflows");

    expect(await screen.findByText("Nobody's Job")).toBeDefined();
    expect(pagesRequested).toEqual(["1", "2"]);
  });

  test("lists the library skills no expert has claimed as Autopilot's", async () => {
    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });

    await openTab("Skills");

    expect(screen.getByText("Autopilot's Skills")).toBeDefined();
    const list = screen.getByRole("list", { name: "Autopilot skills" });
    const rows = within(list).getAllByTestId("expert-skill-row");
    expect(
      rows.map(
        (row) => within(row).getByText(/^(Bookkeeping|Research)$/).textContent,
      ),
    ).toEqual(["Bookkeeping", "Research"]);
    expect(within(list).getByText("Dig up sources")).toBeDefined();
    expect(within(list).queryByRole("button", { name: /Remove/ })).toBeNull();
  });

  test("tells you when the team has nothing set up yet", async () => {
    server.use(
      getListExpertsMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([]),
      getGetV2ListLibraryAgentsMockHandler200(libraryResponse([])),
      getListCopilotSkillsMockHandler200([]),
    );

    render(<AutopilotPage />);
    await screen.findByRole("heading", { name: "Autopilot" });

    await openTab("Workflows");
    expect(screen.getByText(/No workflows yet/)).toBeDefined();

    await openTab("Skills");
    expect(screen.getByText(/No skills yet/)).toBeDefined();

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
