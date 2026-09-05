import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import {
  getArchiveExpertMockHandler,
  getGetExpertActivityMockHandler,
  getGetExpertDetachPreviewMockHandler,
  getGetExpertMockHandler,
  getInstallExpertWorkflowMockHandler,
  getListExpertRunsMockHandler,
  getListExpertsMockHandler,
  getRemoveExpertWorkflowMockHandler204,
  getResumeExpertSchedulesMockHandler,
  getUpdateExpertSkillsMockHandler200,
  getUpdateExpertAvatarMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetLibraryAgentMockHandler200,
  getGetV2ListLibraryAgentsMockHandler200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentResponse } from "@/app/api/__generated__/models/libraryAgentResponse";
import { ExpertActivityDay } from "@/app/api/__generated__/models/expertActivityDay";
import { ExpertRun } from "@/app/api/__generated__/models/expertRun";
import {
  getDeleteV1DeleteExecutionScheduleMockHandler,
  getGetV1ListExecutionSchedulesForAUserMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { getGetV2ListStoreAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/store/store.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { format, subDays } from "date-fns";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

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

const { uploadAvatarSpy } = vi.hoisted(() => ({ uploadAvatarSpy: vi.fn() }));
vi.mock("@/lib/direct-upload", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/direct-upload")>();
  return { ...actual, uploadSubmissionMediaDirect: uploadAvatarSpy };
});

const { notFoundMock, pushMock } = vi.hoisted(() => ({
  notFoundMock: vi.fn(),
  pushMock: vi.fn(),
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
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
    notFoundMock();
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
  protected_soul_rules: [
    "The expert discloses that it is AI when acting externally.",
    "External actions require approval.",
  ],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
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
      chain: [
        { kind: "input", provider: null },
        { kind: "integration", provider: "google" },
        { kind: "ai", provider: null },
      ],
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
};

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

const mariaRuns: ExpertRun[] = [
  {
    execution_id: "run-1",
    graph_id: "graph-1",
    agent_name: "Weekly Report",
    library_agent_id: "lib-1",
    status: "completed",
    output_type: "table",
    output_key: "result",
    needs_review: false,
    started_at: null,
    ended_at: null,
    link: "/library/agents/lib-1?activeTab=runs&activeItem=run-1",
  },
  {
    execution_id: "run-2",
    graph_id: "graph-2",
    agent_name: "SEO Audit",
    library_agent_id: "lib-2",
    status: "review",
    output_type: "doc",
    output_key: "report",
    needs_review: true,
    started_at: null,
    ended_at: null,
    link: "/library/agents/lib-2?activeTab=runs&activeItem=run-2",
  },
];

const ACTIVITY_WINDOW_DAYS = 365;

/** The backend sends `day` as a bare `YYYY-MM-DD`, which the client keeps as
 *  a string despite the generated `Date` type. */
function activityDays(
  counts: Record<number, { sessions?: number; runs?: number }>,
): ExpertActivityDay[] {
  return Array.from({ length: ACTIVITY_WINDOW_DAYS }, (_, index) => {
    const daysAgo = ACTIVITY_WINDOW_DAYS - 1 - index;
    return {
      day: format(
        subDays(new Date(), daysAgo),
        "yyyy-MM-dd",
      ) as unknown as Date,
      sessions: counts[daysAgo]?.sessions ?? 0,
      runs: counts[daysAgo]?.runs ?? 0,
    };
  });
}

const ownLibraryAgent = {
  id: "lib-private",
  graph_id: "graph-private",
  graph_version: 1,
  name: "My Private Agent",
  description: "Never published to the marketplace",
  creator_name: "You",
  image_url: null,
  marketplace_listing: null,
} as unknown as LibraryAgent;

function libraryResponse(agents: LibraryAgent[]): LibraryAgentResponse {
  return {
    agents,
    pagination: {
      total_items: agents.length,
      total_pages: 1,
      current_page: 1,
      page_size: 10,
    },
  } as LibraryAgentResponse;
}

beforeEach(() => {
  server.use(
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([mariaSchedule]),
    getListExpertRunsMockHandler([]),
    getGetExpertActivityMockHandler({
      timezone: "UTC",
      days: activityDays({}),
    }),
  );
});

afterEach(() => {
  window.localStorage.removeItem("team-workflows-view");
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  pushMock.mockReset();
});

async function openTab(name: string) {
  await userEvent.click(await screen.findByRole("tab", { name }));
}

describe("ExpertDetailPage", () => {
  test("renders the expert profile with workflows and schedule state", async () => {
    render(<ExpertDetailPage />);

    expect(await screen.findByRole("heading", { name: "Maria" })).toBeDefined();
    expect(screen.getByText("Marketing Strategist")).toBeDefined();
    expect(
      screen.getByText("Maria is a senior marketing strategist."),
    ).toBeDefined();
    expect(screen.getByText("Identity")).toBeDefined();
    expect(
      screen.getByText("You are Maria, a senior marketing strategist."),
    ).toBeDefined();
    expect(screen.getByText("Voice")).toBeDefined();
    expect(screen.getByText("Warm, concise, and direct.")).toBeDefined();
    expect(screen.getByText("Boundaries")).toBeDefined();
    expect(screen.getByText("Never invent customer evidence.")).toBeDefined();

    await openTab("Skills");
    expect(screen.getByText("Content strategy")).toBeDefined();

    await openTab("Workflows");
    const workflowRows = screen.getAllByTestId("expert-workflow-row");
    expect(workflowRows).toHaveLength(2);
    expect(within(workflowRows[0]).getByText("Content Calendar")).toBeDefined();
    expect(
      within(workflowRows[0]).getByText(/Every day at 07:40/),
    ).toBeDefined();
    expect(within(workflowRows[1]).getByText("SEO Audit")).toBeDefined();
    expect(within(workflowRows[1]).getByText("Needs setup")).toBeDefined();
  });

  test("keeps the budget above the tabs and the summary in Basics", async () => {
    render(<ExpertDetailPage />);

    const budget = await screen.findByRole("region", {
      name: "Maria budget",
    });
    const summary = await screen.findByRole("complementary", {
      name: "Maria at a glance",
    });
    const basicsPanel = screen.getByRole("tabpanel");
    expect(within(basicsPanel).getByRole("complementary")).toBe(summary);
    expect(
      within(basicsPanel).queryByRole("region", { name: "Maria budget" }),
    ).toBeNull();
    expect(
      within(summary).getByRole("region", { name: "Maria activity" }),
    ).toBeDefined();
    expect(
      within(summary).getByRole("region", { name: "Maria activity streak" }),
    ).toBeDefined();
    expect(within(summary).queryByText("Schedules")).toBeNull();
    expect(within(summary).queryByText("Skills")).toBeNull();
    expect(within(summary).queryByText("Workflows")).toBeNull();

    await openTab("Work");
    expect(screen.getByRole("region", { name: "Maria budget" })).toBe(budget);
    expect(
      screen.queryByRole("complementary", { name: "Maria at a glance" }),
    ).toBeNull();
  });

  test("opens the Soul panel from Basics", async () => {
    const user = userEvent.setup();
    render(<ExpertDetailPage />);

    await user.click(await screen.findByRole("button", { name: "Edit Soul" }));

    expect(
      screen.getByRole("complementary", { name: "Maria's Soul" }),
    ).toBeDefined();
    expect(
      screen.getByRole("textbox", { name: "Identity and personality" }),
    ).toBeDefined();
  });

  test("shows the workflow's block chain with provider logos and icons", async () => {
    const user = userEvent.setup();
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    await user.click(screen.getByRole("button", { name: "Grid view" }));
    const [calendar, seo] = screen.getAllByTestId("expert-workflow-row");
    const chain = within(calendar).getByTestId("workflow-chain");
    expect(
      within(chain).getByRole("img", { name: "Agent input" }),
    ).toBeDefined();
    expect(within(chain).getByRole("img", { name: "google" })).toBeDefined();
    expect(within(chain).getByRole("img", { name: "AI model" })).toBeDefined();
    expect(within(seo).queryByTestId("workflow-chain")).toBeNull();
  });

  test("shows each workflow's total runs from its library agent", async () => {
    server.use(
      getGetV2GetLibraryAgentMockHandler200({
        ...ownLibraryAgent,
        id: "lib-1",
        execution_count: 12,
      } as LibraryAgent),
    );
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    const [calendar] = screen.getAllByTestId("expert-workflow-row");
    expect(await within(calendar).findByText(/12 runs/)).toBeDefined();
  });

  test("opens the workflow's library page from the card", async () => {
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    const link = screen.getByRole("link", {
      name: "Open Content Calendar tasks",
    });
    expect(link.getAttribute("href")).toBe("/library/agents/lib-1");
    const [calendar] = screen.getAllByTestId("expert-workflow-row");
    expect(
      within(calendar)
        .getByRole("link", { name: "Edit workflow" })
        .getAttribute("href"),
    ).toBe("/build?flowID=graph-1");
    expect(within(calendar).queryByText("See tasks")).toBeNull();
  });

  test("opens the run dialog from the card's Run button", async () => {
    const user = userEvent.setup();
    server.use(
      getGetV2GetLibraryAgentMockHandler200({
        ...ownLibraryAgent,
        id: "lib-1",
        name: "Content Calendar",
      } as LibraryAgent),
    );
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    const [calendar] = screen.getAllByTestId("expert-workflow-row");
    await user.click(
      await within(calendar).findByRole("button", { name: "Run" }),
    );

    expect(await screen.findByRole("dialog")).toBeDefined();
  });

  test("removes a workflow from the expert after confirming", async () => {
    const user = userEvent.setup();
    const deleted: string[] = [];
    let removed = false;
    server.use(
      getGetExpertMockHandler(() =>
        removed ? { ...maria, workflows: [maria.workflows[1]] } : maria,
      ),
      getRemoveExpertWorkflowMockHandler204(({ params }) => {
        deleted.push(String(params.workflowId));
        removed = true;
        return undefined;
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    const [calendar] = screen.getAllByTestId("expert-workflow-row");
    await user.click(
      within(calendar).getByRole("button", { name: "More actions" }),
    );
    await user.click(
      await screen.findByRole("menuitem", { name: "Remove from expert" }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: "Remove Content Calendar?",
    });
    await user.click(within(dialog).getByRole("button", { name: "Remove" }));

    await waitFor(() => {
      expect(screen.getAllByTestId("expert-workflow-row")).toHaveLength(1);
    });
    expect(deleted).toEqual(["wf-1"]);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  test("titles the schedules tab and searches schedules", async () => {
    const user = userEvent.setup();
    render(<ExpertDetailPage />);

    await openTab("Schedules");
    expect(screen.getByText("Maria's Schedules")).toBeDefined();
    expect(await screen.findByTestId("schedule-row")).toBeDefined();

    await user.type(
      screen.getByRole("searchbox", { name: "Search schedules" }),
      "nothing-like-this",
    );
    expect(screen.getByText("No schedules match.")).toBeDefined();
  });

  test("lists the expert's workflows in the create schedule dialog", async () => {
    const user = userEvent.setup();
    render(<ExpertDetailPage />);

    await openTab("Schedules");
    await user.click(screen.getByRole("button", { name: "Create schedule" }));

    const dialog = await screen.findByRole("dialog", {
      name: "Create schedule",
    });
    const list = within(dialog).getByRole("list", {
      name: "Schedulable workflows",
    });
    expect(within(list).getByText("Content Calendar")).toBeDefined();
    expect(within(list).getByText("SEO Audit")).toBeDefined();
  });

  test("defaults workflows to the stacked list and can switch to the grid", async () => {
    const user = userEvent.setup();
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    const list = screen.getByTestId("workflow-list");
    expect(within(list).getAllByTestId("expert-workflow-row")).toHaveLength(2);
    expect(
      within(list).getAllByRole("button", { name: "More actions" }),
    ).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "Grid view" }));
    expect(screen.queryByTestId("workflow-list")).toBeNull();
    expect(screen.getAllByTestId("expert-workflow-row")).toHaveLength(2);
  });

  test("remembers the chosen workflows view in local storage", async () => {
    const user = userEvent.setup();
    const { unmount } = render(<ExpertDetailPage />);

    await openTab("Workflows");
    await user.click(screen.getByRole("button", { name: "Grid view" }));
    expect(window.localStorage.getItem("team-workflows-view")).toBe("grid");
    unmount();

    render(<ExpertDetailPage />);
    await openTab("Workflows");
    await waitFor(() => {
      expect(screen.queryByTestId("workflow-list")).toBeNull();
    });
    expect(screen.getAllByTestId("expert-workflow-row")).toHaveLength(2);
  });

  test("stacks the credentials a workflow will use beside Run", async () => {
    server.use(
      getGetV2GetLibraryAgentMockHandler200({
        ...ownLibraryAgent,
        id: "lib-1",
        credentials_input_schema: {
          type: "object",
          properties: {
            github: { credentials_provider: ["github"] },
            notion: { credentials_provider: ["notion"] },
            slack: { credentials_provider: ["slack"] },
            linear: { credentials_provider: ["linear"] },
            discord: { credentials_provider: ["discord"] },
          },
        },
      } as LibraryAgent),
    );
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    const [calendar] = screen.getAllByTestId("expert-workflow-row");
    const stack = await within(calendar).findByRole("list", {
      name: "Credentials used",
    });
    expect(within(stack).getAllByRole("listitem")).toHaveLength(4);
    expect(within(stack).getByText("+3")).toBeDefined();
  });

  test("lists the expert's skills with library details and adds one", async () => {
    const user = userEvent.setup();
    const puts: string[][] = [];
    let skills = ["Content strategy"];
    server.use(
      getGetExpertMockHandler(() => ({ ...maria, skills })),
      getListCopilotSkillsMockHandler200([
        {
          name: "Content strategy",
          description: "How to plan a content calendar",
          triggers: ["plan content"],
        },
        { name: "Deep Research", description: "Research anything thoroughly" },
      ]),
      getUpdateExpertSkillsMockHandler200(async ({ request }) => {
        const body = (await request.json()) as { skills: string[] };
        puts.push(body.skills);
        skills = body.skills;
        return { ...maria, skills };
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Skills");
    expect(screen.getByText("Maria's Skills")).toBeDefined();
    const list = screen.getByRole("list", { name: "Expert skills" });
    expect(
      await within(list).findByText("How to plan a content calendar"),
    ).toBeDefined();
    expect(within(list).getByText("plan content")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Add skill" }));
    const dialog = await screen.findByRole("dialog", { name: "Add a skill" });
    expect(within(dialog).queryByText("Content strategy")).toBeNull();
    await user.click(within(dialog).getByRole("button", { name: "Add" }));

    await waitFor(() => {
      expect(puts).toEqual([["Content strategy", "Deep Research"]]);
    });
    expect(await within(list).findByText("Deep Research")).toBeDefined();
  });

  test("offers marketplace skills as a second source in the add dialog", async () => {
    const user = userEvent.setup();
    server.use(
      getListCopilotSkillsMockHandler200([]),
      getGetV2ListStoreAgentsMockHandler200({
        agents: [
          {
            slug: "seo-audit",
            agent_name: "SEO Audit Pro",
            agent_image: "",
            creator: "acme",
            creator_avatar: "",
            sub_heading: "Audit any page for SEO gaps",
            description: "",
            runs: 3,
            rating: 4.5,
            agent_graph_id: "graph-seo",
          },
        ],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Skills");
    await user.click(screen.getByRole("button", { name: "Add skill" }));
    const dialog = await screen.findByRole("dialog", { name: "Add a skill" });
    await user.click(within(dialog).getByRole("tab", { name: "Marketplace" }));

    const list = await within(dialog).findByRole("list", {
      name: "Marketplace skills",
    });
    expect(within(list).getByText("SEO Audit Pro")).toBeDefined();
  });

  test("removes a skill from the expert", async () => {
    const user = userEvent.setup();
    const puts: string[][] = [];
    let skills = ["Content strategy"];
    server.use(
      getGetExpertMockHandler(() => ({ ...maria, skills })),
      getListCopilotSkillsMockHandler200([]),
      getUpdateExpertSkillsMockHandler200(async ({ request }) => {
        const body = (await request.json()) as { skills: string[] };
        puts.push(body.skills);
        skills = body.skills;
        return { ...maria, skills };
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Skills");
    await user.click(
      await screen.findByRole("button", { name: "Remove Content strategy" }),
    );

    await waitFor(() => {
      expect(puts).toEqual([[]]);
    });
    expect(await screen.findByText(/No skills yet/)).toBeDefined();
  });

  test("searches and filters the workflow list", async () => {
    const user = userEvent.setup();
    render(<ExpertDetailPage />);

    await openTab("Workflows");
    expect(screen.getAllByTestId("expert-workflow-row")).toHaveLength(2);

    await user.type(
      screen.getByRole("searchbox", { name: "Search workflows" }),
      "seo",
    );
    expect(screen.getAllByTestId("expert-workflow-row")).toHaveLength(1);
    expect(screen.getByText("SEO Audit")).toBeDefined();

    await user.clear(
      screen.getByRole("searchbox", { name: "Search workflows" }),
    );
    await user.click(screen.getByRole("button", { name: "Filter workflows" }));
    await user.click(
      await screen.findByRole("menuitemradio", { name: "Scheduled" }),
    );
    expect(screen.getAllByTestId("expert-workflow-row")).toHaveLength(1);
    expect(screen.getByText("Content Calendar")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Filter workflows" }));
    await user.click(
      await screen.findByRole("menuitemradio", { name: "Manual" }),
    );
    expect(screen.getByText("No workflows match.")).toBeDefined();
  });

  test("toggles the Soul panel closed from the header button", async () => {
    const user = userEvent.setup();
    render(<ExpertDetailPage />);

    const editSoul = await screen.findByRole("button", { name: "Edit Soul" });
    await user.click(editSoul);
    expect(
      screen.getByRole("complementary", { name: "Maria's Soul" }),
    ).toBeDefined();

    await user.click(editSoul);
    await waitFor(() => {
      expect(
        screen.queryByRole("complementary", { name: "Maria's Soul" }),
      ).toBeNull();
    });
  });

  test("draws the activity graph and marks the expert active this week", async () => {
    server.use(
      getGetExpertActivityMockHandler({
        timezone: "UTC",
        days: activityDays({
          0: { sessions: 2, runs: 1 },
          1: { runs: 1 },
          2: { sessions: 1 },
          29: { sessions: 1 },
        }).slice(-31),
      }),
    );

    render(<ExpertDetailPage />);

    expect(await screen.findByText("Active this week")).toBeDefined();
    expect(
      screen.getByRole("img", { name: "Activity over the last 3 months" }),
    ).toBeDefined();
    expect(
      screen.getByText("4 sessions · 2 runs · last 3 months"),
    ).toBeDefined();
    const graph = screen.getByTestId("expert-activity-graph");
    expect(graph.querySelectorAll("[title]")).toHaveLength(90);
    const streak = screen.getByRole("region", {
      name: "Maria activity streak",
    });
    expect(within(streak).getByText("3")).toBeDefined();
    expect(within(streak).getByText("day streak")).toBeDefined();
    const today = screen.getByTitle(
      `2 sessions, 1 run on ${format(new Date(), "MMM d")}`,
    );
    expect(today.getAttribute("data-level")).toBe("4");
    const lighter = screen.getByTitle(
      `1 session on ${format(subDays(new Date(), 29), "MMM d")}`,
    );
    expect(lighter.getAttribute("data-level")).toBe("2");
  });

  test("marks the expert quiet when nothing happened this week", async () => {
    server.use(
      getGetExpertActivityMockHandler({
        timezone: "UTC",
        days: activityDays({ 10: { runs: 2 } }),
      }),
    );

    render(<ExpertDetailPage />);

    expect(await screen.findByText("Quiet lately")).toBeDefined();
    expect(
      screen.getByText("0 sessions · 2 runs · last 3 months"),
    ).toBeDefined();
  });

  test("shows a fallback when activity fails to load", async () => {
    server.use(
      http.get("/api/proxy/api/experts/:expertId/activity", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<ExpertDetailPage />);

    expect(await screen.findByText("Activity unavailable")).toBeDefined();
    expect(screen.getByRole("heading", { name: "Maria" })).toBeDefined();
  });

  test("lists the expert's schedules with edit and delete actions", async () => {
    render(<ExpertDetailPage />);

    await openTab("Schedules");
    const row = await screen.findByTestId("schedule-row");
    expect(row.getAttribute("data-schedule-id")).toBe("sched-1");
    expect(screen.getByRole("button", { name: /Edit schedule/ })).toBeDefined();
    expect(screen.getByTestId("schedule-delete-button")).toBeDefined();
  });

  test("deletes a schedule after confirmation", async () => {
    const deleteSpy = vi.fn(() => ({}));
    server.use(getDeleteV1DeleteExecutionScheduleMockHandler(deleteSpy));

    render(<ExpertDetailPage />);

    await openTab("Schedules");
    fireEvent.click(await screen.findByTestId("schedule-delete-button"));
    fireEvent.click(await screen.findByTestId("schedule-confirm-delete"));

    await waitFor(() => expect(deleteSpy).toHaveBeenCalled());
  });

  test("shows an empty schedules message when the expert has none", async () => {
    server.use(getGetV1ListExecutionSchedulesForAUserMockHandler([]));

    render(<ExpertDetailPage />);

    await openTab("Schedules");
    expect(await screen.findByText(/No schedules yet/)).toBeDefined();
  });

  test("shows the expert's recent work with honest status chips", async () => {
    server.use(getListExpertRunsMockHandler(mariaRuns));

    render(<ExpertDetailPage />);

    await openTab("Work");
    const workList = await screen.findByRole("list", { name: "Expert work" });
    expect(within(workList).getByText("Weekly Report")).toBeDefined();
    expect(within(workList).getByText("Completed")).toBeDefined();
    // A run paused for review reads "Waiting for review" — never "Completed"
    // with a contradictory badge next to it.
    expect(within(workList).getByText("Waiting for review")).toBeDefined();
    expect(within(workList).queryByText("Needs review")).toBeNull();
  });

  test("filters work to runs that need review", async () => {
    server.use(getListExpertRunsMockHandler(mariaRuns));

    render(<ExpertDetailPage />);

    await openTab("Work");
    await screen.findByRole("list", { name: "Expert work" });

    fireEvent.click(screen.getByRole("button", { name: /Needs review \(1\)/ }));

    const workList = screen.getByRole("list", { name: "Expert work" });
    expect(within(workList).queryByText("Weekly Report")).toBeNull();
    expect(within(workList).getByText("SEO Audit")).toBeDefined();
  });

  test("shows an empty work message when there is no completed work", async () => {
    render(<ExpertDetailPage />);

    await openTab("Work");
    expect(await screen.findByText(/No completed work yet/)).toBeDefined();
  });

  test("shows a retryable error when recent work fails to load", async () => {
    let attempts = 0;
    server.use(
      http.get("/api/proxy/api/experts/:expertId/runs", () => {
        attempts += 1;
        return attempts === 1
          ? HttpResponse.json({ detail: "boom" }, { status: 500 })
          : HttpResponse.json(mariaRuns);
      }),
    );

    render(<ExpertDetailPage />);

    await openTab("Work");
    expect(
      await screen.findByText("We could not load this expert's recent work."),
    ).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: /try again/i }));

    expect(await screen.findByText("Weekly Report")).toBeDefined();
    expect(attempts).toBe(2);
  });

  test("uploads a new photo from the header avatar", async () => {
    const updateSpy = vi.fn((info: { request: Request }) => info.request);
    uploadAvatarSpy.mockResolvedValueOnce("https://cdn.example.com/maria.png");
    server.use(
      getUpdateExpertAvatarMockHandler(async (info) => {
        updateSpy(info);
        return {
          ...maria,
          avatar_url: "https://cdn.example.com/maria.png",
        };
      }),
    );

    render(<ExpertDetailPage />);

    const button = await screen.findByRole("button", {
      name: "Change Maria's photo",
    });
    const fileInput = screen.getByLabelText("Upload Maria photo");
    expect(button.contains(fileInput)).toBe(false);
    const pickerClick = vi.spyOn(fileInput, "click");
    fireEvent.click(button);
    expect(pickerClick).toHaveBeenCalledTimes(1);
    pickerClick.mockRestore();
    const file = new File(["x"], "maria.png", { type: "image/png" });
    fireEvent.change(fileInput, { target: { files: [file] } });

    await waitFor(() => expect(updateSpy).toHaveBeenCalled());
    expect(uploadAvatarSpy).toHaveBeenCalledWith(file);
    const body = await updateSpy.mock.results[0].value.json();
    expect(body).toEqual({ avatar_url: "https://cdn.example.com/maria.png" });
  });

  test("a failed photo upload leaves the expert untouched", async () => {
    uploadAvatarSpy.mockRejectedValueOnce(new Error("Unauthorized"));
    const updateSpy = vi.fn();
    server.use(
      getUpdateExpertAvatarMockHandler(() => {
        updateSpy();
        return maria;
      }),
    );

    render(<ExpertDetailPage />);

    const button = await screen.findByRole("button", {
      name: "Change Maria's photo",
    });
    const fileInput = screen.getByLabelText("Upload Maria photo");
    expect(button.contains(fileInput)).toBe(false);
    fireEvent.change(fileInput, {
      target: { files: [new File(["x"], "maria.png", { type: "image/png" })] },
    });

    await waitFor(() => expect(uploadAvatarSpy).toHaveBeenCalled());
    await waitFor(() => expect(button).not.toHaveProperty("disabled", true));
    expect(updateSpy).not.toHaveBeenCalled();
  });

  test("paused expert offers one-click resume", async () => {
    const resumeSpy = vi.fn(() => ({ ...maria, schedules_paused_at: null }));
    server.use(
      getGetExpertMockHandler({
        ...maria,
        schedules_paused_at: new Date("2026-08-03T12:00:00Z"),
      }),
      getResumeExpertSchedulesMockHandler(resumeSpy),
    );

    render(<ExpertDetailPage />);

    await screen.findByText(/Schedules paused/);
    fireEvent.click(screen.getByRole("button", { name: "Resume schedules" }));
    await waitFor(() => expect(resumeSpy).toHaveBeenCalled());
  });

  test("fires the expert from the settings tab and returns to the team page", async () => {
    const archiveSpy = vi.fn();
    server.use(
      getGetExpertDetachPreviewMockHandler({
        schedule_names: ["Content Calendar"],
        trigger_names: [],
      }),
      getArchiveExpertMockHandler(archiveSpy),
    );

    render(<ExpertDetailPage />);

    await openTab("Settings");
    fireEvent.click(await screen.findByTestId("expert-fire-button"));

    const dialog = await screen.findByRole("dialog", { name: "Fire Maria?" });
    expect(
      await within(dialog).findByText("1 automation will pause."),
    ).toBeDefined();

    const confirm = await screen.findByTestId("fire-expert-confirm");
    await waitFor(() => expect(confirm.hasAttribute("disabled")).toBe(false));
    fireEvent.click(confirm);

    await waitFor(() => expect(archiveSpy).toHaveBeenCalled());
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/team"));
  });

  test("installs one of the user's own agents as a workflow", async () => {
    const user = userEvent.setup();
    let installBody: unknown;
    server.use(
      getListExpertsMockHandler([maria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([ownLibraryAgent]),
      ),
      getInstallExpertWorkflowMockHandler(async (info) => {
        installBody = await info.request.json();
        return {
          id: "wf-private",
          store_listing_version_id: null,
          library_agent_id: "lib-private",
          graph_id: "graph-private",
          name: "My Private Agent",
          description: null,
        };
      }),
    );

    render(<ExpertDetailPage />);

    await screen.findByRole("heading", { name: "Maria" });
    await openTab("Workflows");
    await user.click(
      await screen.findByRole("button", { name: "Install workflow" }),
    );

    const dialog = await screen.findByRole("dialog");
    const row = await within(dialog).findByTestId("install-workflow-option");
    expect(within(row).getByText("My Private Agent")).toBeDefined();
    await user.click(within(row).getByRole("button", { name: "Install" }));

    await waitFor(() =>
      expect(installBody).toEqual({ library_agent_id: "lib-private" }),
    );
  });

  test("hides an already-installed workflow from the Install picker", async () => {
    const user = userEvent.setup();
    const installed = { ...ownLibraryAgent, id: "lib-1" } as LibraryAgent;
    server.use(
      getListExpertsMockHandler([maria]),
      getGetV2ListLibraryAgentsMockHandler200(libraryResponse([installed])),
    );

    render(<ExpertDetailPage />);

    await screen.findByRole("heading", { name: "Maria" });
    await openTab("Workflows");
    await user.click(screen.getByRole("button", { name: "Install workflow" }));

    const dialog = await screen.findByRole("dialog");
    expect(
      await within(dialog).findByText("No workflows in your library."),
    ).toBeDefined();
    expect(within(dialog).queryByTestId("install-workflow-option")).toBeNull();
  });

  test("fetches later installable workflows after an installed first page", async () => {
    const pages: number[] = [];
    server.use(
      getListExpertsMockHandler([maria]),
      http.get("*/api/library/agents", ({ request }) => {
        const page = Number(new URL(request.url).searchParams.get("page") ?? 1);
        pages.push(page);
        const agents =
          page === 1
            ? [{ ...ownLibraryAgent, id: "lib-1", name: "Already installed" }]
            : page === 2
              ? [ownLibraryAgent]
              : [
                  {
                    ...ownLibraryAgent,
                    id: "lib-later",
                    name: "Another workflow",
                  },
                ];
        return HttpResponse.json({
          ...libraryResponse(agents),
          pagination: {
            total_items: 21,
            total_pages: 3,
            current_page: page,
            page_size: 10,
          },
        });
      }),
    );
    render(<ExpertDetailPage />);
    await screen.findByRole("heading", { name: "Maria" });
    await openTab("Workflows");
    await userEvent.click(
      screen.getByRole("button", { name: "Install workflow" }),
    );
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("My Private Agent")).toBeDefined();
    expect(within(dialog).queryByText("Already installed")).toBeNull();
    expect(pages).toEqual([1, 2]);
    await userEvent.click(
      within(dialog).getByRole("button", { name: "Load more workflows" }),
    );
    expect(await within(dialog).findByText("Another workflow")).toBeDefined();
    expect(pages).toEqual([1, 2, 3]);
    expect(
      within(dialog).queryByRole("button", { name: "Load more workflows" }),
    ).toBeNull();
  });

  test("shows the workflow description rather than an unknown creator", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([maria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([{ ...ownLibraryAgent, description: "" }]),
      ),
    );

    render(<ExpertDetailPage />);

    await screen.findByRole("heading", { name: "Maria" });
    await openTab("Workflows");
    await user.click(screen.getByRole("button", { name: "Install workflow" }));

    const dialog = await screen.findByRole("dialog");
    const row = await within(dialog).findByTestId("install-workflow-option");
    expect(within(row).getByText("From your library")).toBeDefined();
    expect(within(row).queryByText("Unknown")).toBeNull();
  });
});
