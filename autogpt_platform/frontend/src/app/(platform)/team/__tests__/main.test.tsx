import {
  getInstallExpertWorkflowMockHandler,
  getInstallExpertWorkflowMockHandler422,
  getListExpertsMockHandler,
  getListExpertsMockHandler401,
  getResumeExpertSchedulesMockHandler,
  getUpdateExpertSoulMockHandler,
  getUpdateExpertSoulMockHandler422,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2ListLibraryAgentsMockHandler200,
  getGetV2ListLibraryAgentsMockHandler401,
  getGetV2ListLibraryAgentsResponseMock200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

const toastMock = vi.hoisted(() => vi.fn());
const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
}));

function libraryResponse(
  agents: LibraryAgent[],
  totalItems = agents.length,
  currentPage = 1,
) {
  const base = getGetV2ListLibraryAgentsResponseMock200();
  return {
    ...base,
    agents,
    pagination: {
      ...base.pagination,
      total_items: totalItems,
      current_page: currentPage,
      page_size: 100,
      total_pages: Math.ceil(totalItems / 100),
    },
  };
}

function makeLibraryAgent(over: Partial<LibraryAgent>): LibraryAgent {
  return {
    id: "lib-agent",
    graph_id: "graph-agent",
    graph_version: 1,
    name: "Research Assistant",
    description: "",
    creator_name: "Acme Labs",
    creator_image_url: "",
    image_url: null,
    status: "COMPLETED",
    created_at: new Date(),
    updated_at: new Date(),
    input_schema: {},
    output_schema: {},
    credentials_input_schema: {},
    has_external_trigger: false,
    has_human_in_the_loop: false,
    has_sensitive_action: false,
    new_output: false,
    can_access_graph: true,
    is_latest_version: true,
    is_favorite: false,
    marketplace_listing: null,
    store_listing_version_id: null,
    ...over,
  } as unknown as LibraryAgent;
}

const adoptableAgent = makeLibraryAgent({
  id: "lib-research",
  graph_id: "graph-research",
  name: "Research Assistant",
  store_listing_version_id: "slv-adopt",
});

const localOnlyAgent = makeLibraryAgent({
  id: "lib-local",
  graph_id: "graph-local",
  name: "My Private Agent",
  store_listing_version_id: null,
});

function makeSchedule(
  over: Partial<GraphExecutionJobInfo> = {},
): GraphExecutionJobInfo {
  return {
    id: "sched-1",
    name: "Content Calendar",
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    cron: "40 7 * * *",
    input_data: {},
    next_run_time: "2026-08-15T07:40:00Z",
    expert_id: "expert-maria",
    ...over,
  };
}

beforeEach(() => {
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(libraryResponse([])),
  );
});

afterEach(() => {
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  toastMock.mockReset();
});

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

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

const notFoundMock = vi.hoisted(() => vi.fn());
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
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const hiredMaria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
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
      description: null,
    },
    {
      id: "wf-2",
      store_listing_version_id: "slv-2",
      library_agent_id: "lib-2",
      graph_id: "graph-2",
      name: "SEO Audit",
      description: null,
    },
  ],
};

const scheduledMaria: Expert = {
  ...hiredMaria,
  last_run_at: new Date("2026-08-03T07:40:00Z"),
  last_run_status: "COMPLETED",
  workflows: [
    {
      ...hiredMaria.workflows[0],
      schedule_cron: "40 7 * * *",
      schedule_id: "sched-1",
    },
    hiredMaria.workflows[1],
  ],
};

describe("TeamPage", () => {
  test("renders the Autopilot card first", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    const autopilot = await screen.findByText("Autopilot");
    expect(screen.getByText(/runs the shop/i)).toBeDefined();

    const maria = await screen.findByText("Maria");
    expect(
      autopilot.compareDocumentPosition(maria) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test("renders hired experts with a workflow count instead of chips", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    expect(await screen.findByText("Maria")).toBeDefined();
    expect(screen.getByText("Marketing Strategist")).toBeDefined();
    expect(screen.getByText("2 workflows")).toBeDefined();
    const card = screen.getByRole("link", { name: "View Maria" });
    expect(within(card).queryByText("Content Calendar")).toBeNull();
    expect(within(card).queryByText("SEO Audit")).toBeNull();
  });

  test("links the card content to the expert page", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    await screen.findByText("Maria");
    const link = screen.getByRole("link", { name: "View Maria" });
    expect(link.getAttribute("href")).toBe("/team/expert-maria");
  });

  test("links Chat to the expert's copilot thread", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    await screen.findByText("Maria");
    const chatLinks = screen.getAllByRole("link", { name: "Chat" });
    const hrefs = chatLinks.map((link) => link.getAttribute("href"));
    expect(hrefs).toContain("/copilot");
    expect(hrefs).toContain(`/copilot?expertId=${hiredMaria.id}`);

    expect(
      screen.getByRole("button", { name: "Install workflow" }),
    ).toBeDefined();
  });

  test("shows a schedule count with the next run on the expert card", async () => {
    const inTwoDays = new Date(Date.now() + 2 * 24 * 60 * 60 * 1000);
    const mariaSchedule = makeSchedule({
      next_run_time: inTwoDays.toISOString(),
    });
    server.use(
      getListExpertsMockHandler([scheduledMaria]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([mariaSchedule]),
    );

    render(<TeamPage />);

    await screen.findByText("Maria");
    expect(await screen.findByText(/1 schedule · next in/)).toBeDefined();
  });

  test("marks scheduled workflows without a schedule as needing setup", async () => {
    const needsSetupMaria: Expert = {
      ...hiredMaria,
      workflows: [
        {
          ...hiredMaria.workflows[0],
          schedule_cron: "40 7 * * *",
          schedule_id: null,
        },
      ],
    };
    server.use(getListExpertsMockHandler([needsSetupMaria]));

    render(<TeamPage />);

    await screen.findByText("Maria");
    expect(screen.getByText(/1 needs setup/)).toBeDefined();
  });

  test("shows weekly spend as a progress bar on the expert card", async () => {
    const budgetMaria: Expert = {
      ...hiredMaria,
      weekly_budget: 50,
      weekly_spend: 12,
    };
    server.use(getListExpertsMockHandler([budgetMaria]));

    render(<TeamPage />);

    await screen.findByText("Maria");
    expect(screen.getByText("Credits this week")).toBeDefined();
    expect(screen.getByText("12 / 50")).toBeDefined();
  });

  test("paused expert offers one-click resume", async () => {
    const pausedMaria: Expert = {
      ...hiredMaria,
      schedules_paused_at: new Date("2026-08-03T12:00:00Z"),
    };
    const resumeSpy = vi.fn(() => ({
      ...pausedMaria,
      schedules_paused_at: null,
    }));
    server.use(
      getListExpertsMockHandler([pausedMaria]),
      getResumeExpertSchedulesMockHandler(resumeSpy),
    );

    render(<TeamPage />);

    await screen.findByText("Maria");
    expect(screen.getByText(/Schedules paused/)).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Resume schedules" }));
    await waitFor(() => expect(resumeSpy).toHaveBeenCalled());
  });

  test("opens the current Soul document from the expert card", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    await user.click(await screen.findByRole("button", { name: "Edit Soul" }));

    expect(screen.getByRole("dialog", { name: "Maria's Soul" })).toBeDefined();
    expect(
      (screen.getByRole("textbox", { name: "Name" }) as HTMLInputElement).value,
    ).toBe("Maria");
    expect(
      (
        screen.getByRole("textbox", {
          name: "Identity and personality",
        }) as HTMLTextAreaElement
      ).value,
    ).toBe("You are Maria, a senior marketing strategist.");
    expect(
      (screen.getByRole("textbox", { name: "Voice" }) as HTMLTextAreaElement)
        .value,
    ).toBe("Warm, concise, and direct.");
    expect(
      (
        screen.getByRole("textbox", {
          name: "Boundaries",
        }) as HTMLTextAreaElement
      ).value,
    ).toBe("Never invent customer evidence.");
    expect(
      screen.getByText(
        "The expert discloses that it is AI when acting externally.",
      ),
    ).toBeDefined();
    expect(
      screen.getByText("External actions require approval."),
    ).toBeDefined();
    expect(screen.getAllByRole("textbox")).toHaveLength(4);
    expect(screen.queryByRole("button", { name: /remove/i })).toBeNull();
  });

  test("keeps the Soul drawer mounted for its exit animation", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    await user.click(await screen.findByRole("button", { name: "Edit Soul" }));
    const dialog = screen.getByRole("dialog", { name: "Maria's Soul" });
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => {
      expect(dialog.getAttribute("data-state")).toBe("closed");
    });
    expect(dialog.textContent).toContain("Maria's Soul");
  });

  test("opens only the Soul drawer when activated with the keyboard", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    const editSoul = await screen.findByRole("button", { name: "Edit Soul" });
    editSoul.focus();
    await user.keyboard("{Enter}");

    expect(screen.getByRole("dialog", { name: "Maria's Soul" })).toBeDefined();
    expect(screen.queryByRole("dialog", { name: "Maria" })).toBeNull();
  });

  test("keeps nested card actions independent for keyboard users", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    const installWorkflow = await screen.findByRole("button", {
      name: "Install workflow",
    });
    installWorkflow.focus();
    await user.keyboard("{Enter}");

    expect(
      screen.getByRole("dialog", { name: /Install a workflow/ }),
    ).toBeDefined();
    expect(screen.queryByRole("dialog", { name: "Maria" })).toBeNull();
  });

  test("saves Soul edits and refreshes the experts query", async () => {
    const user = userEvent.setup();
    let listRequests = 0;
    let requestBody: unknown;
    const updatedMaria = { ...hiredMaria, name: "Mara" };
    server.use(
      getListExpertsMockHandler(() => {
        listRequests += 1;
        return listRequests === 1 ? [hiredMaria] : [updatedMaria];
      }),
      getUpdateExpertSoulMockHandler(async ({ request }) => {
        requestBody = await request.json();
        return updatedMaria;
      }),
    );

    render(<TeamPage />);

    await user.click(await screen.findByRole("button", { name: "Edit Soul" }));
    const nameInput = screen.getByRole("textbox", { name: "Name" });
    await user.clear(nameInput);
    await user.type(nameInput, "Mara");
    await user.click(screen.getByRole("button", { name: "Save Soul" }));

    await waitFor(() => expect(listRequests).toBeGreaterThan(1));
    expect(requestBody).toEqual({
      name: "Mara",
      identity: hiredMaria.identity,
      voice_preferences: hiredMaria.voice_preferences,
      boundaries: hiredMaria.boundaries,
    });
    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Soul saved", variant: "success" }),
    );
  });

  test("preserves Soul edits and shows feedback when saving fails", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getUpdateExpertSoulMockHandler422(),
    );

    render(<TeamPage />);

    await user.click(await screen.findByRole("button", { name: "Edit Soul" }));
    const voiceInput = screen.getByRole("textbox", { name: "Voice" });
    await user.clear(voiceInput);
    await user.type(voiceInput, "Calm and conversational.");
    await user.click(screen.getByRole("button", { name: "Save Soul" }));

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't save Soul",
          variant: "destructive",
        }),
      ),
    );
    expect((voiceInput as HTMLTextAreaElement).value).toBe(
      "Calm and conversational.",
    );
    expect(screen.getByRole("dialog", { name: "Maria's Soul" })).toBeDefined();
  });

  test("shows empty state linking to the marketplace when no experts are hired", async () => {
    server.use(getListExpertsMockHandler([]));

    render(<TeamPage />);

    expect(await screen.findByText("Autopilot")).toBeDefined();
    const link = await screen.findByRole("link", {
      name: "Browse the marketplace",
    });
    expect(link.getAttribute("href")).toBe("/marketplace");
  });

  test("shows an error card when loading experts fails", async () => {
    server.use(getListExpertsMockHandler401());

    render(<TeamPage />);

    expect(await screen.findByText("Something went wrong")).toBeDefined();
  });

  test("calls notFound() when the flag is resolved and disabled", () => {
    let listRequests = 0;
    server.use(
      getListExpertsMockHandler(() => {
        listRequests += 1;
        return [hiredMaria];
      }),
    );
    setFlagStatusMock.mockReturnValueOnce({ enabled: false, ready: true });
    notFoundMock.mockClear();

    try {
      render(<TeamPage />);
    } catch {
      // React surfaces the thrown notFound() error; the assertion below is
      // what we actually care about.
    }

    expect(notFoundMock).toHaveBeenCalled();
    expect(listRequests).toBe(0);
    expect(screen.queryByRole("button", { name: "Edit Soul" })).toBeNull();
  });

  test("groups installed workflows under each expert in the What runs zone", async () => {
    const john: Expert = {
      ...hiredMaria,
      id: "expert-john",
      name: "John",
      workflows: [
        {
          id: "wf-j",
          store_listing_version_id: "slv-j",
          library_agent_id: "lib-j",
          graph_id: "graph-j",
          name: "Lead Scraper",
          description: null,
        },
      ],
    };
    server.use(getListExpertsMockHandler([scheduledMaria, john]));

    render(<TeamPage />);

    const zone = await screen.findByRole("region", { name: "What runs" });
    const mariaGroup = within(zone).getByRole("region", {
      name: "Maria runs",
    });
    expect(within(mariaGroup).getByText("Content Calendar")).toBeDefined();
    expect(within(mariaGroup).getByText("SEO Audit")).toBeDefined();
    expect(within(mariaGroup).queryByText("Lead Scraper")).toBeNull();

    const johnGroup = within(zone).getByRole("region", { name: "John runs" });
    expect(within(johnGroup).getByText("Lead Scraper")).toBeDefined();
    expect(within(johnGroup).queryByText("Content Calendar")).toBeNull();
  });

  test("shows a quiet empty message for an expert with nothing installed", async () => {
    const emptyMaria: Expert = { ...hiredMaria, workflows: [] };
    server.use(getListExpertsMockHandler([emptyMaria]));

    render(<TeamPage />);

    const group = await screen.findByRole("region", { name: "Maria runs" });
    expect(within(group).getByText(/nothing installed yet/i)).toBeDefined();
  });

  test("lists adoptable and local-only agents under Your agents", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([adoptableAgent, localOnlyAgent]),
      ),
    );

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    expect(within(agents).getByText("Research Assistant")).toBeDefined();
    expect(within(agents).getByText("My Private Agent")).toBeDefined();
    expect(within(agents).getByRole("button", { name: "Adopt" })).toBeDefined();
    const localOnly = within(agents).getByText("Local only");
    expect(localOnly).toBeDefined();
    await user.hover(localOnly);
    expect((await screen.findByRole("tooltip")).textContent).toBe(
      "Publish this agent to the Marketplace before adopting it.",
    );
  });

  test("hides already-installed agents from Your agents", async () => {
    const installedAgent = makeLibraryAgent({
      id: "lib-installed",
      graph_id: "graph-1",
      name: "Content Calendar",
      store_listing_version_id: "slv-1",
    });
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([installedAgent, adoptableAgent]),
      ),
    );

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    expect(within(agents).getByText("Research Assistant")).toBeDefined();
    expect(within(agents).queryByText("My Private Agent")).toBeNull();
    expect(
      within(agents).getAllByRole("button", { name: "Adopt" }),
    ).toHaveLength(1);
  });

  test("adopts an agent onto the selected expert via the install endpoint", async () => {
    const user = userEvent.setup();
    let installExpertId: string | undefined;
    let installBody: unknown;
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([adoptableAgent]),
      ),
      getInstallExpertWorkflowMockHandler(async (info) => {
        installExpertId = info.params.expertId as string;
        installBody = await info.request.json();
        return {
          id: "wf-adopted",
          store_listing_version_id: "slv-adopt",
          library_agent_id: "lib-research",
          graph_id: "graph-research",
          name: "Research Assistant",
          description: null,
        };
      }),
    );

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    await user.click(within(agents).getByRole("button", { name: "Adopt" }));

    await waitFor(() => expect(installExpertId).toBe("expert-maria"));
    expect(installBody).toEqual({ store_listing_version_id: "slv-adopt" });
  });

  test("asks which expert to adopt into when more than one is hired", async () => {
    const user = userEvent.setup();
    const john: Expert = {
      ...hiredMaria,
      id: "expert-john",
      name: "John",
      workflows: [],
    };
    let installExpertId: string | undefined;
    server.use(
      getListExpertsMockHandler([hiredMaria, john]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([adoptableAgent]),
      ),
      getInstallExpertWorkflowMockHandler((info) => {
        installExpertId = info.params.expertId as string;
        return {
          id: "wf-adopted",
          store_listing_version_id: "slv-adopt",
          library_agent_id: "lib-research",
          graph_id: "graph-research",
          name: "Research Assistant",
          description: null,
        };
      }),
    );

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    await user.click(within(agents).getByRole("button", { name: "Adopt" }));
    expect(screen.queryByText(/undo anytime/i)).toBeNull();
    await user.click(
      await screen.findByRole("button", {
        name: /Adds this agent to John's workflows/i,
      }),
    );

    await waitFor(() => expect(installExpertId).toBe("expert-john"));
  });

  test("filter chips switch between members and agents in the zone", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([adoptableAgent]),
      ),
    );

    render(<TeamPage />);

    await screen.findByRole("region", { name: "Maria runs" });
    await screen.findByRole("region", { name: "Your agents" });

    await user.click(screen.getByRole("button", { name: "Agents" }));
    expect(screen.queryByRole("region", { name: "Maria runs" })).toBeNull();
    expect(screen.getByRole("region", { name: "Your agents" })).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Members" }));
    expect(screen.getByRole("region", { name: "Maria runs" })).toBeDefined();
    expect(screen.queryByRole("region", { name: "Your agents" })).toBeNull();
  });

  test("keeps only workflows with actual jobs when Scheduled is active", async () => {
    const user = userEvent.setup();
    const manuallyScheduledMaria: Expert = {
      ...hiredMaria,
      workflows: [
        { ...hiredMaria.workflows[0], schedule_id: null, schedule_cron: null },
        hiredMaria.workflows[1],
      ],
    };
    server.use(
      getListExpertsMockHandler([manuallyScheduledMaria]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([makeSchedule()]),
    );

    render(<TeamPage />);

    const group = await screen.findByRole("region", { name: "Maria runs" });
    expect(within(group).getByText("Content Calendar")).toBeDefined();
    expect(within(group).getByText("SEO Audit")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Scheduled" }));

    const scheduledGroup = screen.getByRole("region", { name: "Maria runs" });
    expect(within(scheduledGroup).getByText("Content Calendar")).toBeDefined();
    expect(within(scheduledGroup).queryByText("SEO Audit")).toBeNull();
  });

  test("does not treat a deleted scheduler job as scheduled", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([scheduledMaria]));

    render(<TeamPage />);

    await screen.findByRole("region", { name: "Maria runs" });
    await user.click(screen.getByRole("button", { name: "Scheduled" }));

    expect(screen.queryByRole("region", { name: "Maria runs" })).toBeNull();
    expect(screen.getByText("No scheduled workflows yet.")).toBeDefined();
  });

  test("shows every actual schedule attached to a workflow", async () => {
    server.use(
      getListExpertsMockHandler([scheduledMaria]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeSchedule(),
        makeSchedule({ id: "sched-2", cron: "0 9 * * 1" }),
      ]),
    );

    render(<TeamPage />);

    const group = await screen.findByRole("region", { name: "Maria runs" });
    expect(within(group).getByText("Every day at 07:40")).toBeDefined();
    expect(within(group).getByText("Every Monday at 09:00")).toBeDefined();
  });

  test("shows paused instead of recurrence for a paused workflow", async () => {
    const pausedMaria: Expert = {
      ...scheduledMaria,
      schedules_paused_at: new Date("2026-08-14T12:00:00Z"),
    };
    server.use(
      getListExpertsMockHandler([pausedMaria]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([makeSchedule()]),
    );

    render(<TeamPage />);

    const group = await screen.findByRole("region", { name: "Maria runs" });
    const workflow = within(group).getAllByTestId("what-runs-workflow-row")[0];
    expect(within(workflow).getByText("Paused")).toBeDefined();
    expect(within(workflow).queryByText("Every day at 07:40")).toBeNull();
  });

  test("links a workflow with a missing job to its setup page", async () => {
    const staleMaria: Expert = {
      ...hiredMaria,
      workflows: [
        {
          ...hiredMaria.workflows[0],
          schedule_cron: "40 7 * * *",
          schedule_id: "deleted-schedule",
        },
      ],
    };
    server.use(getListExpertsMockHandler([staleMaria]));

    render(<TeamPage />);

    const group = await screen.findByRole("region", { name: "Maria runs" });
    expect(within(group).getByText("Needs setup")).toBeDefined();
    expect(
      within(group).getByRole("link", { name: "Set up" }).getAttribute("href"),
    ).toBe("/library/agents/lib-1");
  });

  test("shows feedback and re-enables Adopt when adoption fails", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([adoptableAgent]),
      ),
      getInstallExpertWorkflowMockHandler422(),
    );

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    await user.click(within(agents).getByRole("button", { name: "Adopt" }));

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't adopt Research Assistant",
          variant: "destructive",
        }),
      ),
    );
    const adoptButton = within(agents).getByRole("button", { name: "Adopt" });
    expect(adoptButton.hasAttribute("disabled")).toBe(false);
  });

  test("tracks pending adoption by library agent id", async () => {
    const user = userEvent.setup();
    let installCalls = 0;
    const sameGraphAgent = makeLibraryAgent({
      ...adoptableAgent,
      id: "lib-research-v2",
      name: "Research Assistant v2",
    });
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([adoptableAgent, sameGraphAgent]),
      ),
      getInstallExpertWorkflowMockHandler(async () => {
        installCalls += 1;
        await new Promise((resolve) => setTimeout(resolve, 150));
        return {
          id: "wf-adopted",
          store_listing_version_id: "slv-adopt",
          library_agent_id: "lib-research",
          graph_id: "graph-research",
          name: "Research Assistant",
          description: null,
        };
      }),
    );

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    const rows = within(agents).getAllByTestId("what-runs-agent-row");
    const firstAdopt = within(rows[0]).getByRole("button", { name: "Adopt" });
    const secondAdopt = within(rows[1]).getByRole("button", { name: "Adopt" });
    await user.click(firstAdopt);
    await waitFor(() => expect(firstAdopt.hasAttribute("disabled")).toBe(true));
    expect(secondAdopt.hasAttribute("disabled")).toBe(false);
    await user.click(firstAdopt);

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "success" }),
      ),
    );
    expect(installCalls).toBe(1);
  });

  test("shows a retry state when loading your agents fails", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler401(),
    );

    render(<TeamPage />);

    expect(
      await screen.findByText(/could not load your agents/i),
    ).toBeDefined();

    server.use(
      getGetV2ListLibraryAgentsMockHandler200(
        libraryResponse([adoptableAgent]),
      ),
    );
    await user.click(screen.getByRole("button", { name: "Retry" }));

    const agents = await screen.findByRole("region", { name: "Your agents" });
    expect(within(agents).getByText("Research Assistant")).toBeDefined();
  });

  test("shows filter-specific empty states for Workflows and Scheduled", async () => {
    const user = userEvent.setup();
    const emptyMaria: Expert = { ...hiredMaria, workflows: [] };
    server.use(getListExpertsMockHandler([emptyMaria]));

    render(<TeamPage />);

    await screen.findByRole("region", { name: "What runs" });

    await user.click(screen.getByRole("button", { name: "Workflows" }));
    expect(screen.getByText("No workflows installed yet.")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Scheduled" }));
    expect(screen.getByText("No scheduled workflows yet.")).toBeDefined();
  });

  test("tells an empty library apart from a fully adopted one", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    expect(
      await within(agents).findByText("No agents in your library yet."),
    ).toBeDefined();
  });

  test("loads every non-hidden library page before showing agents", async () => {
    const installedAgent = makeLibraryAgent({
      id: "lib-installed",
      graph_id: "graph-1",
      name: "Content Calendar",
      store_listing_version_id: "slv-1",
    });
    const requestedPages: string[] = [];
    const hiddenFilters: string[] = [];
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getGetV2ListLibraryAgentsMockHandler200(({ request }) => {
        const params = new URL(request.url).searchParams;
        const page = params.get("page") ?? "1";
        requestedPages.push(page);
        hiddenFilters.push(params.get("is_hidden") ?? "missing");
        return page === "2"
          ? libraryResponse([adoptableAgent], 101, 2)
          : libraryResponse([installedAgent], 101, 1);
      }),
    );

    render(<TeamPage />);

    const agents = await screen.findByRole("region", { name: "Your agents" });
    expect(within(agents).getByText("Research Assistant")).toBeDefined();
    expect(requestedPages).toEqual(["1", "2"]);
    expect(hiddenFilters).toEqual(["false", "false"]);
    expect(within(agents).queryByText(/aren't shown/)).toBeNull();
  });

  test("shows last-run status as a chip beside the workflow count", async () => {
    server.use(getListExpertsMockHandler([scheduledMaria]));

    render(<TeamPage />);

    const group = await screen.findByRole("region", { name: "Maria runs" });
    expect(within(group).getByText("Maria · 2 workflows")).toBeDefined();
    expect(within(group).getByText(/Last run succeeded/)).toBeDefined();
  });
});
