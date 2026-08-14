import {
  getListExpertsMockHandler,
  getListExpertsMockHandler401,
  getResumeExpertSchedulesMockHandler,
  getUpdateExpertSoulMockHandler,
  getUpdateExpertSoulMockHandler422,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

const toastMock = vi.hoisted(() => vi.fn());
const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
}));

beforeEach(() => {
  server.use(getGetV1ListExecutionSchedulesForAUserMockHandler([]));
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

  test("Create menu exposes accessible hire and build actions", async () => {
    const user = userEvent.setup();
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    const trigger = await screen.findByRole("button", { name: /create/i });
    // The Radix trigger advertises its menu to assistive tech and keyboard users.
    expect(trigger.getAttribute("aria-haspopup")).toBe("menu");

    await user.click(trigger);

    const hire = await screen.findByRole("menuitem", {
      name: "Hire an expert",
    });
    expect(hire.getAttribute("href")).toBe("/marketplace#experts");

    const build = screen.getByRole("menuitem", {
      name: "Build an agent from scratch",
    });
    expect(build.getAttribute("href")).toBe("/build");
  });

  test("renders hired experts with a workflow count instead of chips", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    expect(await screen.findByText("Maria")).toBeDefined();
    expect(screen.getByText("Marketing Strategist")).toBeDefined();
    expect(screen.getByText("2 workflows")).toBeDefined();
    expect(screen.queryByText("Content Calendar")).toBeNull();
    expect(screen.queryByText("SEO Audit")).toBeNull();
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
    const mariaSchedule: GraphExecutionJobInfo = {
      id: "sched-1",
      name: "Content Calendar",
      user_id: "user-1",
      graph_id: "graph-1",
      graph_version: 1,
      cron: "40 7 * * *",
      input_data: {},
      next_run_time: inTwoDays.toISOString(),
      expert_id: "expert-maria",
    };
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
});
