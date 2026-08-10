import {
  getGetExpertMockHandler,
  getResumeExpertSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getDeleteV1DeleteExecutionScheduleMockHandler,
  getGetV1ListExecutionSchedulesForAUserMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

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

beforeEach(() => {
  server.use(
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([mariaSchedule]),
  );
});

afterEach(() => {
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
});

describe("ExpertDetailPage", () => {
  test("renders the expert profile with workflows and schedule state", async () => {
    render(<ExpertDetailPage />);

    expect(await screen.findByRole("heading", { name: "Maria" })).toBeDefined();
    expect(screen.getByText("Marketing Strategist")).toBeDefined();
    expect(
      screen.getByText("Maria is a senior marketing strategist."),
    ).toBeDefined();
    expect(screen.getByText("Content strategy")).toBeDefined();

    const workflowRows = screen.getAllByTestId("expert-workflow-row");
    expect(workflowRows).toHaveLength(2);
    expect(within(workflowRows[0]).getByText("Content Calendar")).toBeDefined();
    expect(
      within(workflowRows[0]).getByText(/Every day at 07:40/),
    ).toBeDefined();
    expect(within(workflowRows[1]).getByText("SEO Audit")).toBeDefined();
    expect(within(workflowRows[1]).getByText("Needs setup")).toBeDefined();
  });

  test("lists the expert's schedules with edit and delete actions", async () => {
    render(<ExpertDetailPage />);

    await screen.findByRole("heading", { name: "Maria" });
    const row = await screen.findByTestId("schedule-row");
    expect(row.getAttribute("data-schedule-id")).toBe("sched-1");
    expect(screen.getByRole("button", { name: /Edit schedule/ })).toBeDefined();
    expect(screen.getByTestId("schedule-delete-button")).toBeDefined();
  });

  test("deletes a schedule after confirmation", async () => {
    const deleteSpy = vi.fn(() => ({}));
    server.use(getDeleteV1DeleteExecutionScheduleMockHandler(deleteSpy));

    render(<ExpertDetailPage />);

    fireEvent.click(await screen.findByTestId("schedule-delete-button"));
    fireEvent.click(await screen.findByTestId("schedule-confirm-delete"));

    await waitFor(() => expect(deleteSpy).toHaveBeenCalled());
  });

  test("shows an empty schedules message when the expert has none", async () => {
    server.use(getGetV1ListExecutionSchedulesForAUserMockHandler([]));

    render(<ExpertDetailPage />);

    await screen.findByRole("heading", { name: "Maria" });
    expect(screen.getByText(/No schedules yet/)).toBeDefined();
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
});
