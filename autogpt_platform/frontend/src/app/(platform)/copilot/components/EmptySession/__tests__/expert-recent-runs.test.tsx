import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetExpertMockHandler,
  getListExpertCredentialsMockHandler200,
  getListExpertIdentitiesMockHandler,
  getListExpertsMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { InsetSideSlot } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../EmptySession";
import { WorkArea } from "../../WorkArea/WorkArea";
import { useWorkAreaStore } from "../../../workArea";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "me@example.com", user_metadata: {} },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

const maria = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  job_title: "Head of Growth",
  tagline: "Turns launches into leads.",
  bio: null,
  skills: [],
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  autopilot_mode: "ask_first",
  weekly_budget: null,
  last_run_at: null,
} satisfies Expert;

const WORKFLOW_NAMES: Record<string, string> = {
  "graph-news": "Weekly newsletter",
  "graph-leads": "Lead finder",
  "graph-team": "Teammate flow",
};

function makeRun(
  id: string,
  graphID: string,
  status: AgentExecutionStatus,
  extra: Partial<GraphExecutionMeta> = {},
): GraphExecutionMeta {
  return {
    id,
    user_id: "user-1",
    graph_id: graphID,
    graph_version: 3,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status,
    started_at: new Date("2026-11-10T09:00:00Z"),
    ended_at: null,
    stats: null,
    ...extra,
  };
}

function serveRuns(...responses: GraphExecutionMeta[][]) {
  let call = 0;
  server.use(
    getGetV1ListAllExecutionsMockHandler(() => {
      const runs = responses[Math.min(call, responses.length - 1)];
      call += 1;
      return runs;
    }),
  );
}

function renderHome(searchParams: string) {
  const Wrapper = withNuqsTestingAdapter({ searchParams, hasMemory: true });
  return render(
    <Wrapper>
      <div className="flex">
        <EmptySession
          isCreatingSession={false}
          onCreateSession={() => {}}
          onSend={vi.fn()}
        />
        <WorkArea sessionId={null} />
        <InsetSideSlot />
      </div>
    </Wrapper>,
  );
}

async function openRecentRuns(name: string) {
  const profileTab = await screen
    .findByRole("tab", { name: "Profile" })
    .catch(() => null);
  if (profileTab) await userEvent.click(profileTab);
  else {
    await userEvent.click(
      await screen.findByRole("button", { name: "Open the side panel" }),
    );
  }
  const panel = await screen.findByRole("group", { name: `About ${name}` });
  return within(panel).findByRole("region", { name: "Recent runs" });
}

beforeEach(() => {
  useWorkAreaStore.getState().reset();
  server.use(
    http.get("/api/proxy/api/tasks", () => HttpResponse.json({ tasks: [] })),
    http.get("/api/proxy/api/library/agents", () =>
      HttpResponse.json({
        agents: Object.entries(WORKFLOW_NAMES).map(([graphID, name]) => ({
          id: `lib-${graphID}`,
          graph_id: graphID,
          name,
        })),
        pagination: {
          total_items: 3,
          total_pages: 1,
          current_page: 1,
          page_size: 100,
        },
      }),
    ),
    getListExpertIdentitiesMockHandler([maria]),
    getListExpertsMockHandler200([maria]),
    getGetExpertMockHandler(maria),
    getListExpertCredentialsMockHandler200([]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListCopilotFollowupSchedulesMockHandler([]),
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
  );
});

describe("Home — recent runs in the Profile panel", () => {
  it("lists Otto's five newest runs of the user's own and opens one beside the chat", async () => {
    serveRuns([
      makeRun("run-teammate", "graph-team", "COMPLETED", {
        user_id: "user-2",
      }),
      makeRun("run-1", "graph-news", "COMPLETED", {
        trigger_source: "copilot",
      }),
      makeRun("run-2", "graph-leads", "FAILED", { schedule_id: "sched-1" }),
      makeRun("run-3", "graph-news", "TERMINATED"),
      makeRun("run-4", "graph-leads", "COMPLETED"),
      makeRun("run-5", "graph-news", "COMPLETED"),
      makeRun("run-6", "graph-leads", "COMPLETED"),
    ]);
    renderHome("");
    const recent = await openRecentRuns("Otto");

    await within(recent).findAllByRole("button", { name: /Weekly newsletter/ });
    const rows = within(recent).getAllByRole("button");
    expect(rows).toHaveLength(5);
    expect(within(recent).queryByText("Teammate flow")).toBeNull();
    const [first, failed] = rows;
    expect(first.textContent).toContain("Weekly newsletter");
    expect(first.textContent).toContain("Started by Otto");
    expect(first.textContent).toContain("Done");

    expect(failed.textContent).toContain("Lead finder");
    expect(failed.textContent).toContain("Scheduled run");
    expect(failed.textContent).toContain("Failed");

    await userEvent.click(failed);
    const state = useWorkAreaStore.getState();
    expect(state.activeKey).toBe("workflow:graph-leads");
    expect(state.tabs).toContainEqual(
      expect.objectContaining({
        kind: "workflow",
        graphID: "graph-leads",
        graphVersion: 3,
        executionID: "run-2",
        title: "Lead finder",
      }),
    );
  });

  it("shows only the runs an expert started", async () => {
    serveRuns([
      makeRun("run-otto", "graph-news", "COMPLETED"),
      makeRun("run-maria", "graph-leads", "COMPLETED", {
        expert_id: maria.id,
      }),
    ]);
    renderHome(`?expertId=${maria.id}`);
    const recent = await openRecentRuns("Maria");

    expect(
      await within(recent).findByRole("button", { name: /Lead finder/ }),
    ).toBeDefined();
    expect(within(recent).getAllByRole("button")).toHaveLength(1);
    expect(within(recent).queryByText("Weekly newsletter")).toBeNull();
  });

  it("says there are no runs yet when the user has none", async () => {
    serveRuns([
      makeRun("run-teammate", "graph-team", "COMPLETED", {
        user_id: "user-2",
      }),
    ]);
    renderHome("");
    const recent = await openRecentRuns("Otto");

    expect(await within(recent).findByText("No runs yet.")).toBeDefined();
    expect(within(recent).queryByRole("button")).toBeNull();
  });

  it("refreshes a run while it is going", async () => {
    serveRuns(
      [makeRun("run-live", "graph-news", "RUNNING")],
      [makeRun("run-live", "graph-news", "COMPLETED")],
    );
    renderHome("");
    const recent = await openRecentRuns("Otto");

    const row = await within(recent).findByRole("button", {
      name: /Weekly newsletter/,
    });
    expect(row.textContent).toContain("Running");
    await waitFor(
      () =>
        expect(
          within(recent).getByRole("button", { name: /Weekly newsletter/ })
            .textContent,
        ).toContain("Done"),
      { timeout: 5000 },
    );
  });
});
