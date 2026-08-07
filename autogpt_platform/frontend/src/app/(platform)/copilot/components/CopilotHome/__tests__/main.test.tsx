import { expect, test, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetBriefingsGetLatestBriefingMockHandler200 } from "@/app/api/__generated__/endpoints/briefings/briefings.msw";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getGetV2GetPendingReviewsMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { CopilotHome } from "../CopilotHome";

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.BRIEFING_HOME || flag === actual.Flag.HIRE_EXPERTS,
  };
});

const baseProps = {
  inputLayoutId: "test-layout",
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
};

// Guarantees the pulse strip has at least one chip regardless of faker's
// random defaults — an agent with an external trigger always produces a
// "listening" sitrep item. Also pins pending reviews to empty since the
// generated mock otherwise returns 1-10 random reviews by default, which
// would make the needs-attention slot non-deterministic across tests.
function mockPulseStripAgent() {
  const base = getGetV2ListLibraryAgentsResponseMock();
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...base,
      agents: [
        { ...base.agents[0], graph_id: "g-1", has_external_trigger: true },
      ],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 100,
      },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
    getGetV2GetPendingReviewsMockHandler200([]),
  );
}

const pendingReview: PendingHumanReviewModel = {
  node_exec_id: "ne-1",
  node_id: "n-1",
  user_id: "u-1",
  graph_exec_id: "run-1",
  graph_id: "g-1",
  graph_version: 1,
  payload: { to: "x@y.com" },
  instructions: "Approve outreach email",
  editable: true,
  status: "WAITING",
  expert_id: "exp-1",
  expert_name: "Ana",
  expert_avatar_url: null,
  agent_name: "Lead Finder",
  library_agent_id: "lib-1",
  session_id: null,
  created_at: new Date(),
};

const healthyExpert: Expert = {
  id: "expert-healthy",
  name: "Sales Scout",
  avatar_url: null,
  role: "Sales Researcher",
  bio: null,
  skills: [],
  tagline: "Finds leads while you sleep",
  identity: "You are a senior sales researcher.",
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  last_run_at: new Date("2026-08-06T10:00:00Z"),
  last_run_status: "COMPLETED",
};

const pausedExpert: Expert = {
  id: "expert-paused",
  name: "Support Bot",
  avatar_url: null,
  role: "Support Triager",
  bio: null,
  skills: [],
  tagline: "Triages tickets",
  identity: "You are a support triager.",
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  schedules_paused_at: new Date("2026-08-05T10:00:00Z"),
};

const needsSetupExpert: Expert = {
  id: "expert-needs-setup",
  name: "Ops Analyst",
  avatar_url: null,
  role: "Operations Analyst",
  bio: null,
  skills: [],
  tagline: "Keeps the ops dashboard current",
  identity: "You are an operations analyst.",
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [
    {
      id: "wf-1",
      store_listing_version_id: "slv-1",
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "Daily Report",
      description: null,
      schedule_cron: "0 9 * * *",
      schedule_id: null,
    },
  ],
};

function mockTeamStrip() {
  server.use(
    getListExpertsMockHandler200([healthyExpert, pausedExpert]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
  );
}

test("renders greeting and composer", async () => {
  mockPulseStripAgent();
  render(<CopilotHome {...baseProps} />);
  expect(await screen.findByPlaceholderText(/./)).toBeDefined();
});

test("falls back to pulse strip when there is no briefing", async () => {
  mockPulseStripAgent();
  server.use(getGetBriefingsGetLatestBriefingMockHandler200(null));
  render(<CopilotHome {...baseProps} />);
  expect(
    await screen.findByText("What's happening with your agents"),
  ).toBeDefined();
});

test("renders briefing sections when a briefing is available", async () => {
  mockPulseStripAgent();
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200({
      id: "briefing-1",
      briefing_date: new Date(),
      created_at: new Date(),
      content: {
        generated_at: new Date(),
        timezone: "UTC",
        zero_expert_fallback: false,
        run_items: [
          {
            expert_id: "expert-1",
            expert_name: "Sales Scout",
            expert_avatar_url: null,
            agent_name: "Lead Finder",
            graph_id: "graph-1",
            execution_id: "exec-1",
            library_agent_id: "lib-1",
            status: "COMPLETED",
            summary: "Found 3 leads",
            link: "/library/agents/lib-1/runs/exec-1",
          },
          {
            expert_id: "expert-2",
            expert_name: "Support Bot",
            expert_avatar_url: null,
            agent_name: "Ticket Triager",
            graph_id: "graph-2",
            execution_id: "exec-2",
            library_agent_id: "lib-2",
            status: "FAILED",
            summary: null,
            link: "/library/agents/lib-2/runs/exec-2",
          },
        ],
        decision_items: [
          {
            node_exec_id: "node-1",
            graph_exec_id: "graph-exec-1",
            title: "Approve outreach email",
            expert_id: "expert-1",
            expert_name: "Sales Scout",
            expert_avatar_url: null,
            link: "/library/agents/lib-1/runs/exec-1?node=node-1",
          },
        ],
      },
    }),
  );
  render(<CopilotHome {...baseProps} />);

  expect(await screen.findByText("What ran")).toBeDefined();
  expect(screen.getByText("What was found")).toBeDefined();
  expect(screen.getByText("Needs your decision (1)")).toBeDefined();
  expect(screen.getByText("Found 3 leads")).toBeDefined();

  const decisionLink = screen.getByRole("link", {
    name: /Approve outreach email/,
  });
  expect(decisionLink.getAttribute("href")).toBe(
    "/library/agents/lib-1/runs/exec-1?node=node-1",
  );
});

test("renders the team strip with hired experts", async () => {
  mockPulseStripAgent();
  server.use(getGetBriefingsGetLatestBriefingMockHandler200(null));
  mockTeamStrip();
  render(<CopilotHome {...baseProps} />);

  expect(await screen.findByText("Sales Scout")).toBeDefined();
  expect(screen.getByText("Support Bot")).toBeDefined();
  expect(screen.getByText("Paused")).toBeDefined();

  const chatLinks = screen
    .getAllByRole("link", { name: "Chat" })
    .map((link) => link.getAttribute("href"));
  expect(chatLinks).toContain("/copilot?expertId=expert-healthy");
  expect(chatLinks).toContain("/copilot?expertId=expert-paused");
});

test("uses singular grammar for a single workflow needing setup", async () => {
  mockPulseStripAgent();
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200(null),
    getListExpertsMockHandler200([needsSetupExpert]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
  );
  render(<CopilotHome {...baseProps} />);

  expect(await screen.findByText("1 workflow needs setup")).toBeDefined();
});

test("renders a needs-attention row when a review is pending", async () => {
  mockPulseStripAgent();
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200(null),
    getGetV2GetPendingReviewsMockHandler200([pendingReview]),
  );
  render(<CopilotHome {...baseProps} />);

  expect(await screen.findByText("Approve outreach email")).toBeDefined();
});

test("does not render the needs-attention slot when there are no pending reviews", async () => {
  mockPulseStripAgent();
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200(null),
    getGetV2GetPendingReviewsMockHandler200([]),
  );
  render(<CopilotHome {...baseProps} />);

  await screen.findByText("What's happening with your agents");
  expect(screen.queryByText(/Needs your attention/)).toBeNull();
});
