import { expect, test, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetBriefingsGetLatestBriefingMockHandler200 } from "@/app/api/__generated__/endpoints/briefings/briefings.msw";
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
// "listening" sitrep item.
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
