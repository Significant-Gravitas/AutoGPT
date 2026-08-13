import { expect, test, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetBriefingsGetLatestBriefingMockHandler200 } from "@/app/api/__generated__/endpoints/briefings/briefings.msw";
import { EmptySession } from "../../EmptySession/EmptySession";

// The mirror of main.test.tsx: everything the briefing recap adds rides the
// experts flag, so with it off the pre-existing pulse strip must be what
// renders and none of the new surfaces may mount.
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.AGENT_BRIEFING,
  };
});

const baseProps = {
  inputLayoutId: "test-layout",
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
};

function mockHomeData() {
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
    // Would render something if the briefing recap mounted anyway.
    getGetBriefingsGetLatestBriefingMockHandler200({
      id: "briefing-1",
      briefing_date: new Date(),
      created_at: new Date(),
      delivered_at: new Date(),
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
            link: "/library/agents/lib-1?activeTab=runs&activeItem=exec-1",
          },
        ],
        decision_items: [],
      },
    }),
  );
}

test("keeps the pulse strip and mounts no briefing home when experts is off", async () => {
  mockHomeData();

  // Absence of the rendered text alone could pass just because the briefing
  // hadn't resolved yet. The briefing query only exists inside CopilotHome,
  // so "never requested" is the durable proof the gate held — and it fails
  // loudly if the gate is removed.
  const requestedPaths: string[] = [];
  function record({ request }: { request: Request }) {
    requestedPaths.push(new URL(request.url).pathname);
  }
  server.events.on("request:start", record);

  try {
    render(<EmptySession {...baseProps} />);

    // The pulse strip only renders once its own library + executions
    // requests have resolved, so by here a mounted CopilotHome would have
    // fired its queries too.
    expect(
      await screen.findByText("What's happening with your agents"),
    ).toBeDefined();

    expect(requestedPaths.some((path) => path.includes("briefing"))).toBe(
      false,
    );
    expect(screen.queryByText("What ran")).toBeNull();
    expect(screen.queryByText("What was found")).toBeNull();
  } finally {
    server.events.removeListener("request:start", record);
  }
});
