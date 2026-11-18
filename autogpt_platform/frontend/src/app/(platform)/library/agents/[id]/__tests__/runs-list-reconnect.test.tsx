import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1ListGraphExecutionsMockHandler,
  getGetV1ListGraphExecutionsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import { installLiveSocket } from "@/tests/integrations/live-socket";
import { render, screen } from "@/tests/integrations/test-utils";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, expect, test, vi } from "vitest";
import AgentLibraryPage from "../page";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ id: "library-agent" }),
  usePathname: () => "/library/agents/library-agent",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

afterEach(() => {
  vi.unstubAllGlobals();
});

const GRAPH_ID = "graph-1";

function serveAgentPage(runStatus: () => AgentExecutionStatus) {
  function run() {
    const status = runStatus();
    return {
      id: "run-1",
      graph_id: GRAPH_ID,
      graph_version: 1,
      status,
      started_at: new Date(Date.now() - 60_000),
      ended_at: status === AgentExecutionStatus.RUNNING ? null : new Date(),
      is_dry_run: false,
      stats: null,
    };
  }
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Reconnect workflow",
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler(() => ({
      ...getGetV1ListGraphExecutionsResponseMock(),
      executions: [
        {
          ...getGetV1ListGraphExecutionsResponseMock().executions[0],
          ...run(),
        },
      ],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    })),
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run(),
      inputs: {},
      outputs: {},
      node_executions: [],
    })),
    getGetV1ListExecutionSchedulesForAGraphMockHandler([]),
    getGetV2ListPresetsMockHandler({
      presets: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 1,
        page_size: 100,
      },
    }),
  );
}

test("a run that finished while the live connection was down shows as finished in the runs list once it reconnects", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveAgentPage(() => status);
  const live = installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=runs">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  const first = await live.acceptNext();
  expect(
    await screen.findByRole("img", { name: "Task running" }),
  ).toBeDefined();

  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();

  expect(
    await screen.findByRole("img", { name: "Task completed" }),
  ).toBeDefined();
  expect(screen.queryByRole("img", { name: "Task running" })).toBeNull();
});
