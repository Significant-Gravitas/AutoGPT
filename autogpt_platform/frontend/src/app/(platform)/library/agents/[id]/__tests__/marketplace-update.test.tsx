import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1ListGraphExecutionsMockHandler,
  getGetV1ListGraphExecutionsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
  getPatchV2UpdateLibraryAgentMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getGetV2GetSpecificAgentQueryKey } from "@/app/api/__generated__/endpoints/store/store";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2GetSpecificAgentResponseMock,
  getGetV2ListMySubmissionsMockHandler,
  getGetV2ListMySubmissionsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, expect, test, vi } from "vitest";
import AgentLibraryPage from "../page";

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
  useAuthStore.setState({ user: null });
});

const run = {
  ...getGetV1ListGraphExecutionsResponseMock().executions[0],
  id: "run-1",
  graph_id: "graph-1",
  graph_version: 1,
  status: AgentExecutionStatus.COMPLETED,
  started_at: new Date(Date.now() - 120_000),
  ended_at: new Date(Date.now() - 60_000),
  is_dry_run: false,
  stats: null,
};

interface ServeOptions {
  libraryVersion?: number;
  listedVersions: string[];
  withRun?: boolean;
  submittedVersion?: number;
}

function serveMarketplaceCopy({
  libraryVersion = 1,
  listedVersions,
  withRun = false,
  submittedVersion,
}: ServeOptions) {
  const updateRequests = vi.fn();
  const isCreator = submittedVersion !== undefined;
  const agent = getGetV2GetLibraryAgentResponseMock({
    id: "library-agent",
    graph_id: "graph-1",
    graph_version: libraryVersion,
    name: "Morning digest",
    can_access_graph: isCreator,
    marketplace_listing: {
      id: "listing-1",
      name: "Morning digest",
      slug: "morning-digest",
      creator: { id: "creator-1", name: "Creator", slug: "creator" },
    },
  });
  if (isCreator) {
    useAuthStore.setState({
      user: { id: "user-1", email: "u@example.com", user_metadata: {} },
    });
  }
  const submissions = getGetV2ListMySubmissionsResponseMock();
  server.use(
    getGetV2GetLibraryAgentMockHandler(agent),
    getGetV1ListGraphExecutionsMockHandler({
      executions: withRun ? [run] : [],
      pagination: {
        total_items: withRun ? 1 : 0,
        total_pages: withRun ? 1 : 0,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run,
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
    getGetV2GetSpecificAgentMockHandler(
      getGetV2GetSpecificAgentResponseMock({
        slug: "morning-digest",
        graph_id: "graph-1",
        graph_versions: listedVersions,
      }),
    ),
    getGetV2ListMySubmissionsMockHandler({
      ...submissions,
      submissions:
        submittedVersion === undefined
          ? []
          : [
              {
                ...submissions.submissions[0],
                graph_id: "graph-1",
                graph_version: submittedVersion,
              },
            ],
    }),
    getPatchV2UpdateLibraryAgentMockHandler(async ({ request }) => {
      updateRequests(await request.json());
      return agent;
    }),
  );
  return { updateRequests };
}

function renderPage(queryClient?: QueryClient) {
  render(
    <NuqsTestingAdapter hasMemory>
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
    queryClient ? { queryClient } : undefined,
  );
}

test("a workflow never run offers the newer Marketplace version and updates to it", async () => {
  const { updateRequests } = serveMarketplaceCopy({
    listedVersions: ["1", "2"],
  });
  renderPage();

  expect(await screen.findByText("Ready to get started?")).toBeDefined();
  expect(await screen.findByText("Update available")).toBeDefined();

  await userEvent.click(
    screen.getByRole("button", { name: "Update workflow" }),
  );

  await waitFor(() =>
    expect(updateRequests).toHaveBeenCalledWith({ graph_version: 2 }),
  );
});

test("a workflow never run on the latest Marketplace version shows no update", async () => {
  serveMarketplaceCopy({ listedVersions: ["1"] });
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  renderPage(queryClient);

  expect(await screen.findByText("Ready to get started?")).toBeDefined();
  // Wait for the listing to be applied, not just requested.
  await waitFor(() =>
    expect(
      queryClient.getQueryState(
        getGetV2GetSpecificAgentQueryKey("creator", "morning-digest", {}),
      )?.status,
    ).toBe("success"),
  );

  expect(screen.queryByText("Update available")).toBeNull();
  expect(screen.queryByRole("button", { name: "Update workflow" })).toBeNull();
});

test("a creator's workflow never run offers to publish its unpublished changes", async () => {
  serveMarketplaceCopy({
    libraryVersion: 3,
    listedVersions: ["2"],
    submittedVersion: 2,
  });
  renderPage();

  expect(await screen.findByText("Ready to get started?")).toBeDefined();
  expect(await screen.findByText("Unpublished changes")).toBeDefined();
  expect(screen.getByRole("button", { name: "Publish changes" })).toBeDefined();
  expect(screen.queryByText("Update available")).toBeNull();
});

test("a workflow with runs offers the newer Marketplace version too", async () => {
  const { updateRequests } = serveMarketplaceCopy({
    listedVersions: ["1", "2"],
    withRun: true,
  });
  renderPage();

  expect(await screen.findByText("Update available")).toBeDefined();
  expect(screen.queryByText("Ready to get started?")).toBeNull();

  await userEvent.click(
    screen.getByRole("button", { name: "Update workflow" }),
  );

  await waitFor(() =>
    expect(updateRequests).toHaveBeenCalledWith({ graph_version: 2 }),
  );
});
