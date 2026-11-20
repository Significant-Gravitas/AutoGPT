import { getGetV1ListGraphExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { getGetV1ListGraphExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import { act, render, screen } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { ReactNode } from "react";
import { beforeEach, expect, test, vi } from "vitest";
import { SidebarRunsList } from "../components/NewAgentLibraryView/components/sidebar/SidebarRunsList/SidebarRunsList";

const GRAPH_ID = "graph-1";

const renderedRunIDs = vi.hoisted(() => [] as string[]);

vi.mock(
  "../components/NewAgentLibraryView/components/sidebar/SidebarRunsList/components/TaskListItem",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("../components/NewAgentLibraryView/components/sidebar/SidebarRunsList/components/TaskListItem")
      >();
    function CountedTaskListItem(
      props: Parameters<typeof actual.TaskListItem>[0],
    ) {
      renderedRunIDs.push(props.run.id);
      return <actual.TaskListItem {...props} />;
    }
    return { TaskListItem: CountedTaskListItem };
  },
);

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

function makeRun(id: string): GraphExecutionMeta {
  return {
    id,
    user_id: "user-1",
    graph_id: GRAPH_ID,
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status: AgentExecutionStatus.COMPLETED,
    started_at: new Date("2026-01-05T09:00:00Z"),
    ended_at: new Date("2026-01-05T09:00:30Z"),
    stats: { node_exec_count: 2, node_error_count: 0 },
  };
}

function page(executions: GraphExecutionMeta[]) {
  return {
    executions,
    pagination: {
      total_items: executions.length,
      total_pages: 1,
      current_page: 1,
      page_size: 20,
    },
  };
}

const agent = getGetV2GetLibraryAgentResponseMock({
  id: "library-agent",
  graph_id: GRAPH_ID,
  name: "Prepend workflow",
});

let listedRuns: GraphExecutionMeta[] = [];

beforeEach(() => {
  renderedRunIDs.length = 0;
  listedRuns = [makeRun("run-a"), makeRun("run-b")];
  server.use(
    getGetV1ListGraphExecutionsMockHandler(() => page(listedRuns)),
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
});

function withProviders(ui: ReactNode, queryClient: QueryClient) {
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter searchParams="activeTab=runs&activeItem=run-a">
        <BackendAPIProvider>
          <TooltipProvider>{ui}</TooltipProvider>
        </BackendAPIProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>
  );
}

function renderList(ui: ReactNode, queryClient: QueryClient) {
  return render(withProviders(ui, queryClient));
}

function rendersOf(id: string) {
  return renderedRunIDs.filter((renderedID) => renderedID === id).length;
}

test("a run prepended to the list renders only the new row and keeps the others' elements", async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  renderList(
    <SidebarRunsList
      agent={agent}
      selectedRunId="run-a"
      onSelectRun={vi.fn()}
    />,
    queryClient,
  );

  const runA = await screen.findByTestId("library-run-item-run-a");
  const runB = screen.getByTestId("library-run-item-run-b");
  const rendersBefore = { a: rendersOf("run-a"), b: rendersOf("run-b") };

  listedRuns = [makeRun("run-new"), makeRun("run-a"), makeRun("run-b")];
  await act(async () => {
    await queryClient.refetchQueries({
      queryKey: getGetV1ListGraphExecutionsQueryKey(GRAPH_ID),
    });
  });

  expect(await screen.findByTestId("library-run-item-run-new")).toBeDefined();
  expect(screen.getByTestId("library-run-item-run-a")).toBe(runA);
  expect(screen.getByTestId("library-run-item-run-b")).toBe(runB);
  expect(rendersOf("run-a")).toBe(rendersBefore.a);
  expect(rendersOf("run-b")).toBe(rendersBefore.b);
  expect(rendersOf("run-new")).toBe(1);
  expect(
    screen
      .getAllByTestId(/^library-run-item-/)
      .map((row) => row.dataset.testid),
  ).toEqual([
    "library-run-item-run-new",
    "library-run-item-run-a",
    "library-run-item-run-b",
  ]);
});

test("clicking a run selects it through the latest onSelectRun", async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const firstOnSelectRun = vi.fn();
  const { rerender } = renderList(
    <SidebarRunsList
      agent={agent}
      selectedRunId="run-a"
      onSelectRun={firstOnSelectRun}
    />,
    queryClient,
  );

  await userEvent.click(await screen.findByTestId("library-run-item-run-b"));
  expect(firstOnSelectRun).toHaveBeenCalledWith("run-b", "runs");

  const nextOnSelectRun = vi.fn();
  rerender(
    withProviders(
      <SidebarRunsList
        agent={agent}
        selectedRunId="run-a"
        onSelectRun={nextOnSelectRun}
      />,
      queryClient,
    ),
  );
  await userEvent.click(screen.getByTestId("library-run-item-run-b"));
  expect(nextOnSelectRun).toHaveBeenCalledWith("run-b", "runs");
  expect(firstOnSelectRun).toHaveBeenCalledTimes(1);
});

test("a run's label shows when it started, relative to now", async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  listedRuns = [
    { ...makeRun("run-a"), started_at: new Date(Date.now() - 5 * 60_000) },
  ];
  renderList(
    <SidebarRunsList
      agent={agent}
      selectedRunId="run-a"
      onSelectRun={vi.fn()}
    />,
    queryClient,
  );

  const row = await screen.findByTestId("library-run-item-run-a");
  expect(row.textContent).toContain("5 minutes ago");
});
