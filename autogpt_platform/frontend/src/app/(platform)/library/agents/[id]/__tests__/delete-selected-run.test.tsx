import { getGetV2GetSpecificBlocksMockHandler } from "@/app/api/__generated__/endpoints/default/default.msw";
import {
  getDeleteV1DeleteGraphExecutionMockHandler,
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1ListGraphExecutionsMockHandler,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { server } from "@/mocks/mock-server";
import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse } from "msw";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeEach, expect, test, vi } from "vitest";
import AgentLibraryPage from "../page";

const toast = vi.fn();

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  useToast: () => ({ toast }),
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

const GRAPH_ID = "graph-1";

type User = ReturnType<typeof userEvent.setup>;

let listedRuns: GraphExecutionMeta[] = [];

function makeRun(id: string, minutesAgo: number): GraphExecutionMeta {
  const startedAt = new Date(Date.now() - minutesAgo * 60_000);
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
    started_at: startedAt,
    ended_at: startedAt,
    stats: { node_exec_count: 1, node_error_count: 0 },
  };
}

function outputOf(runID: string) {
  return `Output of ${runID}`;
}

beforeEach(() => {
  toast.mockClear();
  // Newest first, the way the API lists them.
  listedRuns = [makeRun("run-a", 1), makeRun("run-b", 2), makeRun("run-c", 3)];
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Inbox digest",
        input_schema: { type: "object", properties: {} },
        credentials_input_schema: { type: "object", properties: {} },
        has_external_trigger: false,
        trigger_setup_info: null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
        recommended_schedule_cron: null,
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler(() => ({
      executions: listedRuns,
      pagination: {
        total_items: listedRuns.length,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    })),
    getGetV1GetExecutionDetailsMockHandler(({ params }) => {
      const run = listedRuns.find(({ id }) => id === params.graphExecId);
      if (!run) {
        throw HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }
      return {
        ...run,
        inputs: {},
        outputs: { result: [outputOf(run.id)] },
        node_executions: [],
      };
    }),
    getDeleteV1DeleteGraphExecutionMockHandler(({ params }) => {
      listedRuns = listedRuns.filter(({ id }) => id !== params.graphExecId);
    }),
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
    getGetV2GetSpecificBlocksMockHandler([]),
  );
});

async function renderWithRunSelected(runID: string) {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
  render(
    <NuqsTestingAdapter
      hasMemory
      onUrlUpdate={onUrlUpdate}
      searchParams="activeTab=runs"
    >
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  await user.click(await screen.findByTestId(`library-run-item-${runID}`));
  expect(await screen.findByText(outputOf(runID))).toBeDefined();

  return {
    user,
    selectedRunInURL: () =>
      onUrlUpdate.mock.lastCall?.[0].searchParams.get("activeItem"),
  };
}

async function deleteThroughSidebarMenu(user: User, runID: string) {
  const row = screen.getByTestId(`library-run-item-${runID}`);
  await user.click(within(row).getByRole("button", { name: "More actions" }));
  await user.click(
    await screen.findByRole("menuitem", { name: "Delete task" }),
  );
  await user.click(await screen.findByRole("button", { name: "Delete Task" }));
}

async function deleteThroughRunView(user: User) {
  const runActions = screen.getByRole("link", { name: "View task details" })
    .parentElement as HTMLElement;
  await user.click(
    within(runActions).getByRole("button", { name: "More actions" }),
  );
  await user.click(
    await screen.findByRole("menuitem", { name: "Delete this task" }),
  );
  await user.click(await screen.findByRole("button", { name: "Delete Task" }));
}

test.each([
  ["the sidebar's menu", deleteThroughSidebarMenu],
  ["the run view's menu", deleteThroughRunView],
])(
  "deleting the selected run through %s selects the newest run left",
  async (_, deleteRun) => {
    const { user, selectedRunInURL } = await renderWithRunSelected("run-b");

    await deleteRun(user, "run-b");

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({ title: "Task deleted" }),
    );
    expect(await screen.findByText(outputOf("run-a"))).toBeDefined();
    expect(screen.queryByText(outputOf("run-b"))).toBeNull();
    expect(screen.queryByTestId("library-run-item-run-b")).toBeNull();
    await waitFor(() => expect(selectedRunInURL()).toBe("run-a"));
  },
);

test("deleting a run that isn't selected through the sidebar's menu keeps the selection", async () => {
  const { user, selectedRunInURL } = await renderWithRunSelected("run-b");

  await deleteThroughSidebarMenu(user, "run-c");

  await waitFor(() =>
    expect(screen.queryByTestId("library-run-item-run-c")).toBeNull(),
  );
  expect(toast).toHaveBeenCalledWith({ title: "Task deleted" });
  // Let the delete handler finish what it does after the list refetch.
  await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
  expect(screen.getByText(outputOf("run-b"))).toBeDefined();
  expect(selectedRunInURL()).toBe("run-b");
});

test("selecting another run while the selected run's delete is in flight keeps the new selection", async () => {
  let releaseDelete = () => {};
  const deleteHeld = new Promise<void>((resolve) => {
    releaseDelete = resolve;
  });
  server.use(
    getDeleteV1DeleteGraphExecutionMockHandler(async ({ params }) => {
      await deleteHeld;
      listedRuns = listedRuns.filter(({ id }) => id !== params.graphExecId);
    }),
  );

  try {
    const { user, selectedRunInURL } = await renderWithRunSelected("run-b");

    await deleteThroughSidebarMenu(user, "run-b");
    await user.keyboard("{Escape}");
    // Not run-a: the newest run is what an emptied selection falls back to.
    await user.click(screen.getByTestId("library-run-item-run-c"));
    expect(await screen.findByText(outputOf("run-c"))).toBeDefined();

    releaseDelete();

    await waitFor(() =>
      expect(screen.queryByTestId("library-run-item-run-b")).toBeNull(),
    );
    expect(toast).toHaveBeenCalledWith({ title: "Task deleted" });
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
    expect(screen.getByText(outputOf("run-c"))).toBeDefined();
    expect(selectedRunInURL()).toBe("run-c");
  } finally {
    releaseDelete();
  }
});
