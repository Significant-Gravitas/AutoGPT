import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1ListGraphExecutionsMockHandler,
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2GetASpecificPresetResponseMock,
  getGetV2ListPresetsMockHandler,
} from "@/app/api/__generated__/endpoints/presets/presets.msw";
import {
  getGetV1ListExecutionSchedulesForAGraphMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { expect, test, vi } from "vitest";
import { NewAgentLibraryView } from "../components/NewAgentLibraryView/NewAgentLibraryView";
import { useNewAgentLibraryView } from "../components/NewAgentLibraryView/useNewAgentLibraryView";

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

test("selects the first created run while the sidebar still has an empty cached list", async () => {
  const run = getPostV1ExecuteGraphAgentResponseMock({
    id: "new-run",
    graph_id: "graph-1",
    graph_version: 1,
    status: AgentExecutionStatus.COMPLETED,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    stats: null,
  });
  const executionRequests = vi.fn(() => run);
  const detailRequests = vi.fn(() => ({
    ...getGetV1GetExecutionDetailsResponseMock(),
    ...run,
    inputs: {},
    outputs: {},
    node_executions: [],
  }));
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: 1,
        name: "First run agent",
        input_schema: { type: "object", properties: {} },
        credentials_input_schema: { type: "object", properties: {} },
        has_external_trigger: false,
        trigger_setup_info: null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
        recommended_schedule_cron: null,
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler({
      executions: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 1,
        page_size: 20,
      },
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
    getPostV1ExecuteGraphAgentMockHandler(executionRequests),
    getGetV1GetExecutionDetailsMockHandler(detailRequests),
  );

  render(
    <NuqsTestingAdapter hasMemory>
      <NewAgentLibraryView />
    </NuqsTestingAdapter>,
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Setup your task" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Start Task" }),
  );
  await waitFor(() => expect(executionRequests).toHaveBeenCalledOnce());

  await waitFor(() => expect(detailRequests).toHaveBeenCalled());
  expect(await screen.findByRole("button", { name: "Output" })).toBeDefined();
  expect(screen.queryByRole("button", { name: "Setup your task" })).toBeNull();
});

interface Props {
  type: "scheduled" | "triggers";
}

function CreatedItemProbe({ type }: Props) {
  const view = useNewAgentLibraryView();

  function createItem() {
    if (type === "scheduled") {
      view.onScheduleCreated(
        getPostV1CreateExecutionScheduleResponseMock({ id: "new-item" }),
      );
    } else {
      view.onTriggerSetup(
        getGetV2GetASpecificPresetResponseMock({
          id: "new-item",
          webhook_id: "webhook-1",
        }),
      );
    }
  }

  function reportEmptyCounts() {
    view.handleCountsChange({
      runsCount: 0,
      schedulesCount: 0,
      templatesCount: 0,
      triggersCount: 0,
      loading: false,
      hasError: false,
    });
  }

  return (
    <>
      <button disabled={!view.ready} onClick={createItem}>
        Create item
      </button>
      <button onClick={reportEmptyCounts}>Refresh empty counts</button>
      <output>{`${view.activeTab}:${view.activeItemId}`}</output>
    </>
  );
}

test.each(["scheduled", "triggers"] as const)(
  "keeps a newly created %s item selected when empty counts arrive",
  async (type) => {
    server.use(
      getGetV2GetLibraryAgentMockHandler(
        getGetV2GetLibraryAgentResponseMock({
          id: "library-agent",
          graph_id: "graph-1",
        }),
      ),
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
    const onUrlUpdate = vi.fn();
    render(
      <NuqsTestingAdapter hasMemory onUrlUpdate={onUrlUpdate}>
        <CreatedItemProbe type={type} />
      </NuqsTestingAdapter>,
    );
    const create = screen.getByRole("button", { name: "Create item" });
    await waitFor(() => expect(create.hasAttribute("disabled")).toBe(false));
    await userEvent.click(create);
    await userEvent.click(
      screen.getByRole("button", { name: "Refresh empty counts" }),
    );
    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toBe(`${type}:new-item`);
      const update = onUrlUpdate.mock.lastCall?.[0];
      expect(update?.searchParams.get("activeItem")).toBe(
        type === "triggers" ? "preset:new-item" : "new-item",
      );
    });
  },
);
