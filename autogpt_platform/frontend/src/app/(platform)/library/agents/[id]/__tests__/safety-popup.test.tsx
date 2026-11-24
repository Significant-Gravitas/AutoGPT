import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1ListGraphExecutionsMockHandler,
  getPatchV1UpdateGraphSettingsMockHandler,
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphSettings } from "@/app/api/__generated__/models/graphSettings";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, expect, test, vi } from "vitest";
import { NewAgentLibraryView } from "../components/NewAgentLibraryView/NewAgentLibraryView";

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

beforeEach(() => {
  window.localStorage.clear();
});

const EMPTY_PAGE = {
  total_items: 0,
  total_pages: 0,
  current_page: 1,
  page_size: 20,
};

// A new workflow with a human-in-the-loop block, whose settings PATCHes land
// in the agent the page reads back.
function serveApprovalWorkflow() {
  let settings: GraphSettings = {
    human_in_the_loop_safe_mode: true,
    sensitive_action_safe_mode: false,
  };
  const agent = getGetV2GetLibraryAgentResponseMock({
    id: "library-agent",
    graph_id: "graph-1",
    graph_version: 1,
    name: "Approval workflow",
    input_schema: { type: "object", properties: {} },
    credentials_input_schema: { type: "object", properties: {} },
    has_external_trigger: false,
    trigger_setup_info: null,
    has_sensitive_action: false,
    has_human_in_the_loop: true,
    recommended_schedule_cron: null,
  });
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
  const runs = vi.fn(() => run);

  server.use(
    getGetV2GetLibraryAgentMockHandler(() => ({ ...agent, settings })),
    getPatchV1UpdateGraphSettingsMockHandler(async ({ request }) => {
      settings = { ...settings, ...((await request.json()) as GraphSettings) };
      return settings;
    }),
    getGetV1ListGraphExecutionsMockHandler({
      executions: [],
      pagination: EMPTY_PAGE,
    }),
    getGetV1ListExecutionSchedulesForAGraphMockHandler([]),
    getGetV2ListPresetsMockHandler({
      presets: [],
      pagination: { ...EMPTY_PAGE, page_size: 100 },
    }),
    getPostV1ExecuteGraphAgentMockHandler(runs),
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run,
      inputs: {},
      outputs: {},
      node_executions: [],
    })),
  );
  return runs;
}

function renderPage() {
  render(
    <NuqsTestingAdapter hasMemory>
      <NewAgentLibraryView />
    </NuqsTestingAdapter>,
  );
}

async function startTask(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    await screen.findByRole("button", { name: "Setup your task" }),
  );
  await user.click(await screen.findByRole("button", { name: "Start Task" }));
}

test("holds the first run of a workflow with human-in-the-loop review on behind the safety popup", async () => {
  const runs = serveApprovalWorkflow();
  const user = userEvent.setup();
  renderPage();

  await startTask(user);

  const popup = await screen.findByRole("dialog", { name: "Safety Checks" });
  expect(runs).not.toHaveBeenCalled();
  await user.click(within(popup).getByRole("button", { name: "Got it" }));
  await waitFor(() => expect(runs).toHaveBeenCalledOnce());
});

test("starts the first run without the safety popup once Workflow Settings turns that review off", async () => {
  const runs = serveApprovalWorkflow();
  const user = userEvent.setup();
  renderPage();

  // The page swaps to its empty-state layout once the runs list is in; a
  // dialog opened before that is unmounted with the old layout.
  await screen.findByRole("button", { name: "Setup your task" });
  await user.click(screen.getByRole("button", { name: "Workflow Settings" }));
  const review = await screen.findByRole("switch", {
    name: "Human-in-the-loop approval",
  });
  expect(review.getAttribute("aria-checked")).toBe("true");
  await user.click(review);
  await waitFor(() =>
    expect(
      screen
        .getByRole("switch", { name: "Human-in-the-loop approval" })
        .getAttribute("aria-checked"),
    ).toBe("false"),
  );
  await user.keyboard("{Escape}");

  await startTask(user);

  expect(screen.queryByRole("dialog", { name: "Safety Checks" })).toBeNull();
  await waitFor(() => expect(runs).toHaveBeenCalledOnce());
});
