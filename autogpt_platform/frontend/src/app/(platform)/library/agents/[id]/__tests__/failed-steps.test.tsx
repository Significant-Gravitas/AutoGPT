import { getGetV2GetSpecificBlocksMockHandler } from "@/app/api/__generated__/endpoints/default/default.msw";
import {
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
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { expect, test, vi } from "vitest";
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

const GRAPH_ID = "graph-1";
const RUN_ID = "run-1";
const MATCH_BLOCK_ID = "match-block";
const OUTPUT_BLOCK_ID = "output-block";
const REGEX_ERROR =
  "Invalid regex pattern: missing ), unterminated subpattern at position 0";

function makeRun(overrides: Partial<GraphExecutionMeta>): GraphExecutionMeta {
  return {
    id: RUN_ID,
    user_id: "user-1",
    graph_id: GRAPH_ID,
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status: AgentExecutionStatus.COMPLETED,
    started_at: new Date(),
    ended_at: new Date(),
    stats: { node_exec_count: 2, node_error_count: 0 },
    ...overrides,
  };
}

function makeNodeExecution(
  overrides: Partial<NodeExecutionResult>,
): NodeExecutionResult {
  return {
    user_id: "user-1",
    graph_id: GRAPH_ID,
    graph_version: 1,
    graph_exec_id: RUN_ID,
    node_exec_id: "node-exec-match",
    node_id: "node-match",
    block_id: MATCH_BLOCK_ID,
    status: AgentExecutionStatus.COMPLETED,
    input_data: {},
    output_data: {},
    add_time: new Date(),
    queue_time: null,
    start_time: null,
    end_time: null,
    ...overrides,
  };
}

function failedMatchStep() {
  return makeNodeExecution({
    status: AgentExecutionStatus.FAILED,
    output_data: { error: [REGEX_ERROR] },
  });
}

function makeBlock(id: string, name: string): BlockInfo {
  return {
    id,
    name,
    inputSchema: {},
    outputSchema: {},
    costs: [],
    description: "",
    categories: [],
    contributors: [],
    staticOutput: false,
    uiType: "Standard",
  };
}

interface Scenario {
  run: GraphExecutionMeta;
  nodeExecutions: NodeExecutionResult[];
  outputs?: Record<string, unknown[]>;
}

function renderRun({ run, nodeExecutions, outputs = {} }: Scenario) {
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Pattern agent",
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
      executions: [run],
      pagination: {
        total_items: 1,
        total_pages: 1,
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
    getGetV1GetExecutionDetailsMockHandler({
      ...run,
      inputs: {},
      outputs,
      node_executions: nodeExecutions,
    }),
    getGetV2GetSpecificBlocksMockHandler([
      makeBlock(MATCH_BLOCK_ID, "MatchTextPatternBlock"),
      makeBlock(OUTPUT_BLOCK_ID, "AgentOutputBlock"),
    ]),
  );

  render(
    <NuqsTestingAdapter searchParams={`activeTab=runs&activeItem=${RUN_ID}`}>
      <NewAgentLibraryView />
    </NuqsTestingAdapter>,
  );
}

test("a completed run with a failed step names the step, shows its error, and is marked in the run list", async () => {
  renderRun({
    run: makeRun({ stats: { node_exec_count: 1, node_error_count: 1 } }),
    nodeExecutions: [failedMatchStep()],
  });

  const failures = await screen.findByTestId("run-failures");
  expect(within(failures).getByText("1 step failed")).toBeDefined();
  expect(await within(failures).findByText("Match Text Pattern")).toBeDefined();
  expect(within(failures).getByText(REGEX_ERROR)).toBeDefined();
  expect(screen.getByText("Completed with errors")).toBeDefined();

  const runItem = await screen.findByTestId(`library-run-item-${RUN_ID}`);
  expect(within(runItem).getByText(/^1 step failed · /)).toBeDefined();
});

test("a run with output and a failed step shows both", async () => {
  renderRun({
    run: makeRun({ stats: { node_exec_count: 3, node_error_count: 1 } }),
    nodeExecutions: [
      failedMatchStep(),
      makeNodeExecution({
        node_exec_id: "node-exec-output",
        node_id: "node-output",
        block_id: OUTPUT_BLOCK_ID,
        output_data: { output: ["partial result"] },
      }),
    ],
    outputs: { answer: ["partial result"] },
  });

  expect(await screen.findByText("partial result")).toBeDefined();
  const failures = screen.getByTestId("run-failures");
  expect(within(failures).getByText(REGEX_ERROR)).toBeDefined();
  expect(within(failures).queryByText("Agent Output")).toBeNull();
});

test("a clean run shows no failure notice and a plain completed state", async () => {
  renderRun({
    run: makeRun({}),
    nodeExecutions: [makeNodeExecution({ output_data: { positive: ["ok"] } })],
  });

  const runItem = await screen.findByTestId(`library-run-item-${RUN_ID}`);
  expect(await screen.findByText("Completed")).toBeDefined();
  expect(screen.queryByTestId("run-failures")).toBeNull();
  expect(screen.queryByText("Completed with errors")).toBeNull();
  expect(within(runItem).queryByText(/step failed/)).toBeNull();
});

test("a completed run that counted a failed step says so even without the step's details", async () => {
  renderRun({
    run: makeRun({ stats: { node_exec_count: 2, node_error_count: 1 } }),
    nodeExecutions: [makeNodeExecution({ output_data: { positive: ["ok"] } })],
  });

  const failures = await screen.findByTestId("run-failures");
  expect(within(failures).getByText("1 step failed")).toBeDefined();
  expect(within(failures).queryByRole("listitem")).toBeNull();
  expect(screen.getByText("Completed with errors")).toBeDefined();
});

test("steps swept to FAILED when the user stops a run are not reported as errors", async () => {
  renderRun({
    run: makeRun({ status: AgentExecutionStatus.TERMINATED }),
    nodeExecutions: [
      makeNodeExecution({ status: AgentExecutionStatus.FAILED }),
    ],
  });

  expect(await screen.findByText("Stopped")).toBeDefined();
  expect(screen.queryByTestId("run-failures")).toBeNull();
});

test("a run that failed without a failed step shows the run's error", async () => {
  renderRun({
    run: makeRun({
      status: AgentExecutionStatus.FAILED,
      stats: { error: "Insufficient balance to run this agent" },
    }),
    nodeExecutions: [],
  });

  const failures = await screen.findByTestId("run-failures");
  expect(within(failures).getByText("This run failed")).toBeDefined();
  expect(
    within(failures).getByText("Insufficient balance to run this agent"),
  ).toBeDefined();
});
