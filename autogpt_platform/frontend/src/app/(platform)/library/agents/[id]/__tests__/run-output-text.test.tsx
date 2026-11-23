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
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { expect, test, vi } from "vitest";
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

const GRAPH_ID = "graph-1";

// Text outputs that JSON.parse reads as a number or a boolean. The run view
// should show each one as the text it is.
const TEXT_OUTPUTS: Record<string, string> = {
  exponent_text: "1e5",
  price_text: "1.50",
  padded_text: " 42 ",
  long_id: "12345678901234567890",
  boolean_text: "true",
};

const run: GraphExecutionMeta = {
  id: "run-a",
  user_id: "user-1",
  graph_id: GRAPH_ID,
  graph_version: 1,
  inputs: {},
  credential_inputs: {},
  nodes_input_masks: {},
  preset_id: null,
  status: AgentExecutionStatus.COMPLETED,
  started_at: new Date(Date.now() - 60_000),
  ended_at: new Date(Date.now() - 60_000),
  stats: { node_exec_count: 1, node_error_count: 0 },
};

function mockRunWithOutputs(outputs: Record<string, unknown[]>) {
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Price check",
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
    getGetV1GetExecutionDetailsMockHandler({
      ...run,
      inputs: {},
      outputs,
      node_executions: [],
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
}

function renderRunView() {
  render(
    <NuqsTestingAdapter
      hasMemory
      searchParams={`activeTab=runs&activeItem=${run.id}`}
    >
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
}

// What the run view shows for an output: the value beside its label.
async function shownOutput(name: string) {
  const label = await screen.findByText(name, {
    selector: '[data-testid="output-item-label"]',
  });
  const value = label.parentElement?.querySelector<HTMLElement>(
    '[data-testid="output-item-value"]',
  );
  if (!value) throw new Error(`the run view shows no value for ${name}`);
  return value;
}

test("the run view shows text that reads as a JSON number or boolean exactly as the text", async () => {
  mockRunWithOutputs(
    Object.fromEntries(
      Object.entries(TEXT_OUTPUTS).map(([name, text]) => [name, [text]]),
    ),
  );

  renderRunView();

  const shown: Record<string, string | null> = {};
  for (const name of Object.keys(TEXT_OUTPUTS)) {
    shown[name] = (await shownOutput(name)).textContent;
  }
  expect(shown).toEqual(TEXT_OUTPUTS);
});

// Text holding a JSON object still goes to a renderer that lays it out (code,
// or JSON), ahead of the text renderer that now takes number-like text.
test("the run view shows text holding a JSON object laid out, not as the bare text", async () => {
  const objectText = '{"a":1}';
  mockRunWithOutputs({ json_text: [objectText] });

  renderRunView();

  const { textContent } = await shownOutput("json_text");
  expect(textContent, "the run view should show the object").toMatch(
    /"a":\s*1/,
  );
  expect(
    textContent,
    "the run view should lay the object out, not show its bare text",
  ).not.toBe(objectText);
});
