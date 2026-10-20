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
const EMAIL = "Quarterly numbers: revenue is down 12%";

const GOOGLE_CREDENTIALS = {
  type: "object",
  properties: {
    google_credentials: {
      type: "object",
      credentials_provider: ["google"],
      credentials_types: ["oauth2"],
    },
  },
};

function makeRun(): GraphExecutionMeta {
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
    stats: { node_exec_count: 1, node_error_count: 0 },
  };
}

function renderRun(credentialsInputSchema: Record<string, unknown>) {
  const run = makeRun();
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Inbox digest",
        input_schema: { type: "object", properties: {} },
        credentials_input_schema: credentialsInputSchema,
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
      outputs: { digest: [EMAIL] },
      node_executions: [],
    }),
    getGetV2GetSpecificBlocksMockHandler([]),
  );

  render(
    <NuqsTestingAdapter searchParams={`activeTab=runs&activeItem=${RUN_ID}`}>
      <NewAgentLibraryView />
    </NuqsTestingAdapter>,
  );
}

// Every workflow's output is hidden, not only one listing a Google account: a
// Google Drive picker's credential is left out of credentials_input_schema, so
// a workflow reading a picked spreadsheet lists none.
test.each([
  ["lists a Google account", GOOGLE_CREDENTIALS],
  ["lists no Google account", { type: "object", properties: {} }],
])(
  "a run output is kept out of session replays when the workflow %s",
  async (_, credentialsInputSchema) => {
    renderRun(credentialsInputSchema);

    const output = await screen.findByText(EMAIL);
    expect(output.closest(".ph-no-capture")).not.toBeNull();
    expect(output.closest(".sentry-block")).not.toBeNull();
  },
);
