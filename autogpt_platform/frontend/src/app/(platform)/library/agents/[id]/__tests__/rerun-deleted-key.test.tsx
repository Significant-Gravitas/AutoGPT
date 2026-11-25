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
import { CredentialsMetaInputType } from "@/app/api/__generated__/models/credentialsMetaInputType";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
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

// CredentialsProvider only loads credentials for a signed-in user.
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

const FIELD = "firecrawl_api_key_credentials";
const PICKER = "Select Firecrawl credential";
const SYSTEM_SECTION = /^System credentials/;
const RUN_ID = "past-run";
const NEW_RUN_ID = "new-run";
const RUN_INPUTS = { topic: "news" };

// The platform's own Firecrawl key, as /integrations/credentials lists it.
const PLATFORM_KEY = {
  id: "firecrawl-default",
  provider: "firecrawl",
  type: "api_key",
  title: "Firecrawl API Key",
  is_system: true,
};
// The key the past run used
const USED_KEY = {
  id: "user-firecrawl-key",
  provider: "firecrawl",
  type: "api_key",
  title: "My Firecrawl key",
  is_system: false,
};
const OTHER_KEY = {
  id: "other-firecrawl-key",
  provider: "firecrawl",
  type: "api_key",
  title: "Other Firecrawl key",
  is_system: false,
};

const FIRECRAWL_FIELD = {
  credentials_provider: ["firecrawl"],
  credentials_types: ["api_key"],
  title: "Credentials",
  type: "object",
  properties: {
    id: { title: "Id", type: "string" },
    provider: { const: "firecrawl", type: "string" },
    type: { const: "api_key", type: "string" },
  },
  required: ["id", "provider", "type"],
};

// What a run's credential_inputs hold for a key
function usedBy(key: { id: string; provider: string; title: string }) {
  return {
    id: key.id,
    provider: key.provider,
    type: CredentialsMetaInputType.api_key,
    title: key.title,
  };
}

type ExecuteBody = {
  inputs?: Record<string, unknown>;
  credentials_inputs?: Record<string, { id?: string }>;
};

interface Setup {
  // The key the past run used
  runKey?: { id: string; provider: string; title: string };
  // The keys /integrations/credentials lists, read again on every request
  savedCredentials?: () => Array<Record<string, unknown>>;
  // How the server answers a run request
  answerRun?: () => Response;
}

function startedRun() {
  return HttpResponse.json({
    id: NEW_RUN_ID,
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status: AgentExecutionStatus.QUEUED,
    stats: null,
  });
}

function setup({
  runKey = USED_KEY,
  savedCredentials = () => [PLATFORM_KEY, OTHER_KEY],
  answerRun = startedRun,
}: Setup = {}) {
  const pastRun: GraphExecutionMeta = {
    id: RUN_ID,
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    inputs: RUN_INPUTS,
    credential_inputs: { [FIELD]: usedBy(runKey) },
    nodes_input_masks: {},
    preset_id: null,
    status: AgentExecutionStatus.COMPLETED,
    started_at: new Date("2026-11-20T10:00:00Z"),
    ended_at: new Date("2026-11-20T10:01:00Z"),
    stats: { node_exec_count: 1, node_error_count: 0 },
  };
  const executeBodies: ExecuteBody[] = [];
  // Recorded as each run request arrives, before its body is read
  const runRequests: string[] = [];

  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: 1,
        name: "Scrape agent",
        can_access_graph: true,
        input_schema: {
          type: "object",
          properties: { topic: { title: "Topic", type: "string" } },
          required: ["topic"],
        },
        credentials_input_schema: {
          type: "object",
          properties: { [FIELD]: FIRECRAWL_FIELD },
          required: [FIELD],
        },
        has_external_trigger: false,
        trigger_setup_info: null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
        recommended_schedule_cron: null,
      }),
    ),
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([
        { name: "firecrawl", description: "firecrawl provider" },
      ]),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json(["firecrawl"]),
    ),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json(savedCredentials()),
    ),
    getGetV1ListGraphExecutionsMockHandler({
      executions: [pastRun],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1GetExecutionDetailsMockHandler(({ params }) => ({
      ...pastRun,
      id: String(params.graphExecId),
      inputs: RUN_INPUTS,
      outputs: { result: [`Output of ${params.graphExecId}`] },
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
    http.post(
      "/api/proxy/api/graphs/:graphId/execute/:graphVersion",
      async ({ request }) => {
        runRequests.push(request.url);
        executeBodies.push((await request.json()) as ExecuteBody);
        return answerRun();
      },
    ),
  );

  render(
    <>
      <CredentialsProvider>
        <NuqsTestingAdapter
          hasMemory
          searchParams={`activeTab=runs&activeItem=${RUN_ID}`}
        >
          <NewAgentLibraryView />
        </NuqsTestingAdapter>
      </CredentialsProvider>
      <Toaster />
    </>,
  );

  return { executeBodies, runRequests };
}

// The run view is open and the user's saved keys have loaded: the run's
// "Your input" shows its credential field only once they have
async function waitForRunView() {
  expect(await screen.findByText(`Output of ${RUN_ID}`)).toBeDefined();
  expect(await screen.findByText("Firecrawl credentials")).toBeDefined();
}

async function rerun() {
  await userEvent.click(screen.getByRole("button", { name: "Rerun task" }));
}

function keyIn(body: ExecuteBody | undefined) {
  return body?.credentials_inputs?.[FIELD]?.id;
}

async function clickWhenEnabled(name: string, container: HTMLElement) {
  const button = await within(container).findByRole("button", { name });
  await waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
  await userEvent.click(button);
}

test("rerunning a task whose key was deleted sends nothing, names the key and opens the run form with the task's inputs", async () => {
  const { runRequests } = setup();
  await waitForRunView();

  await rerun();

  const form = await screen.findByRole("dialog", { name: "Rerun task" });
  expect(
    within(form).getByText(
      "A key this task used was deleted: My Firecrawl key. Choose another key below to run it again.",
    ),
  ).toBeDefined();
  expect(within(form).getByDisplayValue("news")).toBeDefined();
  // The field is shown, outside the collapsed system credentials, and left
  // for the user to pick: not moved to the platform's key
  const picker = await within(form).findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe("__none__");
  expect(
    within(form).queryByRole("button", { name: SYSTEM_SECTION }),
  ).toBeNull();
  expect(
    within(form)
      .getByRole("button", { name: "Start Task" })
      .hasAttribute("disabled"),
  ).toBe(true);
  expect(runRequests).toHaveLength(0);
  expect(screen.queryByText("Failed to start run")).toBeNull();
});

test("taking the offer runs the task again with the key the user picks", async () => {
  const { executeBodies } = setup();
  await waitForRunView();

  await rerun();
  const form = await screen.findByRole("dialog", { name: "Rerun task" });
  await userEvent.selectOptions(
    await within(form).findByRole("combobox", { name: PICKER }),
    OTHER_KEY.id,
  );
  await clickWhenEnabled("Start Task", form);

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0])).toBe(OTHER_KEY.id);
  expect(executeBodies[0].inputs).toEqual({ topic: "news" });
  // The page moves to the new run
  expect(await screen.findByText(`Output of ${NEW_RUN_ID}`)).toBeDefined();
});

test("rerunning a task whose key still exists starts it straight away", async () => {
  const { executeBodies } = setup({
    savedCredentials: () => [PLATFORM_KEY, USED_KEY, OTHER_KEY],
  });
  await waitForRunView();

  await rerun();

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0])).toBe(USED_KEY.id);
  expect(executeBodies[0].inputs).toEqual({ topic: "news" });
  expect(await screen.findByText("Run started")).toBeDefined();
  expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();
});

test("a task run on the platform's key is left to the server, never called deleted", async () => {
  // The platform no longer lists its Firecrawl key: that isn't the user
  // deleting it, so the rerun goes to the server as it was
  const { executeBodies } = setup({
    runKey: PLATFORM_KEY,
    savedCredentials: () => [OTHER_KEY],
  });
  await waitForRunView();

  await rerun();

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0])).toBe(PLATFORM_KEY.id);
  expect(screen.queryByText(/was deleted/)).toBeNull();
  expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();
});

function refusedOverCredentials(message: string) {
  return () =>
    HttpResponse.json(
      {
        detail: {
          type: "validation_error",
          message: "Workflow validation failed: 1 issues on 1 nodes",
          node_errors: { "scrape-step": { credentials: message } },
        },
      },
      { status: 400 },
    );
}

test("when the server refuses the rerun over a step's credentials, the message says what is wrong with them", async () => {
  setup({
    savedCredentials: () => [PLATFORM_KEY, USED_KEY, OTHER_KEY],
    answerRun: refusedOverCredentials(
      "Credentials not available: this API key expired on 1 November 2026",
    ),
  });
  await waitForRunView();

  await rerun();

  expect(await screen.findByText("Failed to start run")).toBeDefined();
  expect(
    await screen.findByText(
      "Credentials not available: this API key expired on 1 November 2026",
      { exact: false },
    ),
  ).toBeDefined();
});

test("a key deleted after the page loaded is named when the server refuses it, and the run form opens to pick another", async () => {
  // Deleted in another tab: this page still lists it until it reads again
  let saved = [PLATFORM_KEY, USED_KEY, OTHER_KEY];
  let refuse = true;
  const { executeBodies } = setup({
    savedCredentials: () => saved,
    answerRun: () =>
      refuse
        ? refusedOverCredentials(`Unknown credentials #${USED_KEY.id}`)()
        : startedRun(),
  });
  await waitForRunView();
  saved = [PLATFORM_KEY, OTHER_KEY];

  await rerun();

  expect(await screen.findByText("Failed to start run")).toBeDefined();
  expect(
    await screen.findByText(
      "A key this task used was deleted: My Firecrawl key.",
    ),
  ).toBeDefined();
  refuse = false;
  const form = await screen.findByRole("dialog", { name: "Rerun task" });
  const picker = await within(form).findByRole("combobox", { name: PICKER });
  // The page read the user's keys again, so the deleted one isn't offered
  await waitFor(() =>
    expect(
      within(picker).queryByRole("option", { name: /My Firecrawl key/ }),
    ).toBeNull(),
  );
  await userEvent.selectOptions(picker, OTHER_KEY.id);
  await clickWhenEnabled("Start Task", form);

  await waitFor(() => expect(executeBodies).toHaveLength(2));
  expect(keyIn(executeBodies[1])).toBe(OTHER_KEY.id);
  expect(executeBodies[1].inputs).toEqual({ topic: "news" });
});
