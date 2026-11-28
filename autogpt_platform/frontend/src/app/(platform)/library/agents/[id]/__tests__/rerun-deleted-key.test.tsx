import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1ListGraphExecutionsMockHandler,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import {
  getGetV1ListExecutionSchedulesForAGraphMockHandler,
  getPostV1CreateExecutionScheduleMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { CredentialsMetaInputType } from "@/app/api/__generated__/models/credentialsMetaInputType";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import CredentialsProvider, {
  CredentialsProvidersContext,
} from "@/providers/agent-credentials/credentials-provider";
import {
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { useContext } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
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
// The workflow version the past run ran
const RUN_VERSION = 1;
// A later run, of the library's version
const LATER_RUN_ID = "later-run";

// Run requests a test holds back. Teardown unmounts the page, then answers
// whatever a failed test left held, so none outlives its test.
const heldAnswers: Array<() => void> = [];

afterEach(() => {
  cleanup();
  heldAnswers.splice(0).forEach((answer) => answer());
});

type Key = { id: string; provider: string; title: string };

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
// A key saved in another tab after the page loaded
const NEW_KEY = {
  id: "new-firecrawl-key",
  provider: "firecrawl",
  type: "api_key",
  title: "New Firecrawl key",
  is_system: false,
};

interface WorkflowField {
  name: string;
  schema: Record<string, unknown>;
  // The providers it takes keys from, each with a platform key
  providers: string[];
}

const FIRECRAWL_FIELD: WorkflowField = {
  name: FIELD,
  providers: ["firecrawl"],
  schema: {
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
  },
};

// The inputs of the version the past run ran
const RUN_VERSION_INPUTS = {
  type: "object",
  properties: { topic: { title: "Topic", type: "string" } },
  required: ["topic"],
};
// A later version asks for something else
const LATER_VERSION_INPUTS = {
  type: "object",
  properties: { query: { title: "Search query", type: "string" } },
  required: ["query"],
};

function credentialsSchema(field: WorkflowField) {
  return {
    type: "object",
    properties: { [field.name]: field.schema },
    required: [field.name],
  };
}

function inputsAt(version: number) {
  return version === RUN_VERSION ? RUN_VERSION_INPUTS : LATER_VERSION_INPUTS;
}

// The workflow at a version, as GET /graphs/{id}?version= answers
function graphAt(version: number, field: WorkflowField) {
  return {
    id: "graph-1",
    version,
    name: "Scrape agent",
    description: "",
    user_id: "user-1",
    created_at: "2026-11-01T10:00:00Z",
    nodes: [],
    links: [],
    sub_graphs: [],
    input_schema: inputsAt(version),
    output_schema: { type: "object", properties: {} },
    credentials_input_schema: credentialsSchema(field),
    has_external_trigger: false,
    has_human_in_the_loop: false,
    has_sensitive_action: false,
    trigger_setup_info: null,
  };
}

// What a run's credential_inputs hold for a key
function usedBy(key: Key) {
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

type ScheduleBody = {
  name?: string;
  cron?: string;
  graph_version?: number | null;
  inputs?: Record<string, unknown>;
  credentials?: Record<string, { id?: string }>;
};

interface Setup {
  // The key the past run used
  runKey?: Key;
  // The keys /integrations/credentials lists, read again on every request
  savedCredentials?: () =>
    | Array<Record<string, unknown>>
    | Promise<Array<Record<string, unknown>>>
    | Response;
  // How the server answers a run request
  answerRun?: () => Response | Promise<Response>;
  // The workflow's credential field
  field?: WorkflowField;
  // The workflow's version in the library
  libraryVersion?: number;
  // How the server answers a read of the version the past run ran
  answerRunVersionRead?: () => Response;
  // Lists a later run of the library's version above the past run
  withLaterRun?: boolean;
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
  field = FIRECRAWL_FIELD,
  libraryVersion = RUN_VERSION,
  answerRunVersionRead = () => HttpResponse.json(graphAt(RUN_VERSION, field)),
  withLaterRun = false,
}: Setup = {}) {
  const pastRun: GraphExecutionMeta = {
    id: RUN_ID,
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: RUN_VERSION,
    inputs: RUN_INPUTS,
    credential_inputs: { [field.name]: usedBy(runKey) },
    nodes_input_masks: {},
    preset_id: null,
    status: AgentExecutionStatus.COMPLETED,
    started_at: new Date("2026-11-20T10:00:00Z"),
    ended_at: new Date("2026-11-20T10:01:00Z"),
    stats: { node_exec_count: 1, node_error_count: 0 },
  };
  const laterRun: GraphExecutionMeta = {
    ...pastRun,
    id: LATER_RUN_ID,
    graph_version: libraryVersion,
    inputs: { query: "weather" },
    credential_inputs: { [field.name]: usedBy(OTHER_KEY) },
    started_at: new Date("2026-11-21T10:00:00Z"),
    ended_at: new Date("2026-11-21T10:01:00Z"),
  };
  const runs = withLaterRun ? [laterRun, pastRun] : [pastRun];
  const executeBodies: ExecuteBody[] = [];
  // Recorded as each run request arrives, before its body is read
  const runRequests: string[] = [];
  // The version each read of the workflow asked for
  const graphReads: Array<string | null> = [];
  const scheduleBodies: ScheduleBody[] = [];
  // The workflow's schedules, as the list answers once one is made
  const schedules: GraphExecutionJobInfo[] = [];

  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: libraryVersion,
        name: "Scrape agent",
        can_access_graph: true,
        input_schema: inputsAt(libraryVersion),
        credentials_input_schema: credentialsSchema(field),
        has_external_trigger: false,
        trigger_setup_info: null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
        recommended_schedule_cron: null,
      }),
    ),
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json(
        field.providers.map((name) => ({
          name,
          description: `${name} provider`,
        })),
      ),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json(field.providers),
    ),
    http.get("*/api/integrations/credentials", async () => {
      const answer = await savedCredentials();
      return answer instanceof Response ? answer : HttpResponse.json(answer);
    }),
    http.get("*/api/graphs/:graphId", ({ request }) => {
      const version = new URL(request.url).searchParams.get("version");
      graphReads.push(version);
      return version === String(RUN_VERSION)
        ? answerRunVersionRead()
        : HttpResponse.json(graphAt(libraryVersion, field));
    }),
    getGetV1ListGraphExecutionsMockHandler({
      executions: runs,
      pagination: {
        total_items: runs.length,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1GetExecutionDetailsMockHandler(({ params }) => {
      const run = runs.find(({ id }) => id === params.graphExecId) ?? pastRun;
      return {
        ...run,
        id: String(params.graphExecId),
        inputs: run.inputs ?? {},
        outputs: { result: [`Output of ${params.graphExecId}`] },
        node_executions: [],
      };
    }),
    getGetV1ListExecutionSchedulesForAGraphMockHandler(() => schedules),
    getPostV1CreateExecutionScheduleMockHandler(async ({ request }) => {
      const body = (await request.json()) as ScheduleBody;
      scheduleBodies.push(body);
      const schedule = getPostV1CreateExecutionScheduleResponseMock({
        id: "new-schedule",
        name: body.name || "Rerun schedule",
        cron: body.cron || "0 9 * * 1",
        graph_id: "graph-1",
        graph_version: body.graph_version ?? libraryVersion,
      });
      schedules.push(schedule);
      return schedule;
    }),
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
        <SavedKeysLoaded />
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

  return { executeBodies, runRequests, graphReads, scheduleBodies };
}

// Shows once the user's saved keys, which a rerun checks, have loaded. The
// run's "Your input" isn't a sign of it: it leaves out a key whose title
// reads like the platform's.
function SavedKeysLoaded() {
  const providers = useContext(CredentialsProvidersContext);
  return providers ? <span>Saved keys loaded</span> : null;
}

async function waitForRunView() {
  expect(await screen.findByText(`Output of ${RUN_ID}`)).toBeDefined();
  expect(await screen.findByText("Saved keys loaded")).toBeDefined();
}

async function rerun() {
  await userEvent.click(screen.getByRole("button", { name: "Rerun task" }));
}

async function showTask(runId: string) {
  await userEvent.click(screen.getByTestId(`library-run-item-${runId}`));
  expect(await screen.findByText(`Output of ${runId}`)).toBeDefined();
}

function keyIn(body: ExecuteBody | undefined, field = FIELD) {
  return body?.credentials_inputs?.[field]?.id;
}

async function clickWhenEnabled(name: string, container: HTMLElement) {
  const button = await within(container).findByRole("button", { name });
  await waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
  await userEvent.click(button);
}

function isDisabled(name: string, container: HTMLElement) {
  return within(container)
    .getByRole("button", { name })
    .hasAttribute("disabled");
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
  expect(await within(form).findByDisplayValue("news")).toBeDefined();
  // The field is shown, outside the collapsed system credentials, and left
  // for the user to pick: not moved to the platform's key
  const picker = await within(form).findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe("__none__");
  expect(
    within(form).queryByRole("button", { name: SYSTEM_SECTION }),
  ).toBeNull();
  expect(isDisabled("Start Task", form)).toBe(true);
  expect(runRequests).toHaveLength(0);
  expect(screen.queryByText("Failed to start run")).toBeNull();
});

test("closing the rerun form sends nothing, and Rerun opens it again without the earlier pick", async () => {
  const { runRequests } = setup();
  await waitForRunView();

  await rerun();
  const form = await screen.findByRole("dialog", { name: "Rerun task" });
  await userEvent.selectOptions(
    await within(form).findByRole("combobox", { name: PICKER }),
    OTHER_KEY.id,
  );
  await userEvent.keyboard("{Escape}");
  await waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull(),
  );

  await rerun();
  const again = await screen.findByRole("dialog", { name: "Rerun task" });
  const picker = await within(again).findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe("__none__");
  expect(await within(again).findByDisplayValue("news")).toBeDefined();
  expect(runRequests).toHaveLength(0);
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
      within(picker).queryByRole("option", { name: /^My Firecrawl key \(/ }),
    ).toBeNull(),
  );
  await userEvent.selectOptions(picker, OTHER_KEY.id);
  await clickWhenEnabled("Start Task", form);

  await waitFor(() => expect(executeBodies).toHaveLength(2));
  expect(keyIn(executeBodies[1])).toBe(OTHER_KEY.id);
  expect(executeBodies[1].inputs).toEqual({ topic: "news" });
});

test("a Rerun pressed while the saved keys load opens the form when the server refuses a deleted key", async () => {
  // The page's first read of the user's keys is held until the test answers
  let answerKeys: (() => void) | undefined;
  const keysRead = new Promise<void>((resolve) => {
    answerKeys = resolve;
    heldAnswers.push(resolve);
  });
  let refuse: (() => void) | undefined;
  const { runRequests } = setup({
    savedCredentials: async () => {
      await keysRead;
      return [PLATFORM_KEY, OTHER_KEY];
    },
    answerRun: () =>
      new Promise<Response>((resolve) => {
        refuse = () =>
          resolve(
            refusedOverCredentials(`Unknown credentials #${USED_KEY.id}`)(),
          );
        heldAnswers.push(refuse);
      }),
  });
  expect(await screen.findByText(`Output of ${RUN_ID}`)).toBeDefined();
  expect(screen.queryByText("Saved keys loaded")).toBeNull();

  // Nothing is known to be deleted yet, so the rerun is sent
  await rerun();
  await waitFor(() => expect(runRequests).toHaveLength(1));
  answerKeys?.();
  expect(await screen.findByText("Saved keys loaded")).toBeDefined();
  refuse?.();

  // The refusal is read against the keys as they are now, not as they were
  // when Rerun was pressed
  expect(
    await screen.findByText(
      "A key this task used was deleted: My Firecrawl key.",
    ),
  ).toBeDefined();
  expect(
    await screen.findByRole("dialog", { name: "Rerun task" }),
  ).toBeDefined();
});

test("when the keys can't be read again, the rerun goes to the server on the task's key", async () => {
  // The page's list doesn't have the task's key; the read that would confirm
  // it is gone fails
  let readsFail = false;
  const { executeBodies } = setup({
    savedCredentials: () =>
      readsFail
        ? HttpResponse.json({ detail: "Unavailable" }, { status: 500 })
        : [PLATFORM_KEY, OTHER_KEY],
  });
  await waitForRunView();
  readsFail = true;

  await rerun();

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0])).toBe(USED_KEY.id);
  expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();
});

test("the form offers a key saved in another tab since the page loaded", async () => {
  let saved = [PLATFORM_KEY, OTHER_KEY];
  setup({ savedCredentials: () => saved });
  await waitForRunView();
  saved = [PLATFORM_KEY, OTHER_KEY, NEW_KEY];

  await rerun();

  const form = await screen.findByRole("dialog", { name: "Rerun task" });
  const picker = await within(form).findByRole("combobox", { name: PICKER });
  // The read that confirmed the task's key is gone also brought the page's
  // keys up to date
  expect(
    await within(picker).findByRole("option", {
      name: /^New Firecrawl key \(/,
    }),
  ).toBeDefined();
});

describe("which keys are the platform's is decided by their ids", () => {
  test("a deleted key of the user's named like the platform's is still called deleted", async () => {
    // The user names their keys, so a title can read like a platform key's
    const { runRequests } = setup({
      runKey: { ...USED_KEY, title: "Production system key" },
    });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    expect(
      within(form).getByText(
        "A key this task used was deleted: Production system key. Choose another key below to run it again.",
      ),
    ).toBeDefined();
    expect(runRequests).toHaveLength(0);
  });

  test("a platform key the server refuses is never called deleted, whatever its title", async () => {
    // One of the platform's built-in keys: the saved keys mark it as the
    // platform's, though nothing in its id or title says so
    const builtInKey = {
      id: "6b9fc200-4726-4973-86c9-cd526f5ce5db",
      provider: "firecrawl",
      type: "api_key",
      title: "Firecrawl API Key",
      is_system: true,
    };
    setup({
      runKey: builtInKey,
      savedCredentials: () => [builtInKey, OTHER_KEY],
      answerRun: refusedOverCredentials(
        `Unknown credentials #${builtInKey.id}`,
      ),
    });
    await waitForRunView();

    await rerun();

    expect(await screen.findByText("Failed to start run")).toBeDefined();
    expect(
      await screen.findByText(`Unknown credentials #${builtInKey.id}`, {
        exact: false,
      }),
    ).toBeDefined();
    expect(screen.queryByText(/was deleted/)).toBeNull();
    expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();
  });
});

describe("a field left to pick waits for the user's pick", () => {
  test("when only the platform's key is left, the form offers it and starts nothing until the user picks it", async () => {
    // The user deleted their only Firecrawl key
    const { executeBodies } = setup({ savedCredentials: () => [PLATFORM_KEY] });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    const picker = await within(form).findByRole("combobox", { name: PICKER });
    expect((picker as HTMLSelectElement).value).toBe("__none__");
    expect(
      within(picker).getByRole("option", { name: /^Firecrawl API Key \(/ }),
    ).toBeDefined();
    expect(isDisabled("Start Task", form)).toBe(true);

    await userEvent.selectOptions(picker, PLATFORM_KEY.id);
    await clickWhenEnabled("Start Task", form);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(keyIn(executeBodies[0])).toBe(PLATFORM_KEY.id);
  });

  test("a field open to several providers offers the platform's key of the provider the task used", async () => {
    // An LLM step whose model comes from a link takes OpenAI or Anthropic keys
    const llmField: WorkflowField = {
      name: "llm_credentials",
      providers: ["openai", "anthropic"],
      schema: {
        credentials_provider: ["openai", "anthropic"],
        credentials_types: ["api_key"],
        discriminator: "model",
        discriminator_mapping: {
          "gpt-4o": "openai",
          "claude-sonnet-4": "anthropic",
        },
        title: "Credentials",
        type: "object",
        properties: {
          id: { title: "Id", type: "string" },
          provider: { enum: ["openai", "anthropic"], type: "string" },
          type: { const: "api_key", type: "string" },
        },
        required: ["id", "provider", "type"],
      },
    };
    const openAIPlatformKey = {
      id: "53c25cb8-e3ee-465c-a4d1-e75a4c899c2a",
      provider: "openai",
      type: "api_key",
      title: "Use Credits for OpenAI",
      is_system: true,
    };
    const anthropicPlatformKey = {
      id: "24e5d942-d9e3-4798-8151-90143ee55629",
      provider: "anthropic",
      type: "api_key",
      title: "Use Credits for Anthropic",
      is_system: true,
    };
    // The user's last OpenAI key, since deleted
    const { executeBodies } = setup({
      field: llmField,
      runKey: {
        id: "user-openai-key",
        provider: "openai",
        title: "My OpenAI key",
      },
      savedCredentials: () => [openAIPlatformKey, anthropicPlatformKey],
    });
    await waitForRunView();
    // The run's "Your input" shows its key under the provider it was for
    expect(await screen.findByText("Openai credentials")).toBeDefined();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    const picker = await within(form).findByRole("combobox", {
      name: "Select Openai credential",
    });
    expect((picker as HTMLSelectElement).value).toBe("__none__");
    expect(isDisabled("Start Task", form)).toBe(true);

    await userEvent.selectOptions(picker, openAIPlatformKey.id);
    await clickWhenEnabled("Start Task", form);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(keyIn(executeBodies[0], llmField.name)).toBe(openAIPlatformKey.id);
  });

  test("a field whose steps refuse the platform's keys offers only the user's own, and still waits for their pick", async () => {
    // An Exa Websets step keeps what it creates in the user's Exa account
    const websetsField: WorkflowField = {
      name: "exa_credentials",
      providers: ["exa"],
      schema: {
        credentials_provider: ["exa"],
        credentials_types: ["api_key"],
        refuses_platform_credentials: true,
        title: "Credentials",
        type: "object",
        properties: {
          id: { title: "Id", type: "string" },
          provider: { const: "exa", type: "string" },
          type: { const: "api_key", type: "string" },
        },
        required: ["id", "provider", "type"],
      },
    };
    const exaPlatformKey = {
      id: "exa-default",
      provider: "exa",
      type: "api_key",
      title: "Exa API Key",
      is_system: true,
    };
    const otherExaKey = {
      id: "other-exa-key",
      provider: "exa",
      type: "api_key",
      title: "Other Exa key",
      is_system: false,
    };
    // The user's Exa key the task used, since deleted
    const { executeBodies } = setup({
      field: websetsField,
      runKey: { id: "user-exa-key", provider: "exa", title: "My Exa key" },
      savedCredentials: () => [exaPlatformKey, otherExaKey],
    });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    const picker = await within(form).findByRole("combobox", {
      name: "Select Exa credential",
    });
    // The empty choice and the user's own key: not the platform's
    expect(
      Array.from((picker as HTMLSelectElement).options, ({ value }) => value),
    ).toEqual(["__none__", otherExaKey.id]);
    expect((picker as HTMLSelectElement).value).toBe("__none__");
    expect(isDisabled("Start Task", form)).toBe(true);

    await userEvent.selectOptions(picker, otherExaKey.id);
    await clickWhenEnabled("Start Task", form);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(keyIn(executeBodies[0], websetsField.name)).toBe(otherExaKey.id);
  });
});

describe("the run form reruns the workflow version the task ran", () => {
  test("a task that ran an earlier version reruns that version, with that version's inputs", async () => {
    const { executeBodies, runRequests, graphReads } = setup({
      libraryVersion: 2,
    });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    // Built from the task's version: its Topic input holds the task's
    // input, and the later version's search query isn't asked for
    expect(await within(form).findByDisplayValue("news")).toBeDefined();
    expect(within(form).queryByText("Search query")).toBeNull();
    await userEvent.selectOptions(
      await within(form).findByRole("combobox", { name: PICKER }),
      OTHER_KEY.id,
    );
    await clickWhenEnabled("Start Task", form);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(runRequests[0]).toMatch(/\/graphs\/graph-1\/execute\/1$/);
    expect(executeBodies[0].inputs).toEqual({ topic: "news" });
    expect(keyIn(executeBodies[0])).toBe(OTHER_KEY.id);
    expect(graphReads).toContain(String(RUN_VERSION));
  });

  test("when the task's version can't be loaded, the form says so and runs nothing until it loads", async () => {
    let canRead = false;
    const { executeBodies, runRequests } = setup({
      libraryVersion: 2,
      answerRunVersionRead: () =>
        canRead
          ? HttpResponse.json(graphAt(RUN_VERSION, FIRECRAWL_FIELD))
          : HttpResponse.json(
              { detail: "Workflow #graph-1 not found." },
              { status: 404 },
            ),
    });
    await waitForRunView();

    await rerun();

    const notice = await screen.findByRole("dialog", { name: "Rerun task" });
    expect(
      await within(notice).findByText(
        "This task ran version 1 of the workflow, which couldn't be loaded. Try again, or run the current version with New workflow task.",
      ),
    ).toBeDefined();
    expect(
      within(notice).getByText(
        "A key this task used was deleted: My Firecrawl key.",
      ),
    ).toBeDefined();
    // Nothing runs, and the current version's form isn't offered instead
    expect(
      within(notice).queryByRole("button", { name: "Start Task" }),
    ).toBeNull();
    expect(within(notice).queryByText("Search query")).toBeNull();
    expect(runRequests).toHaveLength(0);

    canRead = true;
    await userEvent.click(
      within(notice).getByRole("button", { name: "Try again" }),
    );

    // The form opens in its place, built from the task's version
    expect(await screen.findByDisplayValue("news")).toBeDefined();
    const loaded = screen.getByRole("dialog", { name: "Rerun task" });
    await userEvent.selectOptions(
      await within(loaded).findByRole("combobox", { name: PICKER }),
      OTHER_KEY.id,
    );
    await clickWhenEnabled("Start Task", loaded);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(runRequests[0]).toMatch(/\/execute\/1$/);
  });

  test("scheduling from the rerun form schedules the version the task ran, and shows the new schedule", async () => {
    const { scheduleBodies, runRequests } = setup({ libraryVersion: 2 });
    await waitForRunView();

    await rerun();
    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    expect(await within(form).findByDisplayValue("news")).toBeDefined();
    await userEvent.selectOptions(
      await within(form).findByRole("combobox", { name: PICKER }),
      OTHER_KEY.id,
    );
    await clickWhenEnabled("Schedule Task", form);
    await clickWhenEnabled("Schedule", document.body);

    await waitFor(() => expect(scheduleBodies).toHaveLength(1));
    expect(scheduleBodies[0].graph_version).toBe(RUN_VERSION);
    expect(scheduleBodies[0].inputs).toEqual({ topic: "news" });
    expect(keyIn({ credentials_inputs: scheduleBodies[0].credentials })).toBe(
      OTHER_KEY.id,
    );
    expect(runRequests).toHaveLength(0);
    // The page moves to the new schedule, as Start Task moves to the new run
    expect(await screen.findByText("Recurrence")).toBeDefined();
    expect(
      screen.getAllByText(scheduleBodies[0].name || "Rerun schedule").length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText(`Output of ${RUN_ID}`)).toBeNull();
  });

  test("a task of the workflow's current version opens its form without reading the workflow", async () => {
    // No key of the user's own is left, so nothing else on the page reads
    // the workflow either
    const { graphReads } = setup({ savedCredentials: () => [PLATFORM_KEY] });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    expect(await within(form).findByDisplayValue("news")).toBeDefined();
    expect(graphReads).toEqual([]);
  });
});

test("a refusal that comes back after the user moved to another task opens no form, there or back on the first task", async () => {
  // Deleted in another tab: this page still lists it until it reads again
  let saved = [PLATFORM_KEY, USED_KEY, OTHER_KEY];
  let refuse: (() => void) | undefined;
  const { runRequests, graphReads } = setup({
    libraryVersion: 2,
    withLaterRun: true,
    savedCredentials: () => saved,
    answerRun: () =>
      new Promise<Response>((resolve) => {
        refuse = () =>
          resolve(
            refusedOverCredentials(`Unknown credentials #${USED_KEY.id}`)(),
          );
        heldAnswers.push(refuse);
      }),
  });
  await waitForRunView();
  // Opened once already, so going back to it shows it at once, with the
  // same task actions
  await showTask(LATER_RUN_ID);
  await showTask(RUN_ID);
  saved = [PLATFORM_KEY, OTHER_KEY];

  await rerun();
  await waitFor(() => expect(runRequests).toHaveLength(1));
  await showTask(LATER_RUN_ID);
  refuse?.();

  // The refusal is still told. Its toast renders in the same pass a stale form
  // would open in, with the first task's inputs on the later task's version
  expect(await screen.findByText("Failed to start run")).toBeDefined();
  expect(
    screen.getByText("A key this task used was deleted: My Firecrawl key."),
  ).toBeDefined();
  expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();

  await showTask(RUN_ID);
  expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();

  // Rerun there opens the first task's form, on the version it ran
  await rerun();
  const form = await screen.findByRole("dialog", { name: "Rerun task" });
  expect(await within(form).findByDisplayValue("news")).toBeDefined();
  expect(within(form).queryByText("Search query")).toBeNull();
  expect(graphReads).toContain(String(RUN_VERSION));
  expect(runRequests).toHaveLength(1);
});
