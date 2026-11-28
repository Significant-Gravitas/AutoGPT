// Rerun's form beyond a single key: workflows with several keys and optional
// ones, review checks, keys added in another tab, and earlier versions with
// their own key fields and safety flags.
import { getGetV1GetSpecificGraphQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
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
import CredentialsProvider, {
  CredentialsProvidersContext,
} from "@/providers/agent-credentials/credentials-provider";
import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { QueryClient, useQuery } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { type ReactNode, useContext } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
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

const RUN_ID = "past-run";
const NEW_RUN_ID = "new-run";
const RUN_VERSION = 1;

const heldAnswers: Array<() => void> = [];

afterEach(() => {
  cleanup();
  heldAnswers.splice(0).forEach((answer) => answer());
  window.localStorage.clear();
});

type Saved = {
  id: string;
  provider: string;
  type: string;
  title: string;
  is_system: boolean;
};

function saved(
  id: string,
  provider: string,
  title: string,
  is_system = false,
): Saved {
  return { id, provider, type: "api_key", title, is_system };
}

const FIRECRAWL_PLATFORM = saved(
  "firecrawl-default",
  "firecrawl",
  "Firecrawl API Key",
  true,
);
const FIRECRAWL_USED = saved(
  "user-firecrawl-key",
  "firecrawl",
  "My Firecrawl key",
);
const FIRECRAWL_OTHER = saved(
  "other-firecrawl-key",
  "firecrawl",
  "Other Firecrawl key",
);
const EXA_PLATFORM = saved(
  "96153e04-9c6c-4486-895f-5bb683b1ecec",
  "exa",
  "Use Credits for Exa",
  true,
);
const EXA_USED = saved("user-exa-key", "exa", "My Exa key");
const EXA_OTHER = saved("other-exa-key", "exa", "Other Exa key");
const GITHUB_ONLY = saved("user-github-key", "github", "My GitHub key");

interface WorkflowField {
  name: string;
  provider: string;
  required?: boolean;
}

const FIRECRAWL_FIELD: WorkflowField = {
  name: "firecrawl_api_key_credentials",
  provider: "firecrawl",
};
const EXA_FIELD: WorkflowField = {
  name: "exa_api_key_credentials",
  provider: "exa",
};
const GITHUB_FIELD: WorkflowField = {
  name: "github_api_key_credentials",
  provider: "github",
  required: false,
};

function fieldSchema({ provider }: WorkflowField) {
  return {
    credentials_provider: [provider],
    credentials_types: ["api_key"],
    title: "Credentials",
    type: "object",
    properties: {
      id: { title: "Id", type: "string" },
      provider: { const: provider, type: "string" },
      type: { const: "api_key", type: "string" },
    },
    required: ["id", "provider", "type"],
  };
}

function credentialsSchema(fields: WorkflowField[]) {
  return {
    type: "object",
    properties: Object.fromEntries(
      fields.map((field) => [field.name, fieldSchema(field)]),
    ),
    required: fields
      .filter(({ required }) => required !== false)
      .map(({ name }) => name),
  };
}

const TOPIC_INPUTS = {
  type: "object",
  properties: { topic: { title: "Topic", type: "string" } },
  required: ["topic"],
};

interface VersionFlags {
  has_sensitive_action?: boolean;
  has_human_in_the_loop?: boolean;
}

function graphAt(
  version: number,
  fields: WorkflowField[],
  flags: VersionFlags = {},
) {
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
    input_schema: TOPIC_INPUTS,
    output_schema: { type: "object", properties: {} },
    credentials_input_schema: credentialsSchema(fields),
    has_external_trigger: false,
    has_human_in_the_loop: flags.has_human_in_the_loop ?? false,
    has_sensitive_action: flags.has_sensitive_action ?? false,
    trigger_setup_info: null,
  };
}

function usedBy(key: Saved) {
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
  // The workflow's credential fields at the version the past run ran
  runFields?: WorkflowField[];
  // ... and at the library's version, when it differs
  libraryFields?: WorkflowField[];
  // The keys the past run used, by field name
  runKeys?: Record<string, Saved>;
  savedCredentials?: () => Saved[];
  providers?: string[];
  systemProviders?: string[];
  libraryVersion?: number;
  runVersionFlags?: VersionFlags;
  libraryFlags?: VersionFlags;
  answerRun?: () => Response | Promise<Response>;
  answerRunVersionRead?: () => Response | Promise<Response>;
  // For a test that makes a query read again
  queryClient?: QueryClient;
  // Rendered beside the page, inside its providers
  beside?: ReactNode;
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
  runFields = [FIRECRAWL_FIELD],
  libraryFields = runFields,
  runKeys = { [FIRECRAWL_FIELD.name]: FIRECRAWL_USED },
  savedCredentials = () => [FIRECRAWL_PLATFORM, FIRECRAWL_OTHER],
  providers = ["firecrawl", "exa", "github"],
  systemProviders = ["firecrawl", "exa"],
  libraryVersion = RUN_VERSION,
  runVersionFlags = {},
  libraryFlags = {},
  answerRun = startedRun,
  answerRunVersionRead = () =>
    HttpResponse.json(graphAt(RUN_VERSION, runFields, runVersionFlags)),
  queryClient,
  beside,
}: Setup = {}) {
  const pastRun: GraphExecutionMeta = {
    id: RUN_ID,
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: RUN_VERSION,
    inputs: { topic: "news" },
    credential_inputs: Object.fromEntries(
      Object.entries(runKeys).map(([field, key]) => [field, usedBy(key)]),
    ),
    nodes_input_masks: {},
    preset_id: null,
    status: AgentExecutionStatus.COMPLETED,
    started_at: new Date("2026-11-20T10:00:00Z"),
    ended_at: new Date("2026-11-20T10:01:00Z"),
    stats: { node_exec_count: 1, node_error_count: 0 },
  };
  const executeBodies: ExecuteBody[] = [];
  const runRequests: string[] = [];
  const graphReads: Array<string | null> = [];
  let credentialReads = 0;

  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: libraryVersion,
        name: "Scrape agent",
        can_access_graph: true,
        input_schema: TOPIC_INPUTS,
        credentials_input_schema: credentialsSchema(libraryFields),
        has_external_trigger: false,
        trigger_setup_info: null,
        has_sensitive_action: libraryFlags.has_sensitive_action ?? false,
        has_human_in_the_loop: libraryFlags.has_human_in_the_loop ?? false,
        settings: {
          human_in_the_loop_safe_mode: true,
          sensitive_action_safe_mode: true,
        },
        recommended_schedule_cron: null,
      }),
    ),
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json(
        providers.map((name) => ({ name, description: `${name} provider` })),
      ),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json(systemProviders),
    ),
    http.get("*/api/integrations/credentials", () => {
      credentialReads += 1;
      return HttpResponse.json(savedCredentials());
    }),
    http.get("*/api/graphs/:graphId", async ({ request }) => {
      const version = new URL(request.url).searchParams.get("version");
      graphReads.push(version);
      return version === String(RUN_VERSION) && libraryVersion !== RUN_VERSION
        ? answerRunVersionRead()
        : HttpResponse.json(
            graphAt(libraryVersion, libraryFields, libraryFlags),
          );
    }),
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
      inputs: pastRun.inputs ?? {},
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
        <SavedKeysLoaded />
        {beside}
        <NuqsTestingAdapter
          hasMemory
          searchParams={`activeTab=runs&activeItem=${RUN_ID}`}
        >
          <NewAgentLibraryView />
        </NuqsTestingAdapter>
      </CredentialsProvider>
      <Toaster />
    </>,
    { queryClient },
  );

  return {
    executeBodies,
    runRequests,
    graphReads,
    credentialReads: () => credentialReads,
  };
}

function SavedKeysLoaded() {
  const providers = useContext(CredentialsProvidersContext);
  return providers ? <span>Saved keys loaded</span> : null;
}

// The read of the workflow at the version the past run ran
const VERSION_READ = getGetV1GetSpecificGraphQueryKey("graph-1", {
  version: RUN_VERSION,
});

// Shows how that read stands. It watches the same query as the page, so an
// answer reaches both in one update: once it shows here, the page has it too.
function VersionReadState() {
  const { status } = useQuery({ queryKey: VERSION_READ, enabled: false });
  return <span>Version read: {status}</span>;
}

async function waitForRunView() {
  expect(await screen.findByText(`Output of ${RUN_ID}`)).toBeDefined();
  expect(await screen.findByText("Saved keys loaded")).toBeDefined();
}

function rerunButton() {
  return screen.getByRole("button", { name: "Rerun task" });
}

async function rerun() {
  await userEvent.click(rerunButton());
}

async function clickWhenEnabled(name: string, container: HTMLElement) {
  const button = await within(container).findByRole("button", { name });
  await waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
  await userEvent.click(button);
}

function picker(form: HTMLElement, provider: string) {
  return within(form).findByRole("combobox", {
    name: `Select ${provider} credential`,
  }) as Promise<HTMLSelectElement>;
}

async function startFromRerunForm() {
  await rerun();
  const form = await screen.findByRole("dialog", { name: "Rerun task" });
  await userEvent.selectOptions(
    await picker(form, "Firecrawl"),
    FIRECRAWL_OTHER.id,
  );
  await clickWhenEnabled("Start Task", form);
}

describe("three bugs", () => {
  test("a key added outside this page after it loaded is not called deleted: the rerun starts on it", async () => {
    // The page reads the user's saved keys once, when the app loads
    let onServer = [FIRECRAWL_PLATFORM, FIRECRAWL_OTHER];
    const { executeBodies, credentialReads } = setup({
      savedCredentials: () => onServer,
    });
    await waitForRunView();
    const readsBefore = credentialReads();
    // Another tab adds the key and runs the task with it: the server holds
    // it, and would start a rerun on it
    onServer = [FIRECRAWL_PLATFORM, FIRECRAWL_USED, FIRECRAWL_OTHER];

    await rerun();

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    // Rerun read the keys again before deciding
    expect(credentialReads()).toBeGreaterThan(readsBefore);
    expect(
      executeBodies[0].credentials_inputs?.[FIRECRAWL_FIELD.name]?.id,
    ).toBe(FIRECRAWL_USED.id);
    expect(screen.queryByText(/was deleted/)).toBeNull();
    expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();
  });

  test("a workflow with review on can be rerun from the form: the safety popup holds the run, and Got it starts it", async () => {
    // A human-in-the-loop workflow whose safety popup this browser hasn't
    // acknowledged: the past task was started somewhere else
    const { runRequests, executeBodies } = setup({
      runVersionFlags: { has_human_in_the_loop: true },
      libraryFlags: { has_human_in_the_loop: true },
    });
    await waitForRunView();

    await startFromRerunForm();

    const popup = await screen.findByRole("dialog", { name: "Safety Checks" });
    expect(runRequests).toHaveLength(0);
    await userEvent.click(
      within(popup).getByRole("button", { name: "Got it" }),
    );
    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(
      executeBodies[0].credentials_inputs?.[FIRECRAWL_FIELD.name]?.id,
    ).toBe(FIRECRAWL_OTHER.id);
  });

  test("a task that ran an earlier version gets that version's safety checks before it starts", async () => {
    // The task's version takes sensitive actions; the current one doesn't
    const { runRequests } = setup({
      libraryVersion: 2,
      runVersionFlags: { has_sensitive_action: true },
      libraryFlags: { has_sensitive_action: false },
    });
    await waitForRunView();

    await startFromRerunForm();

    const popup = await screen.findByRole("dialog", { name: "Safety Checks" });
    expect(runRequests).toHaveLength(0);
    await userEvent.click(
      within(popup).getByRole("button", { name: "Got it" }),
    );
    await waitFor(() => expect(runRequests).toHaveLength(1));
    expect(runRequests[0]).toMatch(/\/execute\/1$/);
  });

  test("an optional key the task ran without stays empty in the rerun form", async () => {
    // The task skipped its optional GitHub step: it ran with no GitHub key,
    // though the user has one
    const { executeBodies } = setup({
      runFields: [FIRECRAWL_FIELD, GITHUB_FIELD],
      runKeys: { [FIRECRAWL_FIELD.name]: FIRECRAWL_USED },
      savedCredentials: () => [
        FIRECRAWL_PLATFORM,
        FIRECRAWL_OTHER,
        GITHUB_ONLY,
      ],
    });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    await userEvent.selectOptions(
      await picker(form, "Firecrawl"),
      FIRECRAWL_OTHER.id,
    );
    const github = await picker(form, "Github");
    expect(github.value).toBe("__none__");
    await clickWhenEnabled("Start Task", form);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(
      executeBodies[0].credentials_inputs?.[GITHUB_FIELD.name],
    ).toBeUndefined();
  });
});

describe("stated properties", () => {
  test("the rerun form keeps the task's other keys, and sends them", async () => {
    // The task ran Firecrawl on a key since deleted, and Exa on the user's
    // own key, though the platform has an Exa key a new form would start on
    const { executeBodies } = setup({
      runFields: [FIRECRAWL_FIELD, EXA_FIELD],
      runKeys: {
        [FIRECRAWL_FIELD.name]: FIRECRAWL_USED,
        [EXA_FIELD.name]: EXA_USED,
      },
      savedCredentials: () => [
        FIRECRAWL_PLATFORM,
        FIRECRAWL_OTHER,
        EXA_PLATFORM,
        EXA_USED,
        EXA_OTHER,
      ],
    });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    expect((await picker(form, "Exa")).value).toBe(EXA_USED.id);
    await userEvent.selectOptions(
      await picker(form, "Firecrawl"),
      FIRECRAWL_OTHER.id,
    );
    await clickWhenEnabled("Start Task", form);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(executeBodies[0].credentials_inputs?.[EXA_FIELD.name]?.id).toBe(
      EXA_USED.id,
    );
    expect(
      executeBodies[0].credentials_inputs?.[FIRECRAWL_FIELD.name]?.id,
    ).toBe(FIRECRAWL_OTHER.id);
  });

  test("a task that ran an earlier version is asked for that version's keys, not the current version's", async () => {
    // The task's version used Firecrawl; the workflow has since moved to Exa
    const { executeBodies, runRequests } = setup({
      libraryVersion: 2,
      runFields: [FIRECRAWL_FIELD],
      libraryFields: [EXA_FIELD],
      savedCredentials: () => [
        FIRECRAWL_PLATFORM,
        FIRECRAWL_OTHER,
        EXA_PLATFORM,
      ],
    });
    await waitForRunView();

    await rerun();

    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    await userEvent.selectOptions(
      await picker(form, "Firecrawl"),
      FIRECRAWL_OTHER.id,
    );
    expect(within(form).queryByText(/Exa credentials/)).toBeNull();
    expect(
      within(form).queryByRole("button", { name: /^System credentials/ }),
    ).toBeNull();
    await clickWhenEnabled("Start Task", form);

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(runRequests[0]).toMatch(/\/execute\/1$/);
    expect(executeBodies[0].credentials_inputs).toEqual({
      [FIRECRAWL_FIELD.name]: expect.objectContaining({
        id: FIRECRAWL_OTHER.id,
      }),
    });
  });

  test("Rerun is busy while the task's version is read, and the form opens once it is", async () => {
    let answer: (() => void) | undefined;
    setup({
      libraryVersion: 2,
      answerRunVersionRead: () =>
        new Promise<Response>((resolve) => {
          answer = () =>
            resolve(HttpResponse.json(graphAt(RUN_VERSION, [FIRECRAWL_FIELD])));
          heldAnswers.push(answer);
        }),
    });
    await waitForRunView();

    await rerun();

    await waitFor(() => expect(answer).toBeDefined());
    expect(rerunButton().hasAttribute("disabled")).toBe(true);
    expect(screen.queryByRole("dialog", { name: "Rerun task" })).toBeNull();

    answer?.();

    expect(
      await screen.findByRole("dialog", { name: "Rerun task" }),
    ).toBeDefined();
  });

  test("an earlier version is read only once Rerun needs it, not when the task is shown", async () => {
    const { graphReads } = setup({ libraryVersion: 2 });
    await waitForRunView();
    // The page's own read of the library's version has landed by now
    await waitFor(() => expect(graphReads).toContain("2"));

    expect(graphReads).not.toContain(String(RUN_VERSION));

    await rerun();
    expect(
      await screen.findByRole("dialog", { name: "Rerun task" }),
    ).toBeDefined();
    expect(graphReads).toContain(String(RUN_VERSION));
  });

  test("a read of the task's version that fails after it loaded keeps the form it built", async () => {
    let canRead = true;
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { runRequests } = setup({
      libraryVersion: 2,
      queryClient,
      beside: <VersionReadState />,
      answerRunVersionRead: () =>
        canRead
          ? HttpResponse.json(graphAt(RUN_VERSION, [FIRECRAWL_FIELD]))
          : HttpResponse.json(
              { detail: "Workflow #graph-1 not found." },
              { status: 404 },
            ),
    });
    await waitForRunView();
    await rerun();
    const form = await screen.findByRole("dialog", { name: "Rerun task" });
    expect(await within(form).findByDisplayValue("news")).toBeDefined();
    expect(screen.getByText("Version read: success")).toBeDefined();

    // The version is read again, as on a return to the tab, and that fails
    canRead = false;
    await act(() => queryClient.refetchQueries({ queryKey: VERSION_READ }));

    // The failure has reached the page: the form is still the only dialog
    expect(await screen.findByText("Version read: error")).toBeDefined();
    expect(screen.queryByText(/couldn't be loaded/)).toBeNull();
    expect(screen.getAllByRole("dialog", { name: "Rerun task" })).toEqual([
      form,
    ]);
    expect(within(form).getByDisplayValue("news")).toBeDefined();
    expect(runRequests).toHaveLength(0);
  });
});
