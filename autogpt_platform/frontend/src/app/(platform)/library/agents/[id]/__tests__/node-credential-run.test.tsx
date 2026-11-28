import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1GetSpecificGraphMockHandler,
  getGetV1GetSpecificGraphMockHandler422,
  getGetV1GetSpecificGraphResponseMock,
  getGetV1ListGraphExecutionsMockHandler,
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2GetASpecificPresetMockHandler,
  getGetV2ListPresetsMockHandler,
  getPostV2SetupTriggerMockHandler,
  getPostV2SetupTriggerResponseMock,
} from "@/app/api/__generated__/endpoints/presets/presets.msw";
import {
  getGetV1ListExecutionSchedulesForAGraphMockHandler,
  getPostV1CreateExecutionScheduleMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphTriggerInfo } from "@/app/api/__generated__/models/graphTriggerInfo";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
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
const LOOKUP_FAILED =
  /^Couldn't load the credentials this workflow's steps use/;

// The platform's own Firecrawl key, as /integrations/credentials lists it.
const PLATFORM_KEY = {
  id: "firecrawl-default",
  provider: "firecrawl",
  type: "api_key",
  title: "Firecrawl API Key",
  is_system: true,
};
const USER_KEY = {
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

// An LLM step whose model comes from a link: its field lists every LLM
// provider, since which one the run needs isn't known before it runs.
const LLM_FIELD = "anthropic-openai_api_key_credentials";
const LLM_PICKER = "Select Openai credential";
const LLM_FIELD_SCHEMA = {
  credentials_provider: ["openai", "anthropic"],
  credentials_types: ["api_key"],
  discriminator: "model",
  discriminator_mapping: { "gpt-4o": "openai", "claude-sonnet-4": "anthropic" },
  title: "Credentials",
  type: "object",
  properties: {
    id: { title: "Id", type: "string" },
    provider: { enum: ["openai", "anthropic"], type: "string" },
    type: { const: "api_key", type: "string" },
  },
  required: ["id", "provider", "type"],
};
const OPENAI_PLATFORM_KEY = {
  id: "openai-default",
  provider: "openai",
  type: "api_key",
  title: "OpenAI API Key",
  is_system: true,
};
const ANTHROPIC_PLATFORM_KEY = {
  id: "anthropic-default",
  provider: "anthropic",
  type: "api_key",
  title: "Anthropic API Key",
  is_system: true,
};
const USER_OPENAI_KEY = {
  id: "user-openai-key",
  provider: "openai",
  type: "api_key",
  title: "My OpenAI key",
  is_system: false,
};

// A second field, for Exa, with a platform key and a key of the user's own
const EXA_FIELD = "exa_api_key_credentials";
const EXA_PICKER = "Select Exa credential";
const EXA_FIELD_SCHEMA = {
  ...FIRECRAWL_FIELD,
  credentials_provider: ["exa"],
  properties: {
    ...FIRECRAWL_FIELD.properties,
    provider: { const: "exa", type: "string" },
  },
};
const EXA_PLATFORM_KEY = {
  id: "exa-default",
  provider: "exa",
  type: "api_key",
  title: "Exa API Key",
  is_system: true,
};
const USER_EXA_KEY = {
  id: "user-exa-key",
  provider: "exa",
  type: "api_key",
  title: "My Exa key",
  is_system: false,
};

const USE_FORM_KEYS = "Use the keys shown in the run form";

// What a step's `input_default.credentials` holds once a key is picked on it.
function savedOnStep(key: { id: string; title: string; provider?: string }) {
  return {
    id: key.id,
    provider: key.provider ?? "firecrawl",
    type: "api_key",
    title: key.title,
  };
}

function scrapeStep(
  id: string,
  graphID: string,
  credentials?: Record<string, unknown>,
) {
  return {
    id,
    block_id: "firecrawl-scrape",
    graph_id: graphID,
    graph_version: 1,
    input_default: credentials
      ? { url: "https://example.com", credentials }
      : { url: "https://example.com" },
    metadata: {},
    input_links: [],
    output_links: [],
  };
}

interface Workflow {
  stepCredentials?: Record<string, unknown>;
  // Keys saved on more of the workflow's steps, and on a sub-workflow's steps
  otherStepCredentials?: Array<Record<string, unknown>>;
  subWorkflowStepCredentials?: Array<Record<string, unknown>>;
  // Holds the graph's answer back until it resolves
  graphReady?: Promise<void>;
  savedCredentials?: Array<Record<string, unknown>>;
  providers?: string[];
  systemProviders?: string[];
  // Whether the user owns the workflow; only an owner can read its steps
  canAccessGraph?: boolean;
  // The run form's credential fields, and which of them are required
  credentialFields?: Record<string, unknown>;
  requiredFields?: string[];
  // A template whose credentials the run form starts from
  templateCredentials?: LibraryAgentPreset["credentials"];
  // The workflow starts from a trigger, so the run form sets one up
  triggerSetupInfo?: GraphTriggerInfo;
}

const TEMPLATE_ID = "template-1";

function mockWorkflow({
  stepCredentials,
  otherStepCredentials = [],
  subWorkflowStepCredentials = [],
  graphReady,
  savedCredentials = [PLATFORM_KEY, USER_KEY],
  providers = ["firecrawl"],
  systemProviders = ["firecrawl"],
  canAccessGraph = true,
  credentialFields = { [FIELD]: FIRECRAWL_FIELD },
  requiredFields = [FIELD],
  templateCredentials,
  triggerSetupInfo,
}: Workflow) {
  const { sub_graphs: _, ...subWorkflow } =
    getGetV1GetSpecificGraphResponseMock({
      id: "sub-graph-1",
      version: 1,
      links: [],
      sub_graphs: [],
      nodes: subWorkflowStepCredentials.map((credentials, index) =>
        scrapeStep(`sub-step-${index}`, "sub-graph-1", credentials),
      ),
    });
  const graph = getGetV1GetSpecificGraphResponseMock({
    id: "graph-1",
    version: 1,
    links: [],
    sub_graphs: subWorkflowStepCredentials.length > 0 ? [subWorkflow] : [],
    nodes: [
      scrapeStep("scrape-step", "graph-1", stepCredentials),
      ...otherStepCredentials.map((credentials, index) =>
        scrapeStep(`other-step-${index}`, "graph-1", credentials),
      ),
    ],
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
  const executeBodies: Array<{ credentials_inputs?: unknown }> = [];
  const scheduleBodies: Array<{ credentials?: unknown }> = [];
  const triggerBodies: Array<{ agent_credentials?: unknown }> = [];
  const graphReads: string[] = [];
  const templates: LibraryAgentPreset[] = templateCredentials
    ? [
        {
          id: TEMPLATE_ID,
          name: "Weekly scrape",
          description: "",
          graph_id: "graph-1",
          graph_version: 1,
          inputs: {},
          credentials: templateCredentials,
          is_active: true,
          user_id: "user-1",
          created_at: new Date(),
          updated_at: new Date(),
          webhook_id: null,
          webhook: null,
        },
      ]
    : [];
  function answerGraphReads() {
    server.use(
      getGetV1GetSpecificGraphMockHandler(async ({ request }) => {
        graphReads.push(request.url);
        await graphReady;
        return graph;
      }),
    );
  }

  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: 1,
        name: "Scrape agent",
        can_access_graph: canAccessGraph,
        input_schema: { type: "object", properties: {} },
        credentials_input_schema: {
          type: "object",
          properties: credentialFields,
          required: requiredFields,
        },
        has_external_trigger: false,
        trigger_setup_info: triggerSetupInfo ?? null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
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
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json(savedCredentials),
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
      presets: templates,
      pagination: {
        total_items: templates.length,
        total_pages: templates.length > 0 ? 1 : 0,
        current_page: 1,
        page_size: 100,
      },
    }),
    ...templates.map((template) =>
      getGetV2GetASpecificPresetMockHandler(template),
    ),
    getPostV1ExecuteGraphAgentMockHandler(async ({ request }) => {
      executeBodies.push((await request.json()) as never);
      return run;
    }),
    getGetV1GetExecutionDetailsMockHandler({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run,
      inputs: {},
      outputs: {},
      node_executions: [],
    }),
    getPostV1CreateExecutionScheduleMockHandler(async ({ request }) => {
      scheduleBodies.push((await request.json()) as never);
      return getPostV1CreateExecutionScheduleResponseMock({
        id: "new-schedule",
        graph_id: "graph-1",
      });
    }),
    getPostV2SetupTriggerMockHandler(async ({ request }) => {
      triggerBodies.push((await request.json()) as never);
      return getPostV2SetupTriggerResponseMock({
        id: "new-trigger",
        graph_id: "graph-1",
      });
    }),
  );
  answerGraphReads();

  return {
    executeBodies,
    scheduleBodies,
    triggerBodies,
    graphReads,
    answerGraphReads,
  };
}

// Every read of the graph fails, until answerGraphReads() is called again
function failGraphReads() {
  server.use(getGetV1GetSpecificGraphMockHandler422());
}

function renderAgentPage(searchParams?: string) {
  render(
    <CredentialsProvider>
      <NuqsTestingAdapter hasMemory searchParams={searchParams}>
        <NewAgentLibraryView />
      </NuqsTestingAdapter>
    </CredentialsProvider>,
  );
}

// An LLM step set to the user's own OpenAI key, with both providers' platform keys
function mockLLMWorkflow(workflow: Workflow = {}) {
  return mockWorkflow({
    stepCredentials: savedOnStep(USER_OPENAI_KEY),
    credentialFields: { [LLM_FIELD]: LLM_FIELD_SCHEMA },
    requiredFields: [LLM_FIELD],
    providers: ["openai", "anthropic"],
    systemProviders: ["openai", "anthropic"],
    savedCredentials: [
      OPENAI_PLATFORM_KEY,
      ANTHROPIC_PLATFORM_KEY,
      USER_OPENAI_KEY,
    ],
    ...workflow,
  });
}

async function openRunForm() {
  await userEvent.click(
    await screen.findByRole("button", { name: "Setup your task" }),
  );
}

async function clickWhenEnabled(name: string) {
  const button = await screen.findByRole("button", { name });
  await waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
  await userEvent.click(button);
}

function keyIn(credentials: unknown, field = FIELD) {
  return (credentials as Record<string, { id?: string }> | undefined)?.[field]
    ?.id;
}

function pickerValue(name = PICKER) {
  return (screen.getByRole("combobox", { name }) as HTMLSelectElement).value;
}

// Waits for the form to settle on a key; the picker can move as it does
async function expectPickerOn(key: { id: string }, name = PICKER) {
  await waitFor(() => expect(pickerValue(name)).toBe(key.id));
}

test("a run uses the key the user saved on the step, not the platform key", async () => {
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
  });
  renderAgentPage();

  await openRunForm();
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("the run form shows the step's key, outside the collapsed system credentials", async () => {
  mockWorkflow({ stepCredentials: savedOnStep(USER_KEY) });
  renderAgentPage();

  await openRunForm();

  const picker = await screen.findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id);
  expect(screen.queryByRole("button", { name: SYSTEM_SECTION })).toBeNull();
});

test("a run form opened before the step's keys load still starts on the step's key", async () => {
  let releaseGraph = () => {};
  const graphReady = new Promise<void>((resolve) => {
    releaseGraph = resolve;
  });
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    graphReady,
  });
  renderAgentPage();

  try {
    await openRunForm();
    expect(
      await screen.findByRole("status", { name: "Loading credentials" }),
    ).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "Start Task" })
        .hasAttribute("disabled"),
    ).toBe(true);
  } finally {
    releaseGraph();
  }

  const picker = await screen.findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id);
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("the user can switch the step's key to the platform key, and the field stays in view", async () => {
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
  });
  renderAgentPage();

  await openRunForm();
  await userEvent.selectOptions(
    await screen.findByRole("combobox", { name: PICKER }),
    PLATFORM_KEY.id,
  );

  const picker = await screen.findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe(PLATFORM_KEY.id);
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
});

test("a key picked inside the system credentials stays put until they close, then shows above them", async () => {
  const { executeBodies } = mockWorkflow({});
  renderAgentPage();

  await openRunForm();
  await userEvent.click(
    await screen.findByRole("button", { name: SYSTEM_SECTION }),
  );
  const section = await screen.findByRole("region", { name: SYSTEM_SECTION });
  await userEvent.selectOptions(
    within(section).getByRole("combobox", { name: PICKER }),
    USER_KEY.id,
  );
  expect(
    (
      within(section).getByRole("combobox", {
        name: PICKER,
      }) as HTMLSelectElement
    ).value,
  ).toBe(USER_KEY.id);

  await userEvent.click(screen.getByRole("button", { name: SYSTEM_SECTION }));

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: SYSTEM_SECTION })).toBeNull(),
  );
  const picker = screen.getByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id);
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("a schedule made from the run form starts on the platform key, not the step's", async () => {
  const { scheduleBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
  });
  renderAgentPage();

  await openRunForm();
  const picker = await screen.findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id);
  await clickWhenEnabled("Schedule Task");
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(scheduleBodies).toHaveLength(1));
  expect(keyIn(scheduleBodies[0].credentials)).toBe(PLATFORM_KEY.id);
});

test("a schedule keeps a key the user picked in the run form, the step's included", async () => {
  const { scheduleBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    savedCredentials: [PLATFORM_KEY, USER_KEY, OTHER_KEY],
  });
  renderAgentPage();

  await openRunForm();
  await userEvent.selectOptions(
    await screen.findByRole("combobox", { name: PICKER }),
    OTHER_KEY.id,
  );
  await userEvent.selectOptions(
    screen.getByRole("combobox", { name: PICKER }),
    USER_KEY.id,
  );
  await clickWhenEnabled("Schedule Task");
  // The schedule already runs on the key the form shows, so nothing to say
  expect(await screen.findByRole("button", { name: "Schedule" })).toBeDefined();
  expect(screen.queryByRole("button", { name: USE_FORM_KEYS })).toBeNull();
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(scheduleBodies).toHaveLength(1));
  expect(keyIn(scheduleBodies[0].credentials)).toBe(USER_KEY.id);
});

test("a trigger set up from the run form starts on the platform key, not the step's", async () => {
  const { triggerBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    triggerSetupInfo: {
      provider: "generic_webhook",
      config_schema: { type: "object", properties: {} },
      credentials_input_name: null,
    },
  });
  renderAgentPage();

  await openRunForm();
  await userEvent.type(
    await screen.findByPlaceholderText("Enter trigger name"),
    "New trigger",
  );
  await clickWhenEnabled("Set up Trigger");

  await waitFor(() => expect(triggerBodies).toHaveLength(1));
  expect(keyIn(triggerBodies[0].agent_credentials)).toBe(PLATFORM_KEY.id);
});

test("a key added from the run form is the one the run uses", async () => {
  // No platform key for Firecrawl, and no key of the user's own yet
  const savedCredentials: Array<Record<string, unknown>> = [];
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    savedCredentials,
    systemProviders: [],
  });
  server.use(
    http.post(
      "*/api/integrations/firecrawl/credentials",
      async ({ request }) => {
        const { title } = (await request.json()) as { title: string };
        const created = { ...USER_KEY, id: "new-firecrawl-key", title };
        savedCredentials.push(created);
        return HttpResponse.json(created);
      },
    ),
  );
  renderAgentPage();

  await openRunForm();
  await userEvent.click(
    await screen.findByRole("button", { name: "Add API key" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: /^API Key/ }),
  );
  await userEvent.type(
    await screen.findByPlaceholderText("My Firecrawl key"),
    "New key",
  );
  await userEvent.type(screen.getByPlaceholderText("sk-..."), "fc-123");
  await userEvent.click(screen.getByRole("button", { name: "Continue" }));

  // The only key shows as a row rather than a picker
  expect(await screen.findByText("New key", { exact: true })).toBeDefined();
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe("new-firecrawl-key");
});

test("a user with no key of their own gets the form at once, on the platform key, while the graph still loads", async () => {
  let releaseGraph = () => {};
  const graphReady = new Promise<void>((resolve) => {
    releaseGraph = resolve;
  });
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    savedCredentials: [PLATFORM_KEY],
    graphReady,
  });
  renderAgentPage();

  try {
    await openRunForm();
    expect(
      await screen.findByRole("button", { name: SYSTEM_SECTION }),
    ).toBeDefined();
    expect(
      screen.queryByRole("status", { name: "Loading credentials" }),
    ).toBeNull();
    await clickWhenEnabled("Start Task");

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
  } finally {
    releaseGraph();
  }
});

test.each([
  { case: "no key of its own", stepCredentials: undefined },
  {
    case: "a key the user has since deleted",
    stepCredentials: savedOnStep({ id: "deleted-key", title: "Old key" }),
  },
])(
  "a step with $case runs on the platform key",
  async ({ stepCredentials }) => {
    const { executeBodies } = mockWorkflow({ stepCredentials });
    renderAgentPage();

    await openRunForm();
    expect(
      await screen.findByRole("button", { name: SYSTEM_SECTION }),
    ).toBeDefined();
    expect(screen.queryByRole("combobox", { name: PICKER })).toBeNull();
    await clickWhenEnabled("Start Task");

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
  },
);

test.each([
  {
    case: "two different keys of the user's",
    otherStepCredentials: [savedOnStep(OTHER_KEY)],
    subWorkflowStepCredentials: [],
  },
  {
    case: "the user's key and the platform key",
    otherStepCredentials: [savedOnStep(PLATFORM_KEY)],
    subWorkflowStepCredentials: [],
  },
  {
    case: "the user's key, and a sub-workflow's step to another",
    otherStepCredentials: [],
    subWorkflowStepCredentials: [savedOnStep(OTHER_KEY)],
  },
])(
  "steps set to $case don't agree, so the run falls back to the platform key",
  async ({ otherStepCredentials, subWorkflowStepCredentials }) => {
    const { executeBodies } = mockWorkflow({
      stepCredentials: savedOnStep(USER_KEY),
      otherStepCredentials,
      subWorkflowStepCredentials,
      savedCredentials: [PLATFORM_KEY, USER_KEY, OTHER_KEY],
    });
    renderAgentPage();

    await openRunForm();
    expect(
      await screen.findByRole("button", { name: SYSTEM_SECTION }),
    ).toBeDefined();
    expect(screen.queryByRole("combobox", { name: PICKER })).toBeNull();
    await clickWhenEnabled("Start Task");

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
  },
);

test("the user's key on every step, a sub-workflow's included, is the one the run uses", async () => {
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    otherStepCredentials: [{}],
    subWorkflowStepCredentials: [savedOnStep(USER_KEY)],
    savedCredentials: [PLATFORM_KEY, USER_KEY, OTHER_KEY],
  });
  renderAgentPage();

  await openRunForm();
  const picker = await screen.findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id);
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("when the steps' keys can't be read, the field waits for the user's pick, with a notice", async () => {
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
  });
  failGraphReads();
  renderAgentPage();

  await openRunForm();
  expect(await screen.findByText(LOOKUP_FAILED)).toBeDefined();
  const picker = await screen.findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe("__none__");
  expect(
    screen.getByRole("button", { name: "Start Task" }).hasAttribute("disabled"),
  ).toBe(true);

  await userEvent.selectOptions(picker, PLATFORM_KEY.id);
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
});

test("after a failed read of the steps' keys, Try again starts the form on the step's key", async () => {
  const { executeBodies, answerGraphReads } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
  });
  failGraphReads();
  renderAgentPage();

  await openRunForm();
  expect(await screen.findByText(LOOKUP_FAILED)).toBeDefined();
  answerGraphReads();
  await clickWhenEnabled("Try again");

  await waitFor(() => expect(screen.queryByText(LOOKUP_FAILED)).toBeNull());
  const picker = await screen.findByRole("combobox", { name: PICKER });
  await waitFor(() =>
    expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id),
  );
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("someone who doesn't own the workflow runs on the platform key, without reading its steps", async () => {
  // The mocked graph answers with the user's key on the step whoever asks,
  // so only the ownership check keeps it out of the run
  const { executeBodies, graphReads } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    canAccessGraph: false,
  });
  renderAgentPage();

  await openRunForm();
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
  expect(graphReads).toEqual([]);
  expect(
    screen.queryByRole("status", { name: "Loading credentials" }),
  ).toBeNull();
});

test("a step input that only looks like a key, with an unknown provider, doesn't break the run form", async () => {
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    otherStepCredentials: [
      { id: "not-a-key", provider: "constructor", type: "api_key" },
    ],
  });
  renderAgentPage();

  await openRunForm();
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("an LLM step's field, open to several providers, starts on the step's key and shows it", async () => {
  const { executeBodies } = mockLLMWorkflow();
  renderAgentPage();

  await openRunForm();
  const picker = await screen.findByRole("combobox", { name: LLM_PICKER });
  expect((picker as HTMLSelectElement).value).toBe(USER_OPENAI_KEY.id);
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs, LLM_FIELD)).toBe(
    USER_OPENAI_KEY.id,
  );
});

test("after a failed read, an LLM step's field open to several providers still shows, and Try again starts it on the step's key", async () => {
  const { executeBodies, answerGraphReads } = mockLLMWorkflow();
  failGraphReads();
  renderAgentPage();

  await openRunForm();
  expect(await screen.findByText(LOOKUP_FAILED)).toBeDefined();
  const picker = await screen.findByRole("combobox", { name: LLM_PICKER });
  expect((picker as HTMLSelectElement).value).toBe("__none__");

  answerGraphReads();
  await clickWhenEnabled("Try again");
  await waitFor(() =>
    expect(
      (screen.getByRole("combobox", { name: LLM_PICKER }) as HTMLSelectElement)
        .value,
    ).toBe(USER_OPENAI_KEY.id),
  );
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs, LLM_FIELD)).toBe(
    USER_OPENAI_KEY.id,
  );
});

test("a template that leaves an optional credential empty runs with none for it", async () => {
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    requiredFields: [],
    templateCredentials: {},
  });
  renderAgentPage(`activeTab=templates&activeItem=${TEMPLATE_ID}`);

  await clickWhenEnabled("New workflow task");
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBeUndefined();
});

test("after a failed read, an optional field starts on the platform key as before, and Try again moves it to the step's key", async () => {
  const { executeBodies, answerGraphReads } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    requiredFields: [],
  });
  failGraphReads();
  renderAgentPage();

  await openRunForm();
  expect(await screen.findByText(LOOKUP_FAILED)).toBeDefined();
  await userEvent.click(screen.getByRole("button", { name: SYSTEM_SECTION }));
  const section = await screen.findByRole("region", { name: SYSTEM_SECTION });
  await waitFor(() =>
    expect(
      (
        within(section).getByRole("combobox", {
          name: PICKER,
        }) as HTMLSelectElement
      ).value,
    ).toBe(PLATFORM_KEY.id),
  );

  answerGraphReads();
  await clickWhenEnabled("Try again");
  await waitFor(() => expect(screen.queryByText(LOOKUP_FAILED)).toBeNull());
  await waitFor(() =>
    expect(
      (screen.getByRole("combobox", { name: PICKER }) as HTMLSelectElement)
        .value,
    ).toBe(USER_KEY.id),
  );
  expect(screen.queryByRole("button", { name: SYSTEM_SECTION })).toBeNull();
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(USER_KEY.id);
});

test("a key of the user's own that doesn't fit the field doesn't hold it back when the steps' keys can't be read", async () => {
  const userLogin = {
    id: "user-firecrawl-login",
    provider: "firecrawl",
    type: "oauth2",
    title: "My Firecrawl login",
    scopes: [],
    is_system: false,
  };
  const { executeBodies } = mockWorkflow({
    savedCredentials: [PLATFORM_KEY, userLogin],
  });
  failGraphReads();
  renderAgentPage();

  await openRunForm();
  expect(
    await screen.findByRole("button", { name: SYSTEM_SECTION }),
  ).toBeDefined();
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
  expect(screen.queryByText(LOOKUP_FAILED)).toBeNull();
});

test("picking None on an optional field that started on the step's only key leaves it empty", async () => {
  // GitHub-like: no platform key, and the user has one key, saved on the step
  const { executeBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    requiredFields: [],
    savedCredentials: [USER_KEY],
    systemProviders: [],
  });
  renderAgentPage();

  await openRunForm();
  const picker = await screen.findByRole("combobox", { name: PICKER });
  expect((picker as HTMLSelectElement).value).toBe(USER_KEY.id);
  await userEvent.selectOptions(picker, "__none__");

  expect(
    (screen.getByRole("combobox", { name: PICKER }) as HTMLSelectElement).value,
  ).toBe("__none__");
  await clickWhenEnabled("Start Task");
  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBeUndefined();
});

// Finds the picker above the system credentials, or opens them to find it.
// After the form reopens, the fields mount inside the click's act(), so a key
// picked back on mount already shows once either is there.
async function findPicker() {
  await waitFor(() =>
    expect(
      screen.queryByRole("combobox", { name: PICKER }) ??
        screen.queryByRole("button", { name: SYSTEM_SECTION }),
    ).not.toBeNull(),
  );
  if (!screen.queryByRole("combobox", { name: PICKER })) {
    await userEvent.click(screen.getByRole("button", { name: SYSTEM_SECTION }));
  }
  return (await screen.findByRole("combobox", {
    name: PICKER,
  })) as HTMLSelectElement;
}

async function closeRunForm() {
  await userEvent.keyboard("{Escape}");
  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Start Task" })).toBeNull(),
  );
}

test.each([
  {
    case: "with several keys to pick from",
    savedCredentials: [PLATFORM_KEY, USER_KEY, OTHER_KEY],
    systemProviders: ["firecrawl"],
  },
  {
    case: "on the user's only key",
    savedCredentials: [USER_KEY],
    systemProviders: [],
  },
])(
  "None on an optional field survives closing and reopening the form, $case",
  async ({ savedCredentials, systemProviders }) => {
    const { executeBodies } = mockWorkflow({
      stepCredentials: savedOnStep(USER_KEY),
      requiredFields: [],
      savedCredentials,
      systemProviders,
    });
    renderAgentPage();

    await openRunForm();
    await expectPickerOn(USER_KEY);
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: PICKER }),
      "__none__",
    );
    // Whatever could pick a key back runs inside the change's act()
    const afterNone = pickerValue();
    await closeRunForm();
    await openRunForm();
    const afterReopen = (await findPicker()).value;
    await clickWhenEnabled("Start Task");

    await waitFor(() => expect(executeBodies).toHaveLength(1));
    expect({
      afterNone,
      afterReopen,
      sent: keyIn(executeBodies[0].credentials_inputs) ?? null,
    }).toEqual({ afterNone: "__none__", afterReopen: "__none__", sent: null });
  },
);

test("after a failed read, a platform key the user picked themselves survives Try again, which moves a field they left to its step's key", async () => {
  const { executeBodies, answerGraphReads } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    otherStepCredentials: [savedOnStep(USER_EXA_KEY)],
    credentialFields: {
      [FIELD]: FIRECRAWL_FIELD,
      [EXA_FIELD]: EXA_FIELD_SCHEMA,
    },
    requiredFields: [],
    providers: ["firecrawl", "exa"],
    systemProviders: ["firecrawl", "exa"],
    savedCredentials: [
      PLATFORM_KEY,
      USER_KEY,
      OTHER_KEY,
      EXA_PLATFORM_KEY,
      USER_EXA_KEY,
    ],
  });
  failGraphReads();
  renderAgentPage();

  await openRunForm();
  expect(await screen.findByText(LOOKUP_FAILED)).toBeDefined();
  await userEvent.click(screen.getByRole("button", { name: SYSTEM_SECTION }));
  const section = await screen.findByRole("region", { name: SYSTEM_SECTION });
  const picker = within(section).getByRole("combobox", { name: PICKER });
  await waitFor(() =>
    expect((picker as HTMLSelectElement).value).toBe(PLATFORM_KEY.id),
  );
  // The user tries another key, then settles on platform credits
  await userEvent.selectOptions(picker, OTHER_KEY.id);
  await userEvent.selectOptions(
    screen.getByRole("combobox", { name: PICKER }),
    PLATFORM_KEY.id,
  );

  answerGraphReads();
  await clickWhenEnabled("Try again");
  // One pass sets every field the steps' keys decide, so once the field the
  // user left is on its step's key, the one they set is final too
  await expectPickerOn(USER_EXA_KEY, EXA_PICKER);
  expect(screen.queryByText(LOOKUP_FAILED)).toBeNull();
  expect((await findPicker()).value).toBe(PLATFORM_KEY.id);
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs)).toBe(PLATFORM_KEY.id);
  expect(keyIn(executeBodies[0].credentials_inputs, EXA_FIELD)).toBe(
    USER_EXA_KEY.id,
  );
});

test("an LLM step whose model comes from a link, saved on Anthropic's platform key, runs on it, not on OpenAI's", async () => {
  const { executeBodies } = mockLLMWorkflow({
    stepCredentials: savedOnStep(ANTHROPIC_PLATFORM_KEY),
  });
  renderAgentPage();

  await openRunForm();
  await clickWhenEnabled("Start Task");

  await waitFor(() => expect(executeBodies).toHaveLength(1));
  expect(keyIn(executeBodies[0].credentials_inputs, LLM_FIELD)).toBe(
    ANTHROPIC_PLATFORM_KEY.id,
  );
});

// No platform key for any LLM provider, and none of the user's own
test.each([
  { case: "a provider of the user's", systemProviders: [] },
  // Listed as the platform's, as when its key isn't set up on the server
  { case: "the platform's", systemProviders: ["openai", "anthropic"] },
])(
  "a required field open to several providers, each $case, with no key to pick, stays in view and says where to add one",
  async ({ systemProviders }) => {
    mockLLMWorkflow({
      stepCredentials: undefined,
      systemProviders,
      savedCredentials: [],
    });
    renderAgentPage();

    await openRunForm();
    const link = await screen.findByRole("link", {
      name: "Settings → Integrations",
    });
    expect(link.getAttribute("href")).toBe("/settings/integrations");
    expect(screen.getByText(/Openai or Anthropic/)).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "Start Task" })
        .hasAttribute("disabled"),
    ).toBe(true);
  },
);

const REFUSED =
  "This block keeps what it creates in the Firecrawl account behind its key, so it can't use the platform's shared Firecrawl key. Connect your own Firecrawl API key and select it instead.";

// The backend checks a schedule's credentials when the schedule is made, and a
// block that keeps what it creates in the vendor's account (Exa Websets,
// SmartLead campaigns: `refuses_platform_credentials`) refuses the platform's
// key there
function refusePlatformKeyOnSchedules() {
  const accepted: Array<{ credentials?: unknown }> = [];
  const refused: Array<{ credentials?: unknown }> = [];
  server.use(
    http.post("*/api/graphs/:graphId/schedules", async ({ request }) => {
      const body = (await request.json()) as { credentials?: unknown };
      if (keyIn(body.credentials) === PLATFORM_KEY.id) {
        refused.push(body);
        return HttpResponse.json({ detail: REFUSED }, { status: 400 });
      }
      accepted.push(body);
      return HttpResponse.json(
        getPostV1CreateExecutionScheduleResponseMock({
          id: "new-schedule",
          graph_id: "graph-1",
        }),
      );
    }),
  );
  return { accepted, refused };
}

test("a schedule for a step that refuses the platform's key is made on the user's key the run form shows", async () => {
  mockWorkflow({ stepCredentials: savedOnStep(USER_KEY) });
  const { accepted, refused } = refusePlatformKeyOnSchedules();
  renderAgentPage();

  await openRunForm();
  await expectPickerOn(USER_KEY);
  await clickWhenEnabled("Schedule Task");
  // The schedule starts on the platform's key, and says so
  expect(
    await screen.findByText(
      "Firecrawl credentials: the platform's key instead of My Firecrawl key",
    ),
  ).toBeDefined();
  await clickWhenEnabled(USE_FORM_KEYS);
  expect(
    await screen.findByText(
      "This schedule uses the keys shown in the run form.",
    ),
  ).toBeDefined();
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(accepted.length + refused.length).toBe(1));
  expect(refused.map((body) => keyIn(body.credentials))).toEqual([]);
  expect(accepted.map((body) => keyIn(body.credentials))).toEqual([
    USER_KEY.id,
  ]);
});

test("a schedule keeps the key the run form shows on an optional field, rather than dropping it", async () => {
  // GitHub-like: no platform key, and the user has two keys of their own
  const { scheduleBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
    requiredFields: [],
    savedCredentials: [USER_KEY, OTHER_KEY],
    systemProviders: [],
  });
  renderAgentPage();

  await openRunForm();
  await expectPickerOn(USER_KEY);
  await clickWhenEnabled("Schedule Task");
  // Sent with no key, the backend skips the field's steps on every run
  expect(
    await screen.findByText(
      "Firecrawl credentials: no key instead of My Firecrawl key, so the steps that use it are skipped",
    ),
  ).toBeDefined();
  await clickWhenEnabled(USE_FORM_KEYS);
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(scheduleBodies).toHaveLength(1));
  expect(keyIn(scheduleBodies[0].credentials)).toBe(USER_KEY.id);
});

test("a schedule for an LLM step on the user's OpenAI key starts on OpenAI's platform key, whatever order the field lists its providers in", async () => {
  const { scheduleBodies } = mockLLMWorkflow({
    credentialFields: {
      [LLM_FIELD]: {
        ...LLM_FIELD_SCHEMA,
        credentials_provider: ["anthropic", "openai"],
      },
    },
  });
  renderAgentPage();

  await openRunForm();
  await expectPickerOn(USER_OPENAI_KEY, LLM_PICKER);
  await clickWhenEnabled("Schedule Task");
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(scheduleBodies).toHaveLength(1));
  expect(keyIn(scheduleBodies[0].credentials, LLM_FIELD)).toBe(
    OPENAI_PLATFORM_KEY.id,
  );
});
