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
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import {
  getGetV1ListExecutionSchedulesForAGraphMockHandler,
  getPostV1CreateExecutionScheduleMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
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

// What a step's `input_default.credentials` holds once a key is picked on it.
function savedOnStep(key: { id: string; title: string }) {
  return {
    id: key.id,
    provider: "firecrawl",
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
  systemProviders?: string[];
}

function mockWorkflow({
  stepCredentials,
  otherStepCredentials = [],
  subWorkflowStepCredentials = [],
  graphReady,
  savedCredentials = [PLATFORM_KEY, USER_KEY],
  systemProviders = ["firecrawl"],
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
  function answerGraphReads() {
    server.use(
      getGetV1GetSpecificGraphMockHandler(async () => {
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
        can_access_graph: true,
        input_schema: { type: "object", properties: {} },
        credentials_input_schema: {
          type: "object",
          properties: {
            [FIELD]: {
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
          },
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
        { name: "firecrawl", description: "Web scraping and crawling" },
      ]),
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
      presets: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 1,
        page_size: 100,
      },
    }),
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
  );
  answerGraphReads();

  return { executeBodies, scheduleBodies, answerGraphReads };
}

// Every read of the graph fails, until answerGraphReads() is called again
function failGraphReads() {
  server.use(getGetV1GetSpecificGraphMockHandler422());
}

function renderAgentPage() {
  render(
    <CredentialsProvider>
      <NuqsTestingAdapter hasMemory>
        <NewAgentLibraryView />
      </NuqsTestingAdapter>
    </CredentialsProvider>,
  );
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

function keyIn(credentials: unknown) {
  return (credentials as Record<string, { id?: string }> | undefined)?.[FIELD]
    ?.id;
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

test("a schedule made from the run form uses the step's key too", async () => {
  const { scheduleBodies } = mockWorkflow({
    stepCredentials: savedOnStep(USER_KEY),
  });
  renderAgentPage();

  await openRunForm();
  await clickWhenEnabled("Schedule Task");
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(scheduleBodies).toHaveLength(1));
  expect(keyIn(scheduleBodies[0].credentials)).toBe(USER_KEY.id);
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
