import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import {
  getGetV1GetSpecificGraphResponseMock,
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { server } from "@/mocks/mock-server";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { ReactFlowProvider } from "@xyflow/react";
import { HttpResponse, http } from "msw";
import { afterEach, expect, test, vi } from "vitest";
import { RunInputDialog } from "../RunInputDialog";

// CredentialsProvider only loads credentials for a signed-in user.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

const FIELD = "firecrawl_api_key_credentials";
const PICKER = "Select Firecrawl credential";
const SYSTEM_SECTION = /^System credentials/;
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

type Key = typeof USER_KEY;

const FIRECRAWL_FIELD = {
  credentials_provider: ["firecrawl"],
  credentials_types: ["api_key"],
  title: "Credentials",
  type: "object",
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

// A step's saved inputs, with this key picked on it
function inputsWith(key?: Key) {
  if (!key) return { url: "https://example.com" };
  return {
    url: "https://example.com",
    credentials: {
      id: key.id,
      provider: key.provider,
      type: "api_key",
      title: key.title,
    },
  };
}

function scrapeStep(id: string, key?: Key) {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      hardcodedValues: inputsWith(key),
      title: "Firecrawl Scrape",
      description: "Scrape a page",
      inputSchema: {},
      outputSchema: {},
      block_id: "firecrawl-scrape",
      costs: [],
      categories: [],
    },
  } as unknown as CustomNode;
}

function setStepKeys(keys: Array<Key | undefined>) {
  useNodeStore.setState({
    nodes: keys.map((key, index) => scrapeStep(`scrape-step-${index}`, key)),
  });
}

// The saved workflow the canvas loaded, whose sub-workflow has these steps
function setSubWorkflowStepKeys(keys: Array<Key | undefined>) {
  const { sub_graphs: _, ...subWorkflow } =
    getGetV1GetSpecificGraphResponseMock({
      id: "sub-graph-1",
      version: 1,
      links: [],
      sub_graphs: [],
      nodes: keys.map((key, index) => ({
        id: `sub-step-${index}`,
        block_id: "firecrawl-scrape",
        graph_id: "sub-graph-1",
        graph_version: 1,
        input_default: inputsWith(key),
        metadata: {},
        input_links: [],
        output_links: [],
      })),
    });
  useGraphStore.setState({
    loadedGraph: getGetV1GetSpecificGraphResponseMock({
      id: "graph-1",
      version: 1,
      links: [],
      nodes: [],
      sub_graphs: [subWorkflow],
    }),
  });
}

function mockBuilderRun({
  stepKeys,
  subWorkflowStepKeys = [],
  field = FIELD,
  fieldSchema = FIRECRAWL_FIELD,
  savedCredentials = [PLATFORM_KEY, USER_KEY, OTHER_KEY],
  required = true,
  systemProviders,
}: {
  stepKeys: Array<Key | undefined>;
  subWorkflowStepKeys?: Array<Key | undefined>;
  field?: string;
  fieldSchema?: typeof FIRECRAWL_FIELD;
  savedCredentials?: Key[];
  required?: boolean;
  // The providers the platform has a key for; all of the field's by default
  systemProviders?: string[];
}) {
  const executeBodies: Array<{ credentials_inputs?: unknown }> = [];
  const providers = fieldSchema.credentials_provider;
  useGraphStore.setState({
    inputSchema: null,
    credentialsInputSchema: {
      type: "object",
      properties: { [field]: fieldSchema },
      required: required ? [field] : [],
    },
  });
  setStepKeys(stepKeys);
  if (subWorkflowStepKeys.length > 0) {
    setSubWorkflowStepKeys(subWorkflowStepKeys);
  }
  server.use(
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json(
        providers.map((name) => ({ name, description: `${name} provider` })),
      ),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json(systemProviders ?? providers),
    ),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json(savedCredentials),
    ),
    getPostV1ExecuteGraphAgentMockHandler(async ({ request }) => {
      executeBodies.push((await request.json()) as never);
      return getPostV1ExecuteGraphAgentResponseMock({
        id: "builder-run",
        graph_id: "graph-1",
      });
    }),
  );
  return executeBodies;
}

function renderRunDialog(purpose: "run" | "schedule" = "run") {
  function dialog(isOpen: boolean) {
    return (
      <CredentialsProvider>
        <ReactFlowProvider>
          <RunInputDialog
            isOpen={isOpen}
            setIsOpen={vi.fn()}
            purpose={purpose}
            graphID="graph-1"
            graphVersion={1}
          />
        </ReactFlowProvider>
      </CredentialsProvider>
    );
  }
  const { rerender } = render(dialog(true));
  return {
    close: () => rerender(dialog(false)),
    reopen: () => rerender(dialog(true)),
  };
}

async function expectPickerOn(key: { id: string }, name = PICKER) {
  const picker = await screen.findByRole("combobox", { name });
  await waitFor(() => expect((picker as HTMLSelectElement).value).toBe(key.id));
}

// The platform key sits inside the collapsed system credentials
async function expectPlatformKeyPicked() {
  await userEvent.click(
    await screen.findByRole("button", { name: SYSTEM_SECTION }),
  );
  await expectPickerOn(PLATFORM_KEY);
}

async function runAndGetKey(
  executeBodies: Array<{ credentials_inputs?: unknown }>,
  field = FIELD,
) {
  await userEvent.click(screen.getByRole("button", { name: "Manual Run" }));
  await waitFor(() => expect(executeBodies).toHaveLength(1));
  return (
    executeBodies[0].credentials_inputs as
      | Record<string, { id?: string }>
      | undefined
  )?.[field]?.id;
}

afterEach(() => {
  useGraphStore.getState().reset();
  useNodeStore.setState({ nodes: [] });
});

test("a builder run uses the key picked on the step, and shows it", async () => {
  const executeBodies = mockBuilderRun({ stepKeys: [USER_KEY] });
  renderRunDialog();

  await expectPickerOn(USER_KEY);

  expect(await runAndGetKey(executeBodies)).toBe(USER_KEY.id);
});

test("a builder run whose step has no key of its own uses the platform key", async () => {
  const executeBodies = mockBuilderRun({ stepKeys: [undefined] });
  renderRunDialog();

  // With no key on the step, the platform key is picked inside the collapsed section
  await expectPlatformKeyPicked();

  expect(await runAndGetKey(executeBodies)).toBe(PLATFORM_KEY.id);
});

test.each([
  {
    case: "two different keys of the user's",
    stepKeys: [USER_KEY, OTHER_KEY],
    subWorkflowStepKeys: [],
  },
  {
    case: "the user's key and the platform key",
    stepKeys: [USER_KEY, PLATFORM_KEY],
    subWorkflowStepKeys: [],
  },
  {
    case: "the user's key, and a sub-workflow's step to another",
    stepKeys: [USER_KEY],
    subWorkflowStepKeys: [OTHER_KEY],
  },
])(
  "builder steps set to $case don't agree, so the run falls back to the platform key",
  async ({ stepKeys, subWorkflowStepKeys }) => {
    const executeBodies = mockBuilderRun({ stepKeys, subWorkflowStepKeys });
    renderRunDialog();

    await expectPlatformKeyPicked();

    expect(await runAndGetKey(executeBodies)).toBe(PLATFORM_KEY.id);
  },
);

test.each([
  {
    case: "on every step, a sub-workflow's included",
    stepKeys: [USER_KEY, undefined],
    subWorkflowStepKeys: [USER_KEY],
  },
  {
    case: "only on a sub-workflow's step",
    stepKeys: [undefined],
    subWorkflowStepKeys: [USER_KEY],
  },
])(
  "the user's key $case is the one a builder run uses",
  async ({ stepKeys, subWorkflowStepKeys }) => {
    const executeBodies = mockBuilderRun({ stepKeys, subWorkflowStepKeys });
    renderRunDialog();

    await expectPickerOn(USER_KEY);

    expect(await runAndGetKey(executeBodies)).toBe(USER_KEY.id);
  },
);

test.each([
  { case: "the platform key", before: undefined, after: USER_KEY },
  { case: "the user's key", before: USER_KEY, after: OTHER_KEY },
])(
  "a builder dialog that started on $case starts on the step's new key when reopened",
  async ({ before, after }) => {
    const executeBodies = mockBuilderRun({ stepKeys: [before] });
    const { close, reopen } = renderRunDialog();
    if (before) await expectPickerOn(before);
    else await expectPlatformKeyPicked();

    close();
    act(() => setStepKeys([after]));
    reopen();

    await expectPickerOn(after);
    expect(await runAndGetKey(executeBodies)).toBe(after.id);
  },
);

test("an open builder dialog moves to the step's key once the saved workflow loads it", async () => {
  const executeBodies = mockBuilderRun({ stepKeys: [undefined] });
  renderRunDialog();
  await expectPlatformKeyPicked();

  act(() => setSubWorkflowStepKeys([USER_KEY]));

  await expectPickerOn(USER_KEY);
  expect(await runAndGetKey(executeBodies)).toBe(USER_KEY.id);
});

test("a key the user picked in the builder dialog stays after the step's key changes", async () => {
  const executeBodies = mockBuilderRun({ stepKeys: [USER_KEY] });
  const { close, reopen } = renderRunDialog();
  await expectPickerOn(USER_KEY);
  await userEvent.selectOptions(
    screen.getByRole("combobox", { name: PICKER }),
    PLATFORM_KEY.id,
  );
  await expectPickerOn(PLATFORM_KEY);

  close();
  act(() => setStepKeys([OTHER_KEY]));
  reopen();

  expect(
    await screen.findByRole("button", { name: SYSTEM_SECTION }),
  ).toBeDefined();
  expect(await runAndGetKey(executeBodies)).toBe(PLATFORM_KEY.id);
});

test("a builder run's LLM field, open to several providers, starts on the step's key and shows it", async () => {
  const executeBodies = mockBuilderRun({
    stepKeys: [USER_OPENAI_KEY],
    field: LLM_FIELD,
    fieldSchema: LLM_FIELD_SCHEMA,
    savedCredentials: [
      OPENAI_PLATFORM_KEY,
      ANTHROPIC_PLATFORM_KEY,
      USER_OPENAI_KEY,
    ],
  });
  renderRunDialog();

  await expectPickerOn(USER_OPENAI_KEY, LLM_PICKER);

  expect(await runAndGetKey(executeBodies, LLM_FIELD)).toBe(USER_OPENAI_KEY.id);
});

test("a reopened builder dialog keeps None on an optional field", async () => {
  // No platform key, and two keys of the user's, so nothing picks one itself
  const executeBodies = mockBuilderRun({
    stepKeys: [USER_KEY],
    savedCredentials: [USER_KEY, OTHER_KEY],
    required: false,
    systemProviders: [],
  });
  const { close, reopen } = renderRunDialog();

  await expectPickerOn(USER_KEY);
  await userEvent.selectOptions(
    screen.getByRole("combobox", { name: PICKER }),
    "__none__",
  );
  // The dialog's keys live in state, with no request behind them, so
  // whatever could pick a key back runs inside the change's act()
  expect(
    (screen.getByRole("combobox", { name: PICKER }) as HTMLSelectElement).value,
  ).toBe("__none__");

  close();
  reopen();
  // The fields mount inside the rerender's act(), and so do the effects that
  // pick a key on mount
  expect(
    (screen.getByRole("combobox", { name: PICKER }) as HTMLSelectElement).value,
  ).toBe("__none__");
  expect(await runAndGetKey(executeBodies)).toBeUndefined();
});

test("a builder LLM step whose model comes from a link, saved on Anthropic's platform key, runs on it", async () => {
  const executeBodies = mockBuilderRun({
    stepKeys: [ANTHROPIC_PLATFORM_KEY],
    field: LLM_FIELD,
    fieldSchema: LLM_FIELD_SCHEMA,
    savedCredentials: [
      OPENAI_PLATFORM_KEY,
      ANTHROPIC_PLATFORM_KEY,
      USER_OPENAI_KEY,
    ],
  });
  renderRunDialog();

  // Anthropic's only key is a row inside the system credentials
  await userEvent.click(
    await screen.findByRole("button", { name: SYSTEM_SECTION }),
  );
  expect(await screen.findByText(ANTHROPIC_PLATFORM_KEY.title)).toBeDefined();

  expect(await runAndGetKey(executeBodies, LLM_FIELD)).toBe(
    ANTHROPIC_PLATFORM_KEY.id,
  );
});

test("the builder's schedule dialog starts on the platform key, not the step's", async () => {
  mockBuilderRun({ stepKeys: [USER_KEY] });
  renderRunDialog("schedule");

  await expectPlatformKeyPicked();
  expect(screen.getAllByRole("combobox", { name: PICKER })).toHaveLength(1);
});

test("a builder field open to several providers, with no key to pick, says where to add one", async () => {
  // No platform key for any LLM provider, and none of the user's own
  mockBuilderRun({
    stepKeys: [undefined],
    field: LLM_FIELD,
    fieldSchema: LLM_FIELD_SCHEMA,
    savedCredentials: [],
    systemProviders: [],
  });
  renderRunDialog();

  const link = await screen.findByRole("link", {
    name: "Settings → Integrations",
  });
  expect(link.getAttribute("href")).toBe("/settings/integrations");
  expect(screen.getByText(/Openai or Anthropic/)).toBeDefined();
});
