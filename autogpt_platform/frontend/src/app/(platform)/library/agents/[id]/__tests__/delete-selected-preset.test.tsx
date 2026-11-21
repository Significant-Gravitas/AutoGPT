import { getGetV2GetSpecificBlocksMockHandler } from "@/app/api/__generated__/endpoints/default/default.msw";
import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1ListGraphExecutionsMockHandler,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getDeleteV2DeleteLibraryAgentMockHandler,
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
  getGetV2ListTriggerAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getDeleteV2DeleteAPresetMockHandler,
  getGetV2GetASpecificPresetMockHandler,
  getGetV2ListPresetsMockHandler,
} from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { server } from "@/mocks/mock-server";
import {
  act,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse } from "msw";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeEach, expect, test, vi } from "vitest";
import AgentLibraryPage from "../page";

const toast = vi.fn();

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  useToast: () => ({ toast }),
}));

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

// Trigger workflows are listed only behind this flag; every other flag is off.
const triggerAgentsFlag = vi.hoisted(() => ({ on: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  function isOn(flag: unknown) {
    return flag === actual.Flag.GENERIC_TRIGGER_AGENTS && triggerAgentsFlag.on;
  }
  return {
    ...actual,
    useGetFlag: isOn,
    useFlagStatus: (flag: unknown) => ({ enabled: isOn(flag), ready: true }),
  };
});

const GRAPH_ID = "graph-1";
const RUN_OUTPUT = "Output of run-a";

type User = ReturnType<typeof userEvent.setup>;
type DeleteThrough = (user: User, id: string) => Promise<void>;

let listedPresets: LibraryAgentPreset[] = [];
let listedTriggerAgents: LibraryAgent[] = [];

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

// Presets with a webhook are listed under Triggers, the rest under Templates.
function makePreset(args: {
  id: string;
  name: string;
  minutesAgo: number;
  webhook?: boolean;
}): LibraryAgentPreset {
  const at = new Date(Date.now() - args.minutesAgo * 60_000);
  return {
    id: args.id,
    name: args.name,
    description: "",
    graph_id: GRAPH_ID,
    graph_version: 1,
    inputs: {},
    credentials: {},
    is_active: true,
    user_id: "user-1",
    created_at: at,
    updated_at: at,
    webhook_id: args.webhook ? `webhook-${args.id}` : null,
    webhook: null,
  };
}

function makeTriggerAgent(id: string, name: string, description: string) {
  return getGetV2GetLibraryAgentResponseMock({
    id,
    graph_id: `${id}-graph`,
    name,
    description,
    is_hidden: true,
  });
}

beforeEach(() => {
  toast.mockClear();
  triggerAgentsFlag.on = false;
  listedPresets = [];
  listedTriggerAgents = [];
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Inbox digest",
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
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...run,
      inputs: {},
      outputs: { result: [RUN_OUTPUT] },
      node_executions: [],
    })),
    getGetV1ListExecutionSchedulesForAGraphMockHandler([]),
    getGetV2ListPresetsMockHandler(() => ({
      presets: listedPresets,
      pagination: {
        total_items: listedPresets.length,
        total_pages: 1,
        current_page: 1,
        page_size: 100,
      },
    })),
    getGetV2GetASpecificPresetMockHandler(({ params }) => {
      const preset = listedPresets.find(({ id }) => id === params.presetId);
      if (!preset) {
        throw HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }
      return preset;
    }),
    getDeleteV2DeleteAPresetMockHandler(({ params }) => {
      listedPresets = listedPresets.filter(({ id }) => id !== params.presetId);
    }),
    getGetV2ListTriggerAgentsMockHandler(() => listedTriggerAgents),
    getDeleteV2DeleteLibraryAgentMockHandler(
      ({ params }: { params: Record<string, string> }) => {
        listedTriggerAgents = listedTriggerAgents.filter(
          ({ id }) => id !== params.libraryAgentId,
        );
      },
    ),
    getGetV2GetSpecificBlocksMockHandler([]),
  );
});

function renderPage(searchParams: string) {
  const user = userEvent.setup();
  const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
  render(
    <NuqsTestingAdapter
      hasMemory
      onUrlUpdate={onUrlUpdate}
      searchParams={searchParams}
    >
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  return {
    user,
    // The URL is written only once the selection changes.
    urlParam: (name: string) =>
      (
        onUrlUpdate.mock.lastCall?.[0].searchParams ??
        new URLSearchParams(searchParams)
      ).get(name),
  };
}

async function confirmInDialog(user: User, confirm: string) {
  const dialog = await screen.findByRole("dialog");
  await user.click(within(dialog).getByRole("button", { name: confirm }));
}

async function deleteThroughSidebarMenu(
  user: User,
  args: { rowTestID: string; menuItem: string; confirm: string },
) {
  const row = await screen.findByTestId(args.rowTestID);
  await user.click(within(row).getByRole("button", { name: "More actions" }));
  await user.click(
    await screen.findByRole("menuitem", { name: args.menuItem }),
  );
  await confirmInDialog(user, args.confirm);
}

async function deleteThroughView(
  user: User,
  args: { button: string; confirm: string },
) {
  await user.click(await screen.findByRole("button", { name: args.button }));
  await confirmInDialog(user, args.confirm);
}

const deleteTemplate: Record<string, DeleteThrough> = {
  "its sidebar row": (user, id) =>
    deleteThroughSidebarMenu(user, {
      rowTestID: `library-template-item-${id}`,
      menuItem: "Delete template",
      confirm: "Delete Template",
    }),
  "its view": (user) =>
    deleteThroughView(user, { button: "Delete template", confirm: "Delete" }),
};

const deleteWebhookTrigger: Record<string, DeleteThrough> = {
  "its sidebar row": (user, id) =>
    deleteThroughSidebarMenu(user, {
      rowTestID: `library-trigger-item-${id}`,
      menuItem: "Delete trigger",
      confirm: "Delete Trigger",
    }),
  "its view": (user) =>
    deleteThroughView(user, { button: "Delete trigger", confirm: "Delete" }),
};

const removeTriggerAgent: Record<string, DeleteThrough> = {
  "its sidebar row": (user, id) =>
    deleteThroughSidebarMenu(user, {
      rowTestID: `library-trigger-agent-item-${id}`,
      menuItem: "Remove trigger",
      confirm: "Remove trigger",
    }),
  "its view": (user) =>
    deleteThroughView(user, {
      button: "Remove trigger",
      confirm: "Remove trigger",
    }),
};

// Each selected item deleted below is the first of its list: a selection
// emptied against the list from before the delete would pick it again.
test.each(Object.entries(deleteTemplate))(
  "deleting the selected template through %s selects the template left",
  async (_, deleteThrough) => {
    listedPresets = [
      makePreset({ id: "tpl-1", name: "Morning digest", minutesAgo: 1 }),
      makePreset({ id: "tpl-2", name: "Evening digest", minutesAgo: 2 }),
    ];
    const { user, urlParam } = renderPage(
      "activeTab=templates&activeItem=tpl-1",
    );
    expect(await screen.findByDisplayValue("Morning digest")).toBeDefined();

    await deleteThrough(user, "tpl-1");

    await waitFor(() => expect(urlParam("activeItem")).toBe("tpl-2"));
    expect(urlParam("activeTab")).toBe("templates");
    expect(await screen.findByDisplayValue("Evening digest")).toBeDefined();
    expect(screen.queryByDisplayValue("Morning digest")).toBeNull();
    expect(screen.queryByText("Morning digest")).toBeNull();
    expect(toast).toHaveBeenCalledWith({ title: "Template deleted" });
  },
);

test("deleting a template that isn't selected through its sidebar row keeps the selection", async () => {
  listedPresets = [
    makePreset({ id: "tpl-1", name: "Morning digest", minutesAgo: 1 }),
    makePreset({ id: "tpl-2", name: "Evening digest", minutesAgo: 2 }),
    makePreset({ id: "tpl-3", name: "Weekly digest", minutesAgo: 3 }),
  ];
  const { user, urlParam } = renderPage("activeTab=templates&activeItem=tpl-2");
  expect(await screen.findByDisplayValue("Evening digest")).toBeDefined();

  await deleteTemplate["its sidebar row"](user, "tpl-3");

  await waitFor(() =>
    expect(screen.queryByTestId("library-template-item-tpl-3")).toBeNull(),
  );
  expect(toast).toHaveBeenCalledWith({ title: "Template deleted" });
  // Let the delete handler finish what it does after the list refetch.
  await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
  // Not tpl-1: the first template is what an emptied selection falls back to.
  expect(screen.getByDisplayValue("Evening digest")).toBeDefined();
  expect(urlParam("activeItem")).toBe("tpl-2");
});

test("deleting the last template through its view goes to the Runs tab", async () => {
  listedPresets = [
    makePreset({ id: "tpl-1", name: "Morning digest", minutesAgo: 1 }),
    // A trigger is a preset too, but not a template left on the tab.
    makePreset({
      id: "trg-1",
      name: "Push hook",
      minutesAgo: 2,
      webhook: true,
    }),
  ];
  const { user, urlParam } = renderPage("activeTab=templates&activeItem=tpl-1");
  expect(await screen.findByDisplayValue("Morning digest")).toBeDefined();

  await deleteTemplate["its view"](user, "tpl-1");

  await waitFor(() => expect(urlParam("activeTab")).toBe("runs"));
  await waitFor(() => expect(urlParam("activeItem")).toBe("run-a"));
  expect(await screen.findByText(RUN_OUTPUT)).toBeDefined();
});

test.each(Object.entries(deleteWebhookTrigger))(
  "deleting the selected webhook trigger through %s selects the trigger left",
  async (_, deleteThrough) => {
    listedPresets = [
      makePreset({
        id: "trg-1",
        name: "Push hook",
        minutesAgo: 1,
        webhook: true,
      }),
      makePreset({
        id: "trg-2",
        name: "Issue hook",
        minutesAgo: 2,
        webhook: true,
      }),
    ];
    const { user, urlParam } = renderPage(
      "activeTab=triggers&activeItem=preset:trg-1",
    );
    expect(await screen.findByDisplayValue("Push hook")).toBeDefined();

    await deleteThrough(user, "trg-1");

    await waitFor(() => expect(urlParam("activeItem")).toBe("preset:trg-2"));
    expect(urlParam("activeTab")).toBe("triggers");
    expect(await screen.findByDisplayValue("Issue hook")).toBeDefined();
    expect(screen.queryByDisplayValue("Push hook")).toBeNull();
    expect(screen.queryByText("Push hook")).toBeNull();
    expect(screen.queryByText("Trigger not found")).toBeNull();
    expect(toast).toHaveBeenCalledWith({ title: "Trigger deleted" });
  },
);

test.each(Object.entries(removeTriggerAgent))(
  "removing the selected trigger workflow through %s selects the trigger left",
  async (_, removeThrough) => {
    triggerAgentsFlag.on = true;
    listedTriggerAgents = [
      makeTriggerAgent("ta-1", "Inbox watcher", "Watches the inbox"),
      makeTriggerAgent("ta-2", "Calendar watcher", "Watches the calendar"),
    ];
    const { user, urlParam } = renderPage(
      "activeTab=triggers&activeItem=agent:ta-1",
    );
    expect(await screen.findByText("Watches the inbox")).toBeDefined();

    await removeThrough(user, "ta-1");

    await waitFor(() => expect(urlParam("activeItem")).toBe("agent:ta-2"));
    expect(urlParam("activeTab")).toBe("triggers");
    expect(await screen.findByText("Watches the calendar")).toBeDefined();
    expect(screen.queryByText("Watches the inbox")).toBeNull();
    expect(screen.queryByText("Inbox watcher")).toBeNull();
    expect(toast).toHaveBeenCalledWith({ title: "Trigger removed" });
  },
);

test.each([
  ...Object.entries(deleteWebhookTrigger).map(
    ([path, deleteThrough]) =>
      ["webhook trigger", path, deleteThrough, "preset:trg-1"] as const,
  ),
  ...Object.entries(removeTriggerAgent).map(
    ([path, deleteThrough]) =>
      ["trigger workflow", path, deleteThrough, "agent:trg-1"] as const,
  ),
])(
  "deleting the last %s through %s leaves the emptied Triggers tab for Runs",
  async (_, __, deleteThrough, activeItem) => {
    triggerAgentsFlag.on = true;
    if (activeItem.startsWith("agent:")) {
      listedTriggerAgents = [
        makeTriggerAgent("trg-1", "Inbox watcher", "Watches the inbox"),
      ];
    } else {
      listedPresets = [
        makePreset({
          id: "trg-1",
          name: "Push hook",
          minutesAgo: 1,
          webhook: true,
        }),
      ];
    }
    const { user, urlParam } = renderPage(
      `activeTab=triggers&activeItem=${activeItem}`,
    );
    expect(
      await screen.findByRole("tab", { name: /^Triggers\b/ }),
    ).toBeDefined();

    await deleteThrough(user, "trg-1");

    await waitFor(() => expect(urlParam("activeTab")).toBe("runs"));
    await waitFor(() => expect(urlParam("activeItem")).toBe("run-a"));
    expect(await screen.findByText(RUN_OUTPUT)).toBeDefined();
    expect(screen.queryByRole("tab", { name: /^Triggers\b/ })).toBeNull();
  },
);

test("deleting the last webhook trigger through its view keeps the Triggers tab while a trigger workflow is left", async () => {
  triggerAgentsFlag.on = true;
  listedPresets = [
    makePreset({
      id: "trg-1",
      name: "Push hook",
      minutesAgo: 1,
      webhook: true,
    }),
  ];
  listedTriggerAgents = [
    makeTriggerAgent("ta-1", "Inbox watcher", "Watches the inbox"),
  ];
  const { user, urlParam } = renderPage(
    "activeTab=triggers&activeItem=preset:trg-1",
  );
  expect(await screen.findByDisplayValue("Push hook")).toBeDefined();

  await deleteWebhookTrigger["its view"](user, "trg-1");

  await waitFor(() => expect(urlParam("activeItem")).toBe("agent:ta-1"));
  expect(urlParam("activeTab")).toBe("triggers");
  expect(await screen.findByText("Watches the inbox")).toBeDefined();
  expect(screen.queryByText("Push hook")).toBeNull();
});

test("selecting another template while the selected template's delete is in flight keeps the new selection", async () => {
  let releaseDelete = () => {};
  const deleteHeld = new Promise<void>((resolve) => {
    releaseDelete = resolve;
  });
  server.use(
    getDeleteV2DeleteAPresetMockHandler(async ({ params }) => {
      await deleteHeld;
      listedPresets = listedPresets.filter(({ id }) => id !== params.presetId);
    }),
  );
  listedPresets = [
    makePreset({ id: "tpl-1", name: "Morning digest", minutesAgo: 1 }),
    makePreset({ id: "tpl-2", name: "Evening digest", minutesAgo: 2 }),
    makePreset({ id: "tpl-3", name: "Weekly digest", minutesAgo: 3 }),
  ];

  try {
    const { user, urlParam } = renderPage(
      "activeTab=templates&activeItem=tpl-2",
    );
    expect(await screen.findByDisplayValue("Evening digest")).toBeDefined();

    await deleteTemplate["its sidebar row"](user, "tpl-2");
    await user.keyboard("{Escape}");
    // Not tpl-1: the first template is what an emptied selection falls back to.
    await user.click(screen.getByTestId("library-template-item-tpl-3"));
    expect(await screen.findByDisplayValue("Weekly digest")).toBeDefined();

    releaseDelete();

    await waitFor(() =>
      expect(screen.queryByTestId("library-template-item-tpl-2")).toBeNull(),
    );
    expect(toast).toHaveBeenCalledWith({ title: "Template deleted" });
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
    expect(screen.getByDisplayValue("Weekly digest")).toBeDefined();
    expect(urlParam("activeItem")).toBe("tpl-3");
  } finally {
    releaseDelete();
  }
});
