import {
  getGetV1ListGraphExecutionsMockHandler,
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
  getGetV2ListTriggerAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2GetASpecificPresetMockHandler,
  getGetV2ListPresetsMockHandler,
  getPatchV2UpdateAnExistingPresetMockHandler,
  getPostV2ExecuteAPresetMockHandler,
  getPostV2ExecuteAPresetResponseMock,
} from "@/app/api/__generated__/endpoints/presets/presets.msw";
import {
  getGetV1ListExecutionSchedulesForAGraphMockHandler,
  getPostV1CreateExecutionScheduleMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, expect, test, vi } from "vitest";
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
const PICKED = "2026-03-15";
// A triggered preset keeps the trigger node's config under a key like this.
const TRIGGER_MASK = "_node_input_mask_trigger";

// What the run form sent for 15 March before #17305: local midnight, as a UTC
// timestamp. The Date input block refuses any but the UTC one.
const LOCAL_MIDNIGHT: Record<string, string> = {
  UTC: "2026-03-15T00:00:00.000Z",
  "America/Los_Angeles": "2026-03-15T07:00:00.000Z",
  "Pacific/Auckland": "2026-03-14T11:00:00.000Z",
};

// Dates stored that way: 15 March picked in Los Angeles and in Auckland, each
// read back where it was picked, and 15 March picked in UTC read back in Los
// Angeles, which the backend reads as 15 March.
const SAVED_DATES = [
  { saved: LOCAL_MIDNIGHT["America/Los_Angeles"], zone: "America/Los_Angeles" },
  { saved: LOCAL_MIDNIGHT["Pacific/Auckland"], zone: "Pacific/Auckland" },
  { saved: LOCAL_MIDNIGHT.UTC, zone: "America/Los_Angeles" },
];

// Deleting TZ leaves Node in the last zone set, so restore the zone by name.
const START_ZONE =
  process.env.TZ ?? Intl.DateTimeFormat().resolvedOptions().timeZone;
afterEach(() => {
  process.env.TZ = START_ZONE;
});

// AgentDateInputBlock's input, as the graph's input schema has it.
const DATE_INPUT = {
  title: "Date",
  type: "string",
  format: "date",
  default: null,
};

const EMPTY_PAGE = {
  total_items: 0,
  total_pages: 0,
  current_page: 1,
  page_size: 20,
};

function mockLibrary({
  dateRequired = false,
  presets = [],
  dateInTriggerConfig = false,
}: {
  dateRequired?: boolean;
  presets?: LibraryAgentPreset[];
  // The date is the trigger block's config, not a graph input.
  dateInTriggerConfig?: boolean;
} = {}) {
  const sent = {
    task: vi.fn(),
    templateTask: vi.fn(),
    savedPreset: vi.fn(),
    schedule: vi.fn(),
  };
  const withDate = {
    type: "object",
    properties: { date: DATE_INPUT },
    required: dateRequired ? ["date"] : [],
  };
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Trip planner",
        input_schema: dateInTriggerConfig
          ? { type: "object", properties: {}, required: [] }
          : withDate,
        credentials_input_schema: { type: "object", properties: {} },
        has_external_trigger: dateInTriggerConfig,
        trigger_setup_info: dateInTriggerConfig
          ? {
              provider: "generic_webhook",
              config_schema: withDate,
              credentials_input_name: null,
            }
          : null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
        recommended_schedule_cron: null,
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler({
      executions: [],
      pagination: EMPTY_PAGE,
    }),
    getGetV1ListExecutionSchedulesForAGraphMockHandler([]),
    getGetV2ListPresetsMockHandler({
      presets,
      pagination: {
        ...EMPTY_PAGE,
        total_items: presets.length,
        total_pages: 1,
        page_size: 100,
      },
    }),
    getGetV2GetASpecificPresetMockHandler(({ params }) => {
      const preset = presets.find(({ id }) => id === params.presetId);
      if (!preset) {
        throw HttpResponse.json({ detail: "Not found" }, { status: 404 });
      }
      return preset;
    }),
    getGetV2ListTriggerAgentsMockHandler([]),
    getPostV1ExecuteGraphAgentMockHandler(async ({ request }) => {
      sent.task(((await request.json()) as { inputs?: unknown }).inputs);
      return getPostV1ExecuteGraphAgentResponseMock({ id: "new-run" });
    }),
    getPostV2ExecuteAPresetMockHandler(async ({ request }) => {
      sent.templateTask(
        ((await request.json()) as { inputs?: unknown }).inputs,
      );
      return getPostV2ExecuteAPresetResponseMock({ id: "new-run" });
    }),
    getPatchV2UpdateAnExistingPresetMockHandler(async ({ request }) => {
      sent.savedPreset(((await request.json()) as { inputs?: unknown }).inputs);
      return presets[0];
    }),
    getPostV1CreateExecutionScheduleMockHandler(async ({ request }) => {
      sent.schedule(((await request.json()) as { inputs?: unknown }).inputs);
      return getPostV1CreateExecutionScheduleResponseMock({
        id: "new-schedule",
        graph_id: GRAPH_ID,
      });
    }),
  );
  return sent;
}

function makePreset(
  inputs: Record<string, unknown>,
  { webhook = false }: { webhook?: boolean } = {},
): LibraryAgentPreset {
  const at = new Date("2026-03-01T12:00:00Z");
  return {
    id: webhook ? "trg-1" : "tpl-1",
    name: webhook ? "Trip hook" : "Trip template",
    description: "",
    graph_id: GRAPH_ID,
    graph_version: 1,
    inputs,
    credentials: {},
    is_active: true,
    user_id: "user-1",
    created_at: at,
    updated_at: at,
    webhook_id: webhook ? "webhook-1" : null,
    webhook: null,
  };
}

function renderPage(searchParams = "") {
  const user = userEvent.setup();
  render(
    <NuqsTestingAdapter hasMemory searchParams={searchParams}>
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  return user;
}

async function openNewTaskForm(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    await screen.findByRole("button", { name: "Setup your task" }),
  );
  return screen.findByRole("dialog");
}

test("a task whose workflow needs a date can start once the date is picked, and gets the day picked", async () => {
  const sent = mockLibrary({ dateRequired: true });
  const user = renderPage();
  const dialog = await openNewTaskForm(user);
  const start = within(dialog).getByRole("button", { name: "Start Task" });
  expect(start).toHaveProperty("disabled", true);

  fireEvent.change(await within(dialog).findByTestId("agent-input-date"), {
    target: { value: PICKED },
  });

  await waitFor(() => expect(start).toHaveProperty("disabled", false));
  expect(
    within(dialog).getByRole("button", { name: "Schedule Task" }),
  ).toHaveProperty("disabled", false);
  await user.click(start);
  await waitFor(() => expect(sent.task).toHaveBeenCalledWith({ date: PICKED }));
});

test("clearing a required date disables Start Task and Schedule Task again", async () => {
  mockLibrary({ dateRequired: true });
  const user = renderPage();
  const dialog = await openNewTaskForm(user);
  const field = await within(dialog).findByTestId("agent-input-date");
  fireEvent.change(field, { target: { value: PICKED } });
  await waitFor(() =>
    expect(
      within(dialog).getByRole("button", { name: "Start Task" }),
    ).toHaveProperty("disabled", false),
  );

  fireEvent.change(field, { target: { value: "" } });

  await waitFor(() =>
    expect(
      within(dialog).getByRole("button", { name: "Start Task" }),
    ).toHaveProperty("disabled", true),
  );
  expect(
    within(dialog).getByRole("button", { name: "Schedule Task" }),
  ).toHaveProperty("disabled", true);
});

test("a schedule made from the run form in Pacific/Auckland gets the day picked", async () => {
  process.env.TZ = "Pacific/Auckland";
  const sent = mockLibrary({ dateRequired: true });
  const user = renderPage();
  const dialog = await openNewTaskForm(user);
  fireEvent.change(await within(dialog).findByTestId("agent-input-date"), {
    target: { value: PICKED },
  });
  await waitFor(() =>
    expect(
      within(dialog).getByRole("button", { name: "Schedule Task" }),
    ).toHaveProperty("disabled", false),
  );
  await user.click(
    within(dialog).getByRole("button", { name: "Schedule Task" }),
  );

  const scheduleDialog = await screen.findByRole("dialog", {
    name: "Schedule Run",
  });
  await user.click(
    within(scheduleDialog).getByRole("button", { name: "Schedule" }),
  );

  await waitFor(() =>
    expect(sent.schedule).toHaveBeenCalledWith({ date: PICKED }),
  );
  // Until the create's answer is handled, so the test leaves nothing open.
  await waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "Schedule Run" })).toBeNull(),
  );
});

test.each(Object.keys(LOCAL_MIDNIGHT))(
  "a date picked in %s reaches the task as the day picked",
  async (zone) => {
    process.env.TZ = zone;
    // The test runs in that zone: the day picked starts at this instant.
    expect(new Date(2026, 2, 15).toISOString()).toBe(LOCAL_MIDNIGHT[zone]);
    // Optional, so the task can start whether or not a picked date counts as
    // filled (#17304): this checks only what is sent.
    const sent = mockLibrary();
    const user = renderPage();
    const dialog = await openNewTaskForm(user);

    const field = await within(dialog).findByTestId("agent-input-date");
    fireEvent.change(field, { target: { value: PICKED } });
    expect(field).toHaveProperty("value", PICKED);
    await user.click(
      within(dialog).getByRole("button", { name: "Start Task" }),
    );

    await waitFor(() =>
      expect(sent.task).toHaveBeenCalledWith({ date: PICKED }),
    );
  },
);

test.each(SAVED_DATES)(
  "a template's date stored as $saved shows in $zone as the day picked, and a task started from the template gets that day",
  async ({ saved, zone }) => {
    process.env.TZ = zone;
    const sent = mockLibrary({
      dateRequired: true,
      presets: [makePreset({ date: saved })],
    });
    const user = renderPage("activeTab=templates&activeItem=tpl-1");

    expect(await screen.findByDisplayValue(PICKED)).toHaveProperty(
      "type",
      "date",
    );
    await user.click(
      screen.getByRole("button", { name: "Start task from template" }),
    );

    await waitFor(() =>
      expect(sent.templateTask).toHaveBeenCalledWith({ date: PICKED }),
    );
  },
);

test.each(SAVED_DATES)(
  "the run form opened on a template shows its date stored as $saved as the day picked in $zone, and sends that day",
  async ({ saved, zone }) => {
    process.env.TZ = zone;
    const sent = mockLibrary({
      dateRequired: true,
      presets: [makePreset({ date: saved })],
    });
    const user = renderPage("activeTab=templates&activeItem=tpl-1");
    await screen.findByDisplayValue("Trip template");
    const open = screen.getByRole("button", { name: "New workflow task" });
    await waitFor(() => expect(open).toHaveProperty("disabled", false));
    await user.click(open);

    const dialog = await screen.findByRole("dialog");
    expect(
      await within(dialog).findByTestId("agent-input-date"),
    ).toHaveProperty("value", PICKED);
    await user.click(
      within(dialog).getByRole("button", { name: "Start Task" }),
    );

    await waitFor(() =>
      expect(sent.task).toHaveBeenCalledWith({ date: PICKED }),
    );
  },
);

test.each(["2026-02-30", "soon"])(
  "the run form opened on a template whose date %s can't be shown starts with it empty, and waits for a date",
  async (stored) => {
    const sent = mockLibrary({
      dateRequired: true,
      presets: [makePreset({ date: stored })],
    });
    const user = renderPage("activeTab=templates&activeItem=tpl-1");
    await screen.findByDisplayValue("Trip template");
    const open = screen.getByRole("button", { name: "New workflow task" });
    await waitFor(() => expect(open).toHaveProperty("disabled", false));
    await user.click(open);

    const dialog = await screen.findByRole("dialog");
    const field = await within(dialog).findByTestId("agent-input-date");
    expect(field).toHaveProperty("value", "");
    const startTask = () =>
      within(dialog).getByRole("button", { name: "Start Task" });
    await waitFor(() => expect(startTask()).toHaveProperty("disabled", true));

    fireEvent.change(field, { target: { value: PICKED } });
    await waitFor(() => expect(startTask()).toHaveProperty("disabled", false));
    await user.click(startTask());
    await waitFor(() =>
      expect(sent.task).toHaveBeenCalledWith({ date: PICKED }),
    );
  },
);

test.each(["2026-02-30", "soon"])(
  "a template whose date %s can't be shown clears it, whether a task is started from it or it is saved",
  async (stored) => {
    const sent = mockLibrary({ presets: [makePreset({ date: stored })] });
    const user = renderPage("activeTab=templates&activeItem=tpl-1");
    await screen.findByDisplayValue("Trip template");

    await user.click(screen.getByRole("button", { name: "Save changes" }));
    await waitFor(() =>
      expect(sent.savedPreset).toHaveBeenCalledWith({ date: null }),
    );

    // Starting from the template merges what it sends over what it stores,
    // so the date must be sent as null, not left out. (Starting opens the
    // new run, so this goes last.)
    await user.click(
      screen.getByRole("button", { name: "Start task from template" }),
    );
    await waitFor(() =>
      expect(sent.templateTask).toHaveBeenCalledWith({ date: null }),
    );
  },
);

test("a webhook trigger's date stored as a timestamp shows as the day picked, and saving the trigger stores that day", async () => {
  process.env.TZ = "Pacific/Auckland";
  const sent = mockLibrary({
    dateRequired: true,
    presets: [
      makePreset(
        {
          date: LOCAL_MIDNIGHT["Pacific/Auckland"],
          [TRIGGER_MASK]: { events: "push" },
        },
        { webhook: true },
      ),
    ],
  });
  const user = renderPage("activeTab=triggers&activeItem=preset:trg-1");

  expect(await screen.findByDisplayValue(PICKED)).toHaveProperty(
    "type",
    "date",
  );
  await user.click(screen.getByRole("button", { name: "Save changes" }));

  await waitFor(() =>
    expect(sent.savedPreset).toHaveBeenCalledWith({
      date: PICKED,
      [TRIGGER_MASK]: { events: "push" },
    }),
  );
});

test("a date in a trigger's own config stored as a timestamp shows as the day picked, and saving the trigger stores that day", async () => {
  process.env.TZ = "Pacific/Auckland";
  const sent = mockLibrary({
    dateRequired: true,
    dateInTriggerConfig: true,
    presets: [
      makePreset(
        { [TRIGGER_MASK]: { date: LOCAL_MIDNIGHT["Pacific/Auckland"] } },
        { webhook: true },
      ),
    ],
  });
  const user = renderPage("activeTab=triggers&activeItem=preset:trg-1");

  expect(await screen.findByDisplayValue(PICKED)).toHaveProperty(
    "type",
    "date",
  );
  await user.click(screen.getByRole("button", { name: "Save changes" }));

  await waitFor(() =>
    expect(sent.savedPreset).toHaveBeenCalledWith({
      [TRIGGER_MASK]: { date: PICKED },
    }),
  );
});
