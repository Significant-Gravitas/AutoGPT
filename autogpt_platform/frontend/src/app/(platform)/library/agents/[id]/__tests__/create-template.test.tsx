import { getGetV2GetSpecificBlocksMockHandler } from "@/app/api/__generated__/endpoints/default/default.msw";
import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1ListGraphExecutionsMockHandler,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2ListPresetsMockHandler,
  getPostV2CreateANewPresetMockHandler,
} from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
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

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

const GRAPH_ID = "graph-1";
const TEMPLATE_NAME = "Morning digest";
const TEMPLATE_DESCRIPTION = "Every weekday before standup";

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
  started_at: new Date(),
  ended_at: new Date(),
  stats: { node_exec_count: 1, node_error_count: 0 },
};

let createdTemplates: string[] = [];

beforeEach(() => {
  toast.mockClear();
  createdTemplates = [];
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
    getGetV1ListGraphExecutionsMockHandler(() => ({
      executions: [run],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    })),
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...run,
      inputs: {},
      outputs: { result: ["Output of run-a"] },
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
    getGetV2GetSpecificBlocksMockHandler([]),
  );
});

function answerCreateWith(status: 200 | 500) {
  server.use(
    getPostV2CreateANewPresetMockHandler(async ({ request }) => {
      const body = (await request.json()) as { name: string };
      createdTemplates.push(body.name);
      if (status !== 200) {
        throw HttpResponse.json({ detail: "Server error" }, { status });
      }
      return { id: "tpl-1", name: body.name } as LibraryAgentPreset;
    }),
  );
}

async function renderWithRunSelected() {
  const user = userEvent.setup();
  render(
    <NuqsTestingAdapter
      hasMemory
      searchParams="activeTab=runs&activeItem=run-a"
    >
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  expect(await screen.findByText("Output of run-a")).toBeDefined();
  return user;
}

type User = Awaited<ReturnType<typeof renderWithRunSelected>>;

async function openFromSidebarMenu(user: User) {
  const row = await screen.findByTestId("library-run-item-run-a");
  await user.click(within(row).getByRole("button", { name: "More actions" }));
  await user.click(
    await screen.findByRole("menuitem", { name: "Save as template" }),
  );
}

async function openFromRunView(user: User) {
  await user.click(
    screen.getByRole("button", { name: "Save task as template" }),
  );
}

test.each([
  ["the sidebar's menu", openFromSidebarMenu],
  ["the run view", openFromRunView],
])(
  "a template that fails to save from %s keeps the dialog and what was typed, and saves on a retry",
  async (_, openDialog) => {
    answerCreateWith(500);
    const user = await renderWithRunSelected();

    await openDialog(user);
    const dialog = await screen.findByRole("dialog", {
      name: "Create Template",
    });
    await user.type(within(dialog).getByLabelText("Name"), TEMPLATE_NAME);
    await user.type(
      within(dialog).getByLabelText("Description"),
      TEMPLATE_DESCRIPTION,
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Create Template" }),
    );

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Failed to create template" }),
      ),
    );
    expect(createdTemplates).toEqual([TEMPLATE_NAME]);
    expect(
      screen.getByRole("dialog", { name: "Create Template" }),
    ).toBeDefined();
    expect(within(dialog).getByLabelText("Name")).toHaveProperty(
      "value",
      TEMPLATE_NAME,
    );
    expect(within(dialog).getByLabelText("Description")).toHaveProperty(
      "value",
      TEMPLATE_DESCRIPTION,
    );

    answerCreateWith(200);
    await user.click(
      within(dialog).getByRole("button", { name: "Create Template" }),
    );

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({ title: "Template created" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Create Template" }),
      ).toBeNull(),
    );
    expect(createdTemplates).toEqual([TEMPLATE_NAME, TEMPLATE_NAME]);
  },
);

test.each([
  ["the sidebar's menu", openFromSidebarMenu],
  ["the run view", openFromRunView],
])("a template saved from %s closes the dialog", async (_, openDialog) => {
  answerCreateWith(200);
  const user = await renderWithRunSelected();

  await openDialog(user);
  const dialog = await screen.findByRole("dialog", {
    name: "Create Template",
  });
  await user.type(within(dialog).getByLabelText("Name"), TEMPLATE_NAME);
  await user.click(
    within(dialog).getByRole("button", { name: "Create Template" }),
  );

  await waitFor(() =>
    expect(toast).toHaveBeenCalledWith({ title: "Template created" }),
  );
  await waitFor(() =>
    expect(
      screen.queryByRole("dialog", { name: "Create Template" }),
    ).toBeNull(),
  );
  expect(createdTemplates).toEqual([TEMPLATE_NAME]);
});
