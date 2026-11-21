import {
  getGetV1ListGraphExecutionsMockHandler,
  getGetV1ListGraphExecutionsResponseMock,
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { ResolvedWorkflowPage } from "@/app/api/__generated__/models/resolvedWorkflowPage";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
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
  useGetFlag: (flag: string) => flag === "workflow-pages",
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

const PAGE_URL = "/api/proxy/api/workflow-pages/:libraryAgentId";

const COMPANY_FIELD = {
  input: "company",
  label: "Company name",
  help: null,
  default: null,
  group: null,
  required: true,
  input_schema: { type: "string", title: "company" },
};

function page(
  overrides: Partial<ResolvedWorkflowPage> = {},
): ResolvedWorkflowPage {
  return {
    library_agent_id: "library-agent",
    graph_id: "graph-1",
    title: "Lead research",
    sections: [
      {
        kind: "text",
        index: 0,
        heading: null,
        body: "Finds leads for one company.",
      },
      {
        kind: "run_form",
        index: 1,
        fields: [COMPANY_FIELD],
        button_label: "Find leads",
      },
      {
        kind: "latest_result",
        index: 2,
        run_id: null,
        blocks: [],
        other_outputs: {},
      },
      { kind: "recent_runs", index: 3, runs: [] },
    ],
    left_out: [],
    has_previous: false,
    ...overrides,
  };
}

function finishedResult(runID: string) {
  return {
    kind: "latest_result" as const,
    index: 2,
    run_id: runID,
    run_status: AgentExecutionStatus.COMPLETED,
    blocks: [
      {
        type: "stats" as const,
        items: [{ label: "Leads found", value: "12" }],
      },
    ],
    other_outputs: {},
  };
}

function serveLibrary(executions: unknown[] = []) {
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: 1,
        name: "Lead research",
        input_schema: {
          type: "object",
          properties: { company: { type: "string", title: "company" } },
          required: ["company"],
        },
        credentials_input_schema: { type: "object", properties: {} },
        has_external_trigger: false,
        trigger_setup_info: null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
        recommended_schedule_cron: null,
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler({
      executions: executions as never[],
      pagination: {
        total_items: executions.length,
        total_pages: executions.length ? 1 : 0,
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
  );
}

function renderView() {
  render(
    <NuqsTestingAdapter hasMemory>
      <NewAgentLibraryView />
    </NuqsTestingAdapter>,
  );
}

test("shows the workflow's page, in its order, where the empty state was", async () => {
  serveLibrary();
  server.use(http.get(PAGE_URL, () => HttpResponse.json(page())));

  renderView();

  const article = await screen.findByTestId("workflow-page");
  const text = article.textContent ?? "";
  const order = [
    "Finds leads for one company.",
    "Company name",
    "Latest result",
    "Recent runs",
  ];
  const positions = order.map((part) => text.indexOf(part));
  expect(positions.every((position) => position >= 0)).toBe(true);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));
  expect(screen.queryByRole("button", { name: "Setup your task" })).toBeNull();
});

test("starts a run from the page's form and shows its result once it finishes", async () => {
  const run = getPostV1ExecuteGraphAgentResponseMock({
    id: "new-run",
    graph_id: "graph-1",
    graph_version: 1,
    status: AgentExecutionStatus.QUEUED,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    stats: null,
  });
  const executions = vi.fn();
  const followed: (string | null)[] = [];
  serveLibrary();
  server.use(
    getPostV1ExecuteGraphAgentMockHandler(async ({ request }) => {
      executions(await request.json());
      return run;
    }),
    http.get(PAGE_URL, ({ request }) => {
      const runID = new URL(request.url).searchParams.get("run_id");
      followed.push(runID);
      if (runID !== "new-run") return HttpResponse.json(page());
      const sections = page().sections.map((s) =>
        s.kind === "latest_result" ? finishedResult("new-run") : s,
      );
      return HttpResponse.json(page({ sections }));
    }),
  );

  renderView();

  await userEvent.type(await screen.findByLabelText(/Company name/), "Acme");
  await userEvent.click(screen.getByRole("button", { name: "Find leads" }));

  await waitFor(() => expect(executions).toHaveBeenCalled());
  expect(executions.mock.calls[0][0]).toMatchObject({
    inputs: { company: "Acme" },
  });
  expect(await screen.findByText("Leads found")).toBeDefined();
  expect(await screen.findByText("12")).toBeDefined();
  expect(followed).toContain("new-run");
});

test("says the page is out of date when a section no longer fits", async () => {
  serveLibrary();
  server.use(
    http.get(PAGE_URL, () =>
      HttpResponse.json(
        page({
          left_out: [
            {
              index: 1,
              kind: "run_form",
              problem: "required input 'region' is missing",
            },
          ],
        }),
      ),
    ),
  );

  renderView();

  expect(await screen.findByText(/This page is out of date/)).toBeDefined();
  expect(
    screen.getByRole("button", { name: "Ask Otto to update it" }),
  ).toBeDefined();
});

test("keeps the standard view when the workflow has no page", async () => {
  serveLibrary();
  server.use(
    http.get(PAGE_URL, () =>
      HttpResponse.json({ detail: "Not found" }, { status: 404 }),
    ),
  );

  renderView();

  expect(
    await screen.findByRole("button", { name: "Setup your task" }),
  ).toBeDefined();
  expect(screen.queryByTestId("workflow-page")).toBeNull();
});

test("opens on the page, with no run picked, when the workflow has run before", async () => {
  const details = vi.fn();
  serveLibrary([
    {
      ...getGetV1ListGraphExecutionsResponseMock().executions[0],
      id: "old-run",
      graph_id: "graph-1",
      graph_version: 1,
      status: AgentExecutionStatus.COMPLETED,
      is_dry_run: false,
    },
  ]);
  server.use(
    http.get(PAGE_URL, () => HttpResponse.json(page())),
    http.get("/api/proxy/api/graphs/:graphID/executions/:runID", () => {
      details();
      return HttpResponse.json({}, { status: 404 });
    }),
  );

  renderView();

  expect(await screen.findByTestId("workflow-page")).toBeDefined();
  expect(details).not.toHaveBeenCalled();
});
