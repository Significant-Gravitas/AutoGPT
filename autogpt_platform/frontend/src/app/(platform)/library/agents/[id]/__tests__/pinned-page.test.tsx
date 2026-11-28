import {
  getGetV1ListGraphExecutionsMockHandler,
  getGetV1ListGraphExecutionsResponseMock,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { server } from "@/mocks/mock-server";
import { COPILOT_PROMPT_KEYS } from "@/services/copilot/prompt-handoff";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, expect, test, vi } from "vitest";
import { useFollowedRunStore } from "@/app/(platform)/pages/components/Page/store";
import { NewAgentLibraryView } from "../components/NewAgentLibraryView/NewAgentLibraryView";

const router = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ id: "library-agent" }),
  usePathname: () => "/library/agents/library-agent",
  useRouter: () => ({ push: router.push, replace: vi.fn(), refresh: vi.fn() }),
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
  useGetFlag: (flag: string) => flag === "pages",
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

beforeEach(() => {
  useFollowedRunStore.setState({ runs: {} });
});

const PINNED_URL = "/api/proxy/api/pages/pinned/:libraryAgentId";
const SOURCE = { library_agent_id: "library-agent", workflow_name: "Leads" };

const COMPANY_FIELD = {
  input: "company",
  label: "Company name",
  help: null,
  default: null,
  group: null,
  required: true,
  input_schema: { type: "string", title: "company" },
};

function page(overrides: Partial<ResolvedPage> = {}): ResolvedPage {
  return {
    id: "page-1",
    title: "Lead research",
    revision: 1,
    pinned_library_agent_id: "library-agent",
    chat_session_id: "chat-1",
    has_previous: false,
    sections: [
      { kind: "text", index: 0, heading: null, body: "Finds leads." },
      {
        kind: "run_form",
        index: 1,
        ...SOURCE,
        fields: [COMPANY_FIELD],
        button_label: "Find leads",
      },
      { kind: "latest_result", index: 2, ...SOURCE, run_id: null },
      { kind: "recent_runs", index: 3, ...SOURCE, runs: [] },
    ],
    ...overrides,
  };
}

function withSection(at: number, section: ResolvedPage["sections"][number]) {
  return page().sections.map((s) => (s.index === at ? section : s));
}

const LEADS_TABLE = {
  kind: "latest_result" as const,
  index: 2,
  ...SOURCE,
  run_id: "ideas-run",
  run_status: AgentExecutionStatus.COMPLETED,
  blocks: [
    {
      type: "table" as const,
      title: null,
      columns: ["Company"],
      rows: [["Acme"], ["Globex"]],
    },
  ],
  row_runs: [
    {
      block_index: 0,
      label: "Research",
      library_agent_id: "deep-agent",
      graph_id: "graph-2",
      graph_version: 3,
      input: "company",
      values: ["Acme", "Globex"],
    },
  ],
};

function serveLibrary(executions: unknown[] = []) {
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: 1,
        name: "Leads",
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

function startedRun(id: string, graphID: string) {
  return getPostV1ExecuteGraphAgentResponseMock({
    id,
    graph_id: graphID,
    graph_version: 1,
    status: AgentExecutionStatus.QUEUED,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    stats: null,
  });
}

function renderView() {
  render(
    <NuqsTestingAdapter hasMemory>
      <NewAgentLibraryView />
    </NuqsTestingAdapter>,
  );
}

test("opens on the pinned page, in its order, where the empty state was", async () => {
  serveLibrary();
  server.use(http.get(PINNED_URL, () => HttpResponse.json(page())));

  renderView();

  const article = await screen.findByTestId("page");
  const text = article.textContent ?? "";
  const order = [
    "Finds leads.",
    "Company name",
    "Latest result",
    "Recent runs",
  ];
  const positions = order.map((part) => text.indexOf(part));
  expect(positions.every((position) => position >= 0)).toBe(true);
  expect(positions).toEqual([...positions].sort((a, b) => a - b));
  expect(screen.queryByRole("button", { name: "Setup your task" })).toBeNull();
});

test("starts a run from the form and follows it until it finishes", async () => {
  const executions = vi.fn();
  const followed: string[][] = [];
  serveLibrary();
  server.use(
    http.post(
      "/api/proxy/api/graphs/:graphID/execute/:version",
      async ({ request }) => {
        executions(await request.json());
        return HttpResponse.json(startedRun("new-run", "graph-1"));
      },
    ),
    http.get(PINNED_URL, ({ request }) => {
      const runIDs = new URL(request.url).searchParams.getAll("run_id");
      followed.push(runIDs);
      if (!runIDs.includes("new-run")) return HttpResponse.json(page());
      return HttpResponse.json(
        page({ sections: withSection(2, { ...LEADS_TABLE, row_runs: [] }) }),
      );
    }),
  );

  renderView();

  await userEvent.type(await screen.findByLabelText(/Company name/), "Acme");
  await userEvent.click(screen.getByRole("button", { name: "Find leads" }));

  await waitFor(() => expect(executions).toHaveBeenCalled());
  expect(executions.mock.calls[0][0]).toMatchObject({
    inputs: { company: "Acme" },
  });
  expect(await screen.findByText("Globex")).toBeDefined();
  expect(followed).toContainEqual(["new-run"]);
});

test("a row button runs its workflow with that row's value and follows it", async () => {
  const runs: { url: string; body: unknown }[] = [];
  const followed: string[][] = [];
  serveLibrary();
  server.use(
    http.post(
      "/api/proxy/api/graphs/:graphID/execute/:version",
      async ({ request }) => {
        runs.push({ url: request.url, body: await request.json() });
        return HttpResponse.json(startedRun("deep-run", "graph-2"));
      },
    ),
    http.get(PINNED_URL, ({ request }) => {
      followed.push(new URL(request.url).searchParams.getAll("run_id"));
      return HttpResponse.json(page({ sections: withSection(2, LEADS_TABLE) }));
    }),
  );

  renderView();

  await userEvent.click(
    await screen.findByRole("button", { name: "Research: Globex" }),
  );

  await waitFor(() => expect(runs).toHaveLength(1));
  expect(new URL(runs[0].url).pathname).toBe(
    "/api/proxy/api/graphs/graph-2/execute/3",
  );
  expect(runs[0].body).toMatchObject({ inputs: { company: "Globex" } });
  await waitFor(() => expect(followed).toContainEqual(["deep-run"]));
});

test("a stale section says so, and asking Otto reopens the page's chat", async () => {
  serveLibrary();
  server.use(
    http.get(PINNED_URL, () =>
      HttpResponse.json(
        page({
          sections: withSection(1, {
            kind: "notice",
            index: 1,
            section_kind: "run_form",
            reason: "stale",
            message:
              "This section no longer fits its workflow. Ask Otto to fix it.",
          }),
        }),
      ),
    ),
  );

  renderView();

  await userEvent.click(
    await screen.findByRole("button", { name: "Ask Otto to fix it" }),
  );

  expect(router.push).toHaveBeenCalledWith("/copilot?sessionId=chat-1");
  expect(sessionStorage.getItem(COPILOT_PROMPT_KEYS.prompt)).toContain(
    'On my page "Lead research" (page id page-1), change section 2',
  );
  expect(sessionStorage.getItem(COPILOT_PROMPT_KEYS.autosubmit)).toBeNull();
});

test("keeps the standard view when no page is pinned to the workflow", async () => {
  serveLibrary();
  server.use(
    http.get(PINNED_URL, () =>
      HttpResponse.json({ detail: "Not found" }, { status: 404 }),
    ),
  );

  renderView();

  expect(
    await screen.findByRole("button", { name: "Setup your task" }),
  ).toBeDefined();
  expect(screen.queryByTestId("page")).toBeNull();
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
    http.get(PINNED_URL, () => HttpResponse.json(page())),
    http.get("/api/proxy/api/graphs/:graphID/executions/:runID", () => {
      details();
      return HttpResponse.json({}, { status: 404 });
    }),
  );

  renderView();

  expect(await screen.findByTestId("page")).toBeDefined();
  expect(details).not.toHaveBeenCalled();
});

test("moving and removing sections and unpinning reach their routes", async () => {
  const edits: unknown[] = [];
  const removed: string[] = [];
  let pinned = true;
  serveLibrary();
  server.use(
    http.get(PINNED_URL, () =>
      pinned
        ? HttpResponse.json(page())
        : HttpResponse.json({ detail: "Not found" }, { status: 404 }),
    ),
    http.patch("/api/proxy/api/pages/:pageID", async ({ request }) => {
      const body = (await request.json()) as { unpin?: boolean };
      edits.push(body);
      if (body.unpin) pinned = false;
      return HttpResponse.json(page());
    }),
    http.delete(
      "/api/proxy/api/pages/:pageID/sections/:index",
      ({ params }) => {
        removed.push(String(params.index));
        return HttpResponse.json(page());
      },
    ),
  );

  renderView();

  await userEvent.click(
    await screen.findByRole("button", {
      name: "Move the Recent runs · Leads section up",
    }),
  );
  await waitFor(() => expect(edits).toEqual([{ order: [0, 1, 3, 2] }]));
  await userEvent.click(
    screen.getByRole("button", {
      name: "Remove the Recent runs · Leads section",
    }),
  );
  await waitFor(() => expect(removed).toEqual(["3"]));
  await userEvent.click(
    screen.getByRole("button", { name: "Use the standard view" }),
  );

  expect(
    await screen.findByRole("button", { name: "Setup your task" }),
  ).toBeDefined();
  expect(edits).toContainEqual({ unpin: true });
});

test("shows when a scheduled workflow runs next", async () => {
  serveLibrary();
  server.use(
    http.get(PINNED_URL, () =>
      HttpResponse.json(
        page({
          sections: withSection(3, {
            kind: "schedule",
            index: 3,
            ...SOURCE,
            is_scheduled: true,
            next_run: "2026-11-23T09:00:00Z",
            unavailable: false,
          }),
        }),
      ),
    ),
  );

  renderView();

  expect(
    await screen.findByText(/Runs on a schedule\. Next run: Nov 23, 2026/),
  ).toBeDefined();
});
