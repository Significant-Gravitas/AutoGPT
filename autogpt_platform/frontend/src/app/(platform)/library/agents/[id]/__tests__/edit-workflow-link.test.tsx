import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1ListGraphExecutionsMockHandler,
  getGetV1ListGraphExecutionsResponseMock,
  getPostV1ExecuteGraphAgentMockHandler,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { HttpResponse } from "msw";
import type { ReactNode } from "react";
import { beforeEach, expect, test, vi } from "vitest";
import AgentLibraryPage from "../page";

const toast = vi.fn();

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  useToast: () => ({ toast }),
}));

beforeEach(() => {
  toast.mockClear();
});

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

// The page holds the version it loaded. A save in the builder since then made
// a newer one, so a link carrying this version would open the old workflow.
const LOADED_VERSION = 3;

function serveAgentPage({
  withRun = false,
  withSchedule = false,
}: {
  withRun?: boolean;
  withSchedule?: boolean;
}) {
  const run = {
    id: "run-1",
    graph_id: GRAPH_ID,
    graph_version: LOADED_VERSION,
    status: AgentExecutionStatus.COMPLETED,
    started_at: new Date(Date.now() - 60_000),
    ended_at: new Date(),
    is_dry_run: false,
    stats: null,
  };
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: LOADED_VERSION,
        name: "Morning digest",
        input_schema: { type: "object", properties: {} },
        credentials_input_schema: { type: "object", properties: {} },
        has_external_trigger: false,
        trigger_setup_info: null,
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler({
      ...getGetV1ListGraphExecutionsResponseMock(),
      executions: withRun
        ? [
            {
              ...getGetV1ListGraphExecutionsResponseMock().executions[0],
              ...run,
            },
          ]
        : [],
      pagination: {
        total_items: withRun ? 1 : 0,
        total_pages: withRun ? 1 : 0,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run,
      inputs: {},
      outputs: {},
      node_executions: [],
    })),
    getGetV1ListExecutionSchedulesForAGraphMockHandler(
      withSchedule
        ? [
            {
              id: "schedule-1",
              name: "Morning run",
              user_id: "user-1",
              graph_id: GRAPH_ID,
              graph_version: LOADED_VERSION,
              cron: "0 8 * * *",
              timezone: "UTC",
              next_run_time: "2030-01-15T08:00:00Z",
              input_data: {},
              input_credentials: {},
            },
          ]
        : [],
    ),
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

test("Edit workflow on a workflow with no tasks opens its active version in the builder", async () => {
  serveAgentPage({ withRun: false });

  render(
    <NuqsTestingAdapter hasMemory>
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  const link = await screen.findByRole("link", { name: "Edit workflow" });
  expect(link.getAttribute("href")).toBe(`/build?flowID=${GRAPH_ID}`);
});

test("Edit workflow from a task's menu opens the workflow's active version in the builder", async () => {
  serveAgentPage({ withRun: true });

  render(
    <NuqsTestingAdapter
      hasMemory
      searchParams="activeTab=runs&activeItem=run-1"
    >
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  // The task's own link still opens the version that task ran.
  const taskLink = await screen.findByRole("link", {
    name: "View task details",
  });
  expect(taskLink.getAttribute("href")).toBe(
    `/build?flowID=${GRAPH_ID}&flowVersion=${LOADED_VERSION}&flowExecutionID=run-1`,
  );

  const taskActions = taskLink.parentElement as HTMLElement;
  await userEvent.click(
    within(taskActions).getByRole("button", { name: "More actions" }),
  );

  const link = await screen.findByRole("menuitem", { name: "Edit workflow" });
  expect(link.getAttribute("href")).toBe(`/build?flowID=${GRAPH_ID}`);
});

test("the scheduled task's builder link opens the workflow's active version", async () => {
  serveAgentPage({ withSchedule: true });

  render(
    <NuqsTestingAdapter
      hasMemory
      searchParams="activeTab=scheduled&activeItem=schedule-1"
    >
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  const link = await screen.findByRole("link", {
    name: "View scheduled task details",
  });
  expect(link.getAttribute("href")).toBe(`/build?flowID=${GRAPH_ID}`);
});

test("Open in builder in a failed run's error opens the workflow's active version", async () => {
  serveAgentPage({});
  server.use(
    getPostV1ExecuteGraphAgentMockHandler(() => {
      throw HttpResponse.json(
        {
          detail: {
            type: "validation_error",
            message: "Some inputs are missing",
            node_errors: { "node-1": { query: "This field is required" } },
          },
        },
        { status: 400 },
      );
    }),
  );

  render(
    <NuqsTestingAdapter hasMemory>
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Setup your task" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Start Task" }),
  );

  await waitFor(() =>
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Failed to execute workflow" }),
    ),
  );
  const { description } = toast.mock.calls.at(-1)![0] as {
    description: ReactNode;
  };
  render(<>{description}</>);
  const link = screen.getByRole("link", { name: "Open in builder" });
  expect(link.getAttribute("href")).toBe(`/build?flowID=${GRAPH_ID}`);
});
