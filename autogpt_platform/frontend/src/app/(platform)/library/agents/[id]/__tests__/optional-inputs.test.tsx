import {
  getGetV1ListGraphExecutionsMockHandler,
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
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
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

const EMPTY_PAGE = {
  total_items: 0,
  total_pages: 0,
  current_page: 1,
  page_size: 20,
};

// Pydantic writes an Optional[...] input as anyOf [the type, null], and the
// options and item type live on that anyOf member, not on the input itself.
const OPTIONAL_INPUTS = {
  topic: {
    title: "Topic",
    anyOf: [{ type: "string", enum: ["news", "sports"] }, { type: "null" }],
  },
  // AgentTableInputBlock.Input.generate_schema() for column_headers
  // ["player", "points"]: the columns are on the field, beside an anyOf member
  // that has none.
  scores: {
    title: "Scores",
    anyOf: [
      { type: "array", items: { type: "object", additionalProperties: true } },
      { type: "null" },
    ],
    default: null,
    type: "array",
    format: "table",
    items: {
      type: "object",
      properties: { player: { type: "string" }, points: { type: "string" } },
    },
  },
  tags: {
    title: "Tags",
    anyOf: [{ type: "array", items: { type: "string" } }, { type: "null" }],
  },
  limit: { title: "Limit", anyOf: [{ type: "integer" }, { type: "null" }] },
};

test("the run form offers each optional input and sends what the user picked", async () => {
  const runInputs = vi.fn();
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: 1,
        name: "Digest agent",
        input_schema: { type: "object", properties: OPTIONAL_INPUTS },
        credentials_input_schema: { type: "object", properties: {} },
        has_external_trigger: false,
        trigger_setup_info: null,
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
      presets: [],
      pagination: { ...EMPTY_PAGE, page_size: 100 },
    }),
    getPostV1ExecuteGraphAgentMockHandler(async ({ request }) => {
      runInputs(((await request.json()) as { inputs: unknown }).inputs);
      return getPostV1ExecuteGraphAgentResponseMock({ id: "new-run" });
    }),
  );

  render(
    <NuqsTestingAdapter hasMemory>
      <NewAgentLibraryView />
    </NuqsTestingAdapter>,
  );
  fireEvent.click(
    await screen.findByRole("button", { name: "Setup your task" }),
  );

  fireEvent.click(await screen.findByRole("combobox", { name: "Topic" }));
  fireEvent.click(await screen.findByRole("option", { name: "sports" }));
  // The table offers the agent's own columns, not the "Column 1" placeholders,
  // so the rows reach the agent keyed the way it reads them.
  expect(screen.getByRole("columnheader", { name: "player" })).toBeDefined();
  expect(screen.getByRole("columnheader", { name: "points" })).toBeDefined();
  fireEvent.click(screen.getByRole("button", { name: "Add Row" }));
  fireEvent.change(screen.getByPlaceholderText("Enter player"), {
    target: { value: "Ada" },
  });
  expect(screen.getByText("Tags")).toBeDefined();
  expect(screen.getByText("Limit")).toBeDefined();

  fireEvent.click(screen.getByRole("button", { name: "Start Task" }));

  await waitFor(() =>
    expect(runInputs).toHaveBeenCalledWith(
      expect.objectContaining({
        topic: "sports",
        scores: [{ player: "Ada", points: "" }],
      }),
    ),
  );
});
