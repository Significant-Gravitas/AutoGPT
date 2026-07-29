import {
  getDeleteV2DeleteLibraryAgentMockHandler,
  getDeleteV2DeleteLibraryAgentMockHandler422,
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
  getGetV2ListTriggerAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListGraphExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import {
  getGetV2GetASpecificPresetMockHandler,
  getGetV2GetASpecificPresetResponseMock,
  getGetV2ListPresetsMockHandler,
  getGetV2ListPresetsMockHandler422,
} from "@/app/api/__generated__/endpoints/presets/presets.msw";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { NewAgentLibraryView } from "../components/NewAgentLibraryView/NewAgentLibraryView";
import { PRESETS_PAGE_SIZE } from "../components/NewAgentLibraryView/hooks/useAgentPresetsQuery";

const PARENT_ID = "parent-agent-id";
const PARENT_GRAPH_ID = "parent-graph-id";
const TRIGGER_ID = "trigger-agent-id";
const TRIGGER_GRAPH_ID = "trigger-graph-id";

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useParams: () => ({ id: PARENT_ID }),
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => `/library/agents/${PARENT_ID}`,
    useSearchParams: () => new URLSearchParams(),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast, toasts: [], dismiss: vi.fn() }),
  toast: mockToast,
  useToastOnFail: () => vi.fn(),
}));

// Default to flag ON so the existing tests exercise the full UI; the
// flag-off branch is covered by a dedicated test that overrides this
// per-call.
const mockUseGetFlag = vi.hoisted(() => vi.fn(() => true));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    GENERIC_TRIGGER_AGENTS: "generic-trigger-agents",
  },
  useGetFlag: mockUseGetFlag,
  useFlagStatus: () => ({ enabled: mockUseGetFlag(), ready: true }),
}));

// Per-test render wrapper so we can set the nuqs initial URL state
// (e.g. activeTab=triggers) — Radix tab clicks don't always round-trip
// through the NuqsTestingAdapter within a single sync frame.
function renderWithInitialParams(ui: ReactNode, searchParams = "") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter searchParams={searchParams}>
        <BackendAPIProvider>
          <OnboardingProvider>
            <TooltipProvider>{ui}</TooltipProvider>
          </OnboardingProvider>
        </BackendAPIProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>,
  );
}

function baseHandlers(
  overrides?: Partial<ReturnType<typeof getGetV2GetLibraryAgentResponseMock>>,
) {
  const parentAgent = getGetV2GetLibraryAgentResponseMock({
    id: PARENT_ID,
    graph_id: PARENT_GRAPH_ID,
    name: "Parent Agent",
    description: "The parent",
    is_hidden: false,
    ...overrides,
  });
  return [
    getGetV2GetLibraryAgentMockHandler(parentAgent),
    getGetV1ListGraphExecutionsMockHandler({
      executions: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 1,
        page_size: 20,
      },
    }),
  ];
}

const emptyPresetsHandler = getGetV2ListPresetsMockHandler({
  presets: [],
  pagination: {
    total_items: 0,
    total_pages: 0,
    current_page: 1,
    page_size: 100,
  },
});
const emptySchedulesHandler =
  getGetV1ListExecutionSchedulesForAGraphMockHandler([]);

function makeWebhookPreset(overrides: Partial<LibraryAgentPreset> = {}) {
  return {
    id: "preset-1",
    user_id: "user-1",
    graph_id: PARENT_GRAPH_ID,
    graph_version: 1,
    name: "Webhook Trigger",
    description: "",
    inputs: {},
    credentials: {},
    is_active: true,
    webhook_id: "webhook-1",
    webhook: null,
    created_at: new Date("2026-01-01T00:00:00.000Z"),
    updated_at: new Date("2026-01-01T00:00:00.000Z"),
    ...overrides,
  };
}

function singlePresetListHandler(
  preset: ReturnType<typeof makeWebhookPreset>,
  totalItems = 1,
) {
  return getGetV2ListPresetsMockHandler({
    presets: [preset],
    pagination: {
      total_items: totalItems,
      total_pages: Math.ceil(totalItems / PRESETS_PAGE_SIZE),
      current_page: 1,
      page_size: PRESETS_PAGE_SIZE,
    },
  });
}

describe("Library agent view — trigger agents", () => {
  beforeEach(() => {
    server.resetHandlers();
    mockToast.mockClear();
    mockUseGetFlag.mockReturnValue(true);
  });

  test("hides Triggers tab when there are no trigger agents and no webhook triggers", async () => {
    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
    );

    renderWithInitialParams(<NewAgentLibraryView />);

    await screen.findByText("Parent Agent");
    expect(screen.queryByRole("tab", { name: /triggers/i })).toBeNull();
  });

  test("shows trigger agent in 'Trigger Agents' subsection when one exists", async () => {
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Email Watcher",
      description: "Watches my inbox",
      is_hidden: true,
    });

    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
    );

    renderWithInitialParams(<NewAgentLibraryView />, "activeTab=triggers");

    await screen.findByText("Parent Agent");
    await screen.findByText("Trigger Agents");
    const rows = await screen.findAllByText("Email Watcher");
    expect(rows.length).toBeGreaterThan(0);
    expect(screen.queryByText("Webhook Triggers")).toBeNull();
  });

  test("shows both 'Webhook Triggers' and 'Trigger Agents' subsections when both exist", async () => {
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "RSS Watcher",
      is_hidden: true,
    });

    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
      singlePresetListHandler(makeWebhookPreset()),
    );

    renderWithInitialParams(<NewAgentLibraryView />, "activeTab=triggers");

    await screen.findByText("Parent Agent");
    await screen.findByText("Webhook Triggers");
    await screen.findByText("Trigger Agents");
    expect((await screen.findAllByText("RSS Watcher")).length).toBeGreaterThan(
      0,
    );
    expect(
      (await screen.findAllByText("Webhook Trigger")).length,
    ).toBeGreaterThan(0);
  });

  test("selecting a trigger agent renders its detail view with schedule info", async () => {
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Daily Summary Trigger",
      description: "Runs every morning at 8am",
      is_hidden: true,
    });

    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
      getGetV1ListExecutionSchedulesForAGraphMockHandler([
        {
          id: "sched-1",
          name: "Morning run",
          user_id: "user-1",
          graph_id: TRIGGER_GRAPH_ID,
          graph_version: 1,
          cron: "0 8 * * *",
          timezone: "UTC",
          next_run_time: "2026-05-01T08:00:00.000Z",
          input_data: {},
          input_credentials: {},
        },
      ]),
    );

    // Render with the trigger already selected via URL state — avoids
    // relying on clicks to transition tab + selection together.
    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=${TRIGGER_ID}`,
    );

    // Description shows in the detail card, not the sidebar row
    await screen.findByText("Runs every morning at 8am");
    // Schedule card header + labels
    await screen.findByText("Schedule");
    await screen.findByText("Recurrence");
    await screen.findByText("Next run");
  });

  test("clicking Remove on a trigger agent deletes it via the library-agent delete endpoint", async () => {
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Remove Me",
      is_hidden: true,
    });

    const deleteCalls: string[] = [];
    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
      getDeleteV2DeleteLibraryAgentMockHandler(
        ({ params }: { params: Record<string, string> }) => {
          deleteCalls.push(String(params.libraryAgentId));
          return new Response(null, { status: 204 });
        },
      ),
    );

    // Render with trigger selected so we can use the action button in
    // the side panel (stable role-based query, avoids dropdown complexity).
    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=${TRIGGER_ID}`,
    );

    // Wait until the trigger detail view is fully rendered
    await screen.findByText("Remove Me");

    // Side panel has "Remove trigger" icon button
    const removeButton = await screen.findByRole("button", {
      name: /remove trigger/i,
    });
    fireEvent.click(removeButton);

    // Confirm dialog — the destructive confirmation button
    const confirmButton = await screen.findByRole("button", {
      name: /^remove trigger$/i,
    });
    // There may be two "Remove trigger" buttons now (icon + confirm).
    // Click the one inside the dialog specifically.
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(deleteCalls).toContain(TRIGGER_ID);
    });
    // Must delete the TRIGGER agent, never the parent
    expect(deleteCalls).not.toContain(PARENT_ID);
  });

  test("detail view shows 'No schedule configured' when trigger agent has no schedule", async () => {
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Idle Trigger",
      description: "No schedule yet",
      is_hidden: true,
    });

    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler, // returns [] for any graph
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=${TRIGGER_ID}`,
    );

    await screen.findByText("No schedule yet");
    // Fallback card renders the "No schedule configured" message
    await screen.findByText(/no schedule configured/i);
    // And NOT the recurrence/next-run labels
    expect(screen.queryByText("Recurrence")).toBeNull();
    expect(screen.queryByText("Next run")).toBeNull();
  });

  test("sidebar dropdown Remove deletes the trigger via the shared delete flow", async () => {
    const user = userEvent.setup();
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Dropdown Victim",
      is_hidden: true,
    });

    const deleteCalls: string[] = [];
    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
      getDeleteV2DeleteLibraryAgentMockHandler(
        ({ params }: { params: Record<string, string> }) => {
          deleteCalls.push(String(params.libraryAgentId));
          return new Response(null, { status: 204 });
        },
      ),
    );

    // Render on Triggers tab WITHOUT activeItem so the sidebar dropdown
    // is the path under test (not the side panel).
    renderWithInitialParams(<NewAgentLibraryView />, "activeTab=triggers");

    await screen.findByText("Dropdown Victim");

    // Open the trigger-agent row's dropdown. Other rows may also have
    // "More actions" buttons (webhook triggers) but we only have one
    // trigger-agent row, and the agent-trigger dropdown is the only
    // one rendered here (no webhook presets in this test).
    const moreButton = await screen.findByRole("button", {
      name: /more actions/i,
    });
    await user.click(moreButton);

    // Click "Remove trigger" menu item
    const removeMenuItem = await screen.findByRole("menuitem", {
      name: /remove trigger/i,
    });
    await user.click(removeMenuItem);

    // Confirm in the destructive dialog
    const confirmButton = await screen.findByRole("button", {
      name: /^remove trigger$/i,
    });
    await user.click(confirmButton);

    await waitFor(() => {
      expect(deleteCalls).toContain(TRIGGER_ID);
    });
    expect(deleteCalls).not.toContain(PARENT_ID);
  });

  test("delete error shows a destructive toast via the hook's onError path", async () => {
    const user = userEvent.setup();
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Error Case",
      is_hidden: true,
    });

    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
      // Backend rejects the delete with 422 — exercises the hook's
      // onError path and proves the mutation doesn't crash with an
      // unhandled rejection when the request fails.
      getDeleteV2DeleteLibraryAgentMockHandler422(),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=${TRIGGER_ID}`,
    );

    await screen.findByText("Error Case");

    const removeButton = await screen.findByRole("button", {
      name: /remove trigger/i,
    });
    await user.click(removeButton);

    const confirmButton = await screen.findByRole("button", {
      name: /^remove trigger$/i,
    });
    await user.click(confirmButton);

    // The hook's onError called toast() with the failure title —
    // a success would produce a different title ("Trigger removed").
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: expect.stringMatching(/failed to remove trigger/i),
          variant: "destructive",
        }),
      );
    });
  });

  test("selecting a webhook trigger renders its preset detail view", async () => {
    const webhookPreset = makeWebhookPreset({
      description: "Fires on webhook",
    });

    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      singlePresetListHandler(webhookPreset),
      getGetV2GetASpecificPresetMockHandler(webhookPreset),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      "activeTab=triggers&activeItem=preset-1",
    );

    await screen.findByText("Trigger Details");
    screen.getByDisplayValue("Webhook Trigger");
  });

  test("agent:-prefixed activeItem renders the trigger agent detail view", async () => {
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Hinted Watcher",
      description: "Selected via type-hinted URL",
      is_hidden: true,
    });

    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=agent:${TRIGGER_ID}`,
    );

    await screen.findByText("Selected via type-hinted URL");
  });

  test("stale trigger id shows a graceful not-found state without fetching a preset", async () => {
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Still Alive",
      is_hidden: true,
    });

    let presetGetCalls = 0;
    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([triggerAgent]),
      // If the view wrongly assumes the unknown id is a preset, this
      // handler fires — the old behavior that produced a 404 error page.
      getGetV2GetASpecificPresetMockHandler(() => {
        presetGetCalls += 1;
        return getGetV2GetASpecificPresetResponseMock();
      }),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      "activeTab=triggers&activeItem=deleted-preset-id",
    );

    await screen.findByText("Trigger not found");
    await screen.findByText(/doesn't exist or is no longer available/i);
    expect(presetGetCalls).toBe(0);
  });

  test("a preset: hint mounts the detail view while the lists are still loading", async () => {
    const webhookPreset = makeWebhookPreset({ name: "Hint Routed Early" });
    let releaseLists!: () => void;
    const listGate = new Promise<void>((resolve) => {
      releaseLists = resolve;
    });

    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler(async () => {
        await listGate;
        return [];
      }),
      getGetV2ListPresetsMockHandler(async () => {
        await listGate;
        return {
          presets: [webhookPreset],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: PRESETS_PAGE_SIZE,
          },
        };
      }),
      getGetV2GetASpecificPresetMockHandler(webhookPreset),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=preset:${webhookPreset.id}`,
    );

    // The detail view renders from the hint alone — both list queries are
    // still gated at this point.
    await screen.findByText("Trigger Details");
    screen.getByDisplayValue("Hint Routed Early");

    releaseLists();
  });

  test("stale selection of the sole trigger keeps its not-found state with a recovery action", async () => {
    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      "activeTab=triggers&activeItem=agent:deleted-trigger-id",
    );

    // Without the activeItem guards, the zero-trigger redirect + zero-item
    // layout replace this with the empty-tasks screen.
    await screen.findByText("Trigger not found");
    const clearButton = await screen.findByRole("button", {
      name: /clear selection/i,
    });

    fireEvent.click(clearButton);
    await waitFor(() => {
      expect(screen.queryByText("Trigger not found")).toBeNull();
    });
  });

  test("a preset beyond the first page resolves from membership once all pages load", async () => {
    const beyondPagePreset = makeWebhookPreset({
      id: "beyond-page-1",
      name: "Beyond Page Trigger",
    });
    const firstPage = Array.from({ length: PRESETS_PAGE_SIZE }, (_, i) =>
      makeWebhookPreset({ id: `page1-preset-${i + 1}`, name: `Trigger ${i}` }),
    );

    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      // 101 presets over two pages; the selected ID lives on page 2. The hook
      // pages through everything, so membership resolves it as a webhook
      // trigger rather than assuming "not-found".
      getGetV2ListPresetsMockHandler((info) => {
        const page = Number(
          new URL(info.request.url).searchParams.get("page") ?? "1",
        );
        return {
          presets: page <= 1 ? firstPage : [beyondPagePreset],
          pagination: {
            total_items: firstPage.length + 1,
            total_pages: 2,
            current_page: page,
            page_size: PRESETS_PAGE_SIZE,
          },
        };
      }),
      getGetV2GetASpecificPresetMockHandler(beyondPagePreset),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      "activeTab=triggers&activeItem=beyond-page-1",
    );

    // The Triggers tab count reflects the full membership (101), not just the
    // first page (100) — this is what fails without eager pagination.
    // Paging through every preset then fetching the detail is a longer async
    // chain than a single-page load, so allow extra time.
    await screen.findByRole(
      "tab",
      { name: /triggers\s*101/i },
      { timeout: 5000 },
    );
    await screen.findByText("Trigger Details", undefined, { timeout: 5000 });
    screen.getByDisplayValue("Beyond Page Trigger");
    expect(screen.queryByText("Trigger not found")).toBeNull();
  });

  test("unknown ID across a fully paginated presets list resolves to not-found without a by-ID fetch", async () => {
    // 150 presets over two pages (page_size 100). The selected ID is on
    // neither page, so once every page has loaded membership is authoritative
    // and the router must land on not-found WITHOUT a throwaway by-ID fetch.
    const firstPage = Array.from({ length: PRESETS_PAGE_SIZE }, (_, i) =>
      makeWebhookPreset({ id: `page1-preset-${i + 1}`, name: `Trigger ${i}` }),
    );
    const secondPage = Array.from({ length: 50 }, (_, i) =>
      makeWebhookPreset({
        id: `page2-preset-${i + 1}`,
        name: `Trigger ${i + 100}`,
      }),
    );
    const totalItems = firstPage.length + secondPage.length;

    let presetGetCalls = 0;
    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      getGetV2ListPresetsMockHandler((info) => {
        const page = Number(
          new URL(info.request.url).searchParams.get("page") ?? "1",
        );
        return {
          presets: page <= 1 ? firstPage : secondPage,
          pagination: {
            total_items: totalItems,
            total_pages: 2,
            current_page: page,
            page_size: PRESETS_PAGE_SIZE,
          },
        };
      }),
      // Fires only if the router wrongly assumes the unknown ID is a preset —
      // the degraded behavior this fix removes.
      getGetV2GetASpecificPresetMockHandler(() => {
        presetGetCalls += 1;
        return getGetV2GetASpecificPresetResponseMock();
      }),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      "activeTab=triggers&activeItem=deleted-preset-id",
    );

    // Paging through every preset before concluding not-found is a longer
    // async chain than a single-page load, so allow extra time.
    await screen.findByText("Trigger not found", undefined, { timeout: 5000 });
    await screen.findByText(/doesn't exist or is no longer available/i);
    expect(presetGetCalls).toBe(0);
  });

  test("an unknown ID beyond the pagination cap falls back to the by-ID preset fetch", async () => {
    // More presets than the eager pagination cap (MAX_PRESET_PAGES * 100 =
    // 2000) can page through: the hook stops at the cap, so `presetsComplete`
    // stays false and list membership is NOT authoritative. An unknown/stale
    // selection must therefore fall back to the graceful by-ID webhook-trigger
    // fetch — the beyond-cap path that the fully-paginated test above
    // deliberately does not exercise (it asserts zero by-ID fetches).
    const cappedPreset = makeWebhookPreset({
      id: "beyond-cap-preset",
      name: "Beyond Cap Trigger",
    });

    let presetGetCalls = 0;
    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      // Report far more presets than the cap can reach, but return one row per
      // page so the test stays light — it's the pagination metadata (not the
      // row count) that keeps `hasNextPage` true past the cap and leaves
      // membership incomplete.
      getGetV2ListPresetsMockHandler((info) => {
        const page = Number(
          new URL(info.request.url).searchParams.get("page") ?? "1",
        );
        return {
          presets: [
            makeWebhookPreset({
              id: `page${page}-preset`,
              name: `Trigger ${page}`,
            }),
          ],
          pagination: {
            total_items: 5000,
            total_pages: 50,
            current_page: page,
            page_size: PRESETS_PAGE_SIZE,
          },
        };
      }),
      // The selected ID lives beyond the paginated window, so with membership
      // incomplete the router resolves it as a webhook trigger and this fires.
      getGetV2GetASpecificPresetMockHandler(() => {
        presetGetCalls += 1;
        return cappedPreset;
      }),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=${cappedPreset.id}`,
    );

    // Paging through the full cap before the by-ID fallback resolves the detail
    // is a long async chain, so allow extra time.
    await screen.findByText("Trigger Details", undefined, { timeout: 8000 });
    screen.getByDisplayValue("Beyond Cap Trigger");
    // The beyond-cap fallback fired the by-ID fetch it's meant to.
    expect(presetGetCalls).toBeGreaterThan(0);
  });

  test("a later presets page failing degrades gracefully instead of blanking the sidebar", async () => {
    // Page 1 loads 100 webhook presets; page 2 fails. The loaded page stays
    // usable — the Triggers list must still render rather than being replaced
    // by a full error card.
    const firstPage = Array.from({ length: PRESETS_PAGE_SIZE }, (_, i) =>
      makeWebhookPreset({ id: `page1-preset-${i + 1}`, name: `Trigger ${i}` }),
    );

    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      http.get(
        "http://localhost:3000/api/proxy/api/library/presets",
        ({ request }) => {
          const page = Number(
            new URL(request.url).searchParams.get("page") ?? "1",
          );
          if (page > 1) {
            return new HttpResponse(JSON.stringify({ detail: "boom" }), {
              status: 422,
              headers: { "Content-Type": "application/json" },
            });
          }
          return HttpResponse.json({
            presets: firstPage,
            pagination: {
              total_items: firstPage.length + 50,
              total_pages: 2,
              current_page: 1,
              page_size: PRESETS_PAGE_SIZE,
            },
          });
        },
      ),
      getGetV2GetASpecificPresetMockHandler(firstPage[0]),
    );

    renderWithInitialParams(<NewAgentLibraryView />, "activeTab=triggers");

    // The sidebar keeps rendering the loaded page-1 triggers instead of being
    // replaced by an error card.
    await screen.findByText("Webhook Triggers", undefined, { timeout: 5000 });
    expect(screen.queryByText(/when retrieving/i)).toBeNull();
  });

  test("webhook trigger deleted between list and detail fetch renders the not-found card", async () => {
    const webhookPreset = makeWebhookPreset({ name: "Just Deleted" });

    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      singlePresetListHandler(webhookPreset),
      // The preset is in the list, but its by-ID detail fetch 404s — the
      // race where it's deleted between the list load and selection.
      http.get(
        "http://localhost:3000/api/proxy/api/library/presets/:presetId",
        () =>
          new HttpResponse(
            JSON.stringify({ detail: "Preset #preset-1 not found" }),
            {
              status: 404,
              headers: { "Content-Type": "application/json" },
            },
          ),
      ),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      `activeTab=triggers&activeItem=preset:${webhookPreset.id}`,
    );

    await screen.findByText("Trigger not found");
    expect(screen.queryByText(/when retrieving/i)).toBeNull();
  });

  test("failed presets fetch shows an error card instead of an endless skeleton", async () => {
    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      getGetV2ListPresetsMockHandler422(),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      "activeTab=triggers&activeItem=some-bare-id",
    );

    await screen.findByText(/when retrieving triggers/i);
  });

  test("templates-tab preset failure still surfaces the page-level error", async () => {
    server.use(
      ...baseHandlers(),
      emptyPresetsHandler,
      emptySchedulesHandler,
      getGetV2ListTriggerAgentsMockHandler([]),
      http.get(
        "http://localhost:3000/api/proxy/api/library/presets/:presetId",
        () =>
          new HttpResponse(
            JSON.stringify({ detail: "Preset #gone-template not found" }),
            {
              status: 404,
              headers: { "Content-Type": "application/json" },
            },
          ),
      ),
    );

    renderWithInitialParams(
      <NewAgentLibraryView />,
      "activeTab=templates&activeItem=gone-template",
    );

    // On the Templates tab the shared preset query's error must still
    // surface — the tab guard only suppresses it elsewhere.
    await screen.findByText(/when retrieving agent/i);
  });

  test("when generic-trigger-agents flag is off, hides 'Trigger Agents' subsection and skips the trigger-agents fetch", async () => {
    mockUseGetFlag.mockReturnValue(false);

    let triggerAgentsCallCount = 0;
    const triggerAgent = getGetV2GetLibraryAgentResponseMock({
      id: TRIGGER_ID,
      graph_id: TRIGGER_GRAPH_ID,
      name: "Hidden Watcher",
      is_hidden: true,
    });

    server.use(
      ...baseHandlers(),
      emptySchedulesHandler,
      // Ensure backend would still serve a trigger agent if asked; we
      // assert the request never fires when the flag is off.
      getGetV2ListTriggerAgentsMockHandler(() => {
        triggerAgentsCallCount += 1;
        return [triggerAgent];
      }),
      // Webhook trigger so the Triggers tab still has reason to exist.
      singlePresetListHandler(makeWebhookPreset()),
    );

    renderWithInitialParams(<NewAgentLibraryView />, "activeTab=triggers");

    await screen.findByText("Parent Agent");
    await screen.findByText("Webhook Triggers");
    // The "Trigger Agents" subsection must not render and the row name
    // must be absent.
    expect(screen.queryByText("Trigger Agents")).toBeNull();
    expect(screen.queryByText("Hidden Watcher")).toBeNull();
    // And the GET .../triggers request never fires.
    expect(triggerAgentsCallCount).toBe(0);
  });
});
