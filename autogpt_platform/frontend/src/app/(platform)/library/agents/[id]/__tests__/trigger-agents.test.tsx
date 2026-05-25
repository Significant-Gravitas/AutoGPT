import {
  getDeleteV2DeleteLibraryAgentMockHandler,
  getDeleteV2DeleteLibraryAgentMockHandler422,
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
  getGetV2ListTriggerAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListGraphExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { NewAgentLibraryView } from "../components/NewAgentLibraryView/NewAgentLibraryView";

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

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
    supabase: {},
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
      getGetV2ListPresetsMockHandler({
        presets: [
          {
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
          },
        ],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 100,
        },
      }),
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
      getGetV2ListPresetsMockHandler({
        presets: [
          {
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
          },
        ],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 100,
        },
      }),
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
