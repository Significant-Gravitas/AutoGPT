import {
  getPostV1ExecuteGraphAgentMockHandler,
  getPostV1ExecuteGraphAgentResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2GetLibraryAgentResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import type { GraphSettings } from "@/app/api/__generated__/models/graphSettings";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, test, vi } from "vitest";
import { RunAgentModal } from "../RunAgentModal";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn(), toasts: [], dismiss: vi.fn() }),
  toast: vi.fn(),
  useToastOnFail: () => vi.fn(),
}));

vi.mock("@/lib/hooks/useBreakpoint", () => ({
  useBreakpoint: () => "lg",
  isLargeScreen: () => true,
}));

beforeAll(() => {
  if (!globalThis.ResizeObserver) {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

beforeEach(() => {
  window.localStorage.clear();
});

// What a workflow saved from the builder gets.
const BUILDER_SETTINGS: GraphSettings = {
  human_in_the_loop_safe_mode: true,
  sensitive_action_safe_mode: false,
};
const ALL_REVIEW_ON: GraphSettings = {
  human_in_the_loop_safe_mode: true,
  sensitive_action_safe_mode: true,
};
const ALL_REVIEW_OFF: GraphSettings = {
  human_in_the_loop_safe_mode: false,
  sensitive_action_safe_mode: false,
};

const SENSITIVE_ACTION_PROMISE =
  "AutoGPT includes safety checks so you'll always have the opportunity to review and approve sensitive actions before they happen.";
const HUMAN_IN_THE_LOOP_PROMISE =
  "This workflow will pause at its human-in-the-loop blocks and wait for your review before continuing.";

interface AgentArgs {
  hasHumanInTheLoop?: boolean;
  hasSensitiveAction?: boolean;
  settings: GraphSettings | undefined;
}

function makeAgent({
  hasHumanInTheLoop = false,
  hasSensitiveAction = false,
  settings,
}: AgentArgs) {
  return getGetV2GetLibraryAgentResponseMock({
    id: "library-agent",
    graph_id: "graph-1",
    graph_version: 1,
    name: "Safety agent",
    has_external_trigger: false,
    trigger_setup_info: null,
    input_schema: { type: "object", properties: {} },
    credentials_input_schema: { type: "object", properties: {} },
    has_human_in_the_loop: hasHumanInTheLoop,
    has_sensitive_action: hasSensitiveAction,
    settings,
  });
}

function trackRuns() {
  const runs = vi.fn(() =>
    getPostV1ExecuteGraphAgentResponseMock({
      id: "new-run",
      graph_id: "graph-1",
    }),
  );
  server.use(getPostV1ExecuteGraphAgentMockHandler(runs));
  return runs;
}

function modal(agent: LibraryAgent) {
  return <RunAgentModal agent={agent} triggerSlot={<button>Open</button>} />;
}

// The trigger is hidden while a run dialog is open, which a run's response
// closes only after it arrives.
async function startTask(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: "Open" }));
  await user.click(await screen.findByRole("button", { name: "Start Task" }));
}

function querySafetyPopup() {
  return screen.queryByRole("dialog", { name: "Safety Checks" });
}

async function acknowledgeSafetyPopup(
  user: ReturnType<typeof userEvent.setup>,
) {
  const popup = await screen.findByRole("dialog", { name: "Safety Checks" });
  await user.click(within(popup).getByRole("button", { name: "Got it" }));
}

describe("the safety popup before a workflow's first run", () => {
  test.each([
    {
      when: "human-in-the-loop review is on",
      agent: { hasHumanInTheLoop: true, settings: BUILDER_SETTINGS },
    },
    {
      when: "sensitive action review is on",
      agent: { hasSensitiveAction: true, settings: ALL_REVIEW_ON },
    },
    {
      when: "no settings are saved, as human-in-the-loop review defaults on",
      agent: { hasHumanInTheLoop: true, settings: undefined },
    },
  ])("holds the run until acknowledged when $when", async ({ agent }) => {
    const runs = trackRuns();
    const user = userEvent.setup();
    render(modal(makeAgent(agent)));

    await startTask(user);

    expect(
      await screen.findByRole("dialog", { name: "Safety Checks" }),
    ).toBeDefined();
    expect(runs).not.toHaveBeenCalled();

    await acknowledgeSafetyPopup(user);
    await waitFor(() => expect(runs).toHaveBeenCalledOnce());
  });

  test.each([
    {
      when: "human-in-the-loop review is off",
      agent: { hasHumanInTheLoop: true, settings: ALL_REVIEW_OFF },
    },
    {
      when: "sensitive action review is off, as on a workflow saved from the builder",
      agent: { hasSensitiveAction: true, settings: BUILDER_SETTINGS },
    },
    {
      when: "no settings are saved, as sensitive action review defaults off",
      agent: { hasSensitiveAction: true, settings: undefined },
    },
    {
      when: "both kinds of review are off",
      agent: {
        hasHumanInTheLoop: true,
        hasSensitiveAction: true,
        settings: ALL_REVIEW_OFF,
      },
    },
  ])("starts the run without it when $when", async ({ agent }) => {
    const runs = trackRuns();
    const user = userEvent.setup();
    render(modal(makeAgent(agent)));

    await startTask(user);

    expect(querySafetyPopup()).toBeNull();
    await waitFor(() => expect(runs).toHaveBeenCalledOnce());
    expect(querySafetyPopup()).toBeNull();
  });

  test("promises sensitive action review when that review is on", async () => {
    trackRuns();
    const user = userEvent.setup();
    render(
      modal(
        makeAgent({
          hasHumanInTheLoop: true,
          hasSensitiveAction: true,
          settings: ALL_REVIEW_ON,
        }),
      ),
    );

    await startTask(user);

    const popup = await screen.findByRole("dialog", { name: "Safety Checks" });
    expect(within(popup).getByText(SENSITIVE_ACTION_PROMISE)).toBeDefined();
  });

  test("promises only the human-in-the-loop pause when sensitive actions run unreviewed", async () => {
    trackRuns();
    const user = userEvent.setup();
    render(
      modal(
        makeAgent({
          hasHumanInTheLoop: true,
          hasSensitiveAction: true,
          settings: BUILDER_SETTINGS,
        }),
      ),
    );

    await startTask(user);

    const popup = await screen.findByRole("dialog", { name: "Safety Checks" });
    expect(within(popup).getByText(HUMAN_IN_THE_LOOP_PROMISE)).toBeDefined();
    expect(within(popup).queryByText(SENSITIVE_ACTION_PROMISE)).toBeNull();
  });

  test("reads the settings the workflow has when Start Task is pressed", async () => {
    const runs = trackRuns();
    const user = userEvent.setup();
    const { rerender } = render(
      modal(makeAgent({ hasHumanInTheLoop: true, settings: BUILDER_SETTINGS })),
    );

    rerender(
      modal(makeAgent({ hasHumanInTheLoop: true, settings: ALL_REVIEW_OFF })),
    );
    await startTask(user);

    expect(querySafetyPopup()).toBeNull();
    await waitFor(() => expect(runs).toHaveBeenCalledOnce());
  });

  test("shows on a later run once review is turned on, as a skipped popup is not marked seen", async () => {
    const runs = trackRuns();
    const user = userEvent.setup();
    const { rerender } = render(
      modal(makeAgent({ hasHumanInTheLoop: true, settings: ALL_REVIEW_OFF })),
    );

    await startTask(user);
    await waitFor(() => expect(runs).toHaveBeenCalledOnce());
    expect(querySafetyPopup()).toBeNull();

    rerender(
      modal(makeAgent({ hasHumanInTheLoop: true, settings: BUILDER_SETTINGS })),
    );
    await startTask(user);

    expect(
      await screen.findByRole("dialog", { name: "Safety Checks" }),
    ).toBeDefined();
    expect(runs).toHaveBeenCalledOnce();

    await acknowledgeSafetyPopup(user);
    await waitFor(() => expect(runs).toHaveBeenCalledTimes(2));
  });

  test("stays dismissed after Got it, across a reload of the page", async () => {
    const runs = trackRuns();
    const user = userEvent.setup();
    const agent = makeAgent({
      hasHumanInTheLoop: true,
      settings: BUILDER_SETTINGS,
    });
    const { unmount } = render(modal(agent));

    await startTask(user);
    await acknowledgeSafetyPopup(user);
    await waitFor(() => expect(runs).toHaveBeenCalledOnce());
    unmount();

    render(modal(agent));
    await startTask(user);

    expect(querySafetyPopup()).toBeNull();
    await waitFor(() => expect(runs).toHaveBeenCalledTimes(2));
  });

  test.each(["not json", '{"library-agent": true}'])(
    "still starts the run when the stored seen list is unreadable (%s)",
    async (stored) => {
      storage.set(Key.AI_AGENT_SAFETY_POPUP_SHOWN, stored);
      const runs = trackRuns();
      const user = userEvent.setup();
      render(
        modal(
          makeAgent({ hasHumanInTheLoop: true, settings: BUILDER_SETTINGS }),
        ),
      );

      await startTask(user);
      await acknowledgeSafetyPopup(user);

      await waitFor(() => expect(runs).toHaveBeenCalledOnce());
    },
  );
});
