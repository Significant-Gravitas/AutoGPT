import {
  Sidebar,
  SidebarProvider,
  useSidebar,
} from "@/components/molecules/Sidebar/Sidebar";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const auth = vi.hoisted(() => ({ userID: "user-1" as string | null }));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: auth.userID ? { id: auth.userID } : null,
    isLoggedIn: auth.userID !== null,
  }),
}));

const onboarding = vi.hoisted(() => ({
  current: {
    state: { completedSteps: [] as string[] } as {
      completedSteps: string[];
    } | null,
    completeStep: vi.fn(),
  },
}));
vi.mock("@/providers/onboarding/onboarding-provider", async (original) => ({
  ...(await original<object>()),
  useOnboarding: () => onboarding.current,
}));

const billing = vi.hoisted(() => ({ enabled: true, trial: true }));
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => billing.enabled,
}));
vi.mock("@/services/usageExperience/useUsageExperience", () => ({
  useUsageExperience: () => ({
    experience: { isActiveTrial: billing.trial },
  }),
}));

import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { SuggestedTaskCards } from "@/app/(platform)/copilot/components/SuggestedTasks/SuggestedTaskCards";
import { SuggestedTaskList } from "@/app/(platform)/copilot/components/SuggestedTasks/SuggestedTaskList";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import {
  tourStartingPoint,
  tourTarget,
  type TourTarget,
} from "@/lib/tour-targets";
import { FirstRunTour } from "../FirstRunTour";
import { useFirstRunTourStore } from "../store";
import { useFirstRunTourAutoStart } from "../useFirstRunTourAutoStart";

// jsdom lays nothing out, so each tagged element gets a box here: present and
// on screen unless listed in `hidden` (a collapsed sidebar, a closed sheet).
const hidden = new Set<string>();
// Targets laid out somewhere other than the default box.
const placed = new Map<string, DOMRectInit>();
function stubLayout() {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    function (this: HTMLElement) {
      const target = this.getAttribute("data-tour-target");
      if (!target || hidden.has(target))
        return DOMRect.fromRect({ x: 0, y: 0, width: 0, height: 0 });
      return DOMRect.fromRect(
        placed.get(target) ?? { x: 10, y: 100, width: 240, height: 40 },
      );
    },
  );
}

function Targets({ only }: { only?: TourTarget[] }) {
  const show = (target: TourTarget) => !only || only.includes(target);
  return (
    <div>
      {show("chats") && <div {...tourTarget("chats")}>Recent chats</div>}
      {show("team") && <div {...tourTarget("team")}>Team</div>}
      {show("files") && <div {...tourTarget("files")}>Files</div>}
      {show("wallet") && <div {...tourTarget("wallet")}>Wallet</div>}
      {show("account") && <div {...tourTarget("account")}>Account</div>}
      {show("menu") && <div {...tourTarget("menu")}>Menu</div>}
      {show("composer") && (
        <div {...tourTarget("composer")}>
          <textarea aria-label="Message Otto" />
        </div>
      )}
      {show("first-tasks") && (
        <ol {...tourTarget("first-tasks")}>
          <li>
            <button type="button" {...tourStartingPoint()}>
              Draft my Monday status report
            </button>
            <button
              type="button"
              aria-label="Hide: Draft my Monday status report"
            />
          </li>
        </ol>
      )}
    </div>
  );
}

function SidebarState() {
  const { open, openMobile } = useSidebar();
  return (
    <>
      <p data-testid="sidebar-state">{open ? "open" : "collapsed"}</p>
      <p data-testid="sheet-state">{openMobile ? "open" : "closed"}</p>
    </>
  );
}

function renderTour(only?: TourTarget[]) {
  return render(
    <SidebarProvider defaultOpen={false}>
      <SidebarState />
      <Targets only={only} />
      <FirstRunTour />
    </SidebarProvider>,
  );
}

function openTour(source: "auto" | "menu" = "auto") {
  act(() => useFirstRunTourStore.getState().open(source));
}

beforeEach(() => {
  auth.userID = "user-1";
  placed.clear();
  window.localStorage.clear();
  capture.mockReset();
  hidden.clear();
  billing.enabled = true;
  billing.trial = true;
  onboarding.current = {
    state: { completedSteps: [] },
    completeStep: vi.fn(),
  };
  stubLayout();
});

afterEach(() => {
  act(() => useFirstRunTourStore.getState().close());
  vi.restoreAllMocks();
});

describe("first-run walkthrough", () => {
  it("points at chats, Team, Files and the wallet, one line each, and ends on Start here", async () => {
    renderTour();
    openTour();

    const steps = [
      ["Every task gets its own chat", "1 of 5"],
      ["Experts take tasks off your plate", "2 of 5"],
      ["Your results are saved in Files", "3 of 5"],
      ["See what you have left", "4 of 5"],
    ];
    for (const [title, counter] of steps) {
      expect(await screen.findByRole("dialog", { name: title })).toBeDefined();
      expect(screen.getByText(counter)).toBeDefined();
      await userEvent.click(screen.getByRole("button", { name: "Next" }));
    }

    expect(
      await screen.findByRole("dialog", { name: "Start with one task" }),
    ).toBeDefined();
    expect(
      screen.getByText(
        "Pick a suggestion and Otto gets going. Or describe your own in the box below.",
      ),
    ).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: /Start here/ }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(onboarding.current.completeStep).toHaveBeenCalledWith(
      "FIRST_RUN_TOUR",
    );
    expect(window.localStorage.getItem("autogpt:first-run-tour-seen")).toBe(
      "user-1",
    );
    // "Start here" lands on the first suggested task, ready to send.
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Draft my Monday status report" }),
      ),
    );
    expect(capture).toHaveBeenCalledWith("first_run_tour_completed", {
      opened_from: "auto",
      step_id: "start",
      step_index: 4,
      via: "start_here",
    });
  });

  it("explains the trial allowance on the wallet step", async () => {
    renderTour();
    openTour();
    act(() => useFirstRunTourStore.getState().goTo("allowance"));

    expect(
      await screen.findByText(
        "Your trial includes an allowance for chats and experts. Check what's left here, and earn extra credits.",
      ),
    ).toBeDefined();
  });

  it("points at the sidebar footer while the wallet has not rendered", async () => {
    // The wallet draws nothing until credits load, and nothing at all if
    // they fail; the step still points at where it lives.
    renderTour(["chats", "team", "files", "account", "first-tasks"]);
    openTour();
    act(() => useFirstRunTourStore.getState().goTo("allowance"));

    expect(
      await screen.findByText(
        "Your allowance and credits are here, next to your profile.",
      ),
    ).toBeDefined();
  });

  it("scrolls a target that is only partly on screen into view", async () => {
    // The walkthrough locks page scrolling, so a target peeking over the
    // bottom edge would otherwise get a sliver of a spotlight for good.
    placed.set("first-tasks", {
      x: 300,
      y: window.innerHeight - 12,
      width: 600,
      height: 240,
    });
    const scrollIntoView = vi.fn();
    const original = HTMLElement.prototype.scrollIntoView;
    HTMLElement.prototype.scrollIntoView = scrollIntoView;
    try {
      renderTour();
      openTour();
      expect(scrollIntoView).not.toHaveBeenCalled();

      act(() => useFirstRunTourStore.getState().goTo("start"));

      await waitFor(() => expect(scrollIntoView).toHaveBeenCalledTimes(1));
      expect(scrollIntoView.mock.contexts[0]).toBe(
        document.querySelector('[data-tour-target="first-tasks"]'),
      );
    } finally {
      HTMLElement.prototype.scrollIntoView = original;
    }
  });

  it("still points at a target below the fold, and scrolls it into view", async () => {
    // On a short screen the first tasks can sit entirely below the fold;
    // they are scrolled away, not missing.
    placed.set("first-tasks", {
      x: 300,
      y: window.innerHeight + 200,
      width: 600,
      height: 240,
    });
    const scrollIntoView = vi.fn();
    const original = HTMLElement.prototype.scrollIntoView;
    HTMLElement.prototype.scrollIntoView = scrollIntoView;
    try {
      renderTour();
      openTour();
      act(() => useFirstRunTourStore.getState().goTo("start"));

      expect(
        await screen.findByText(
          "Pick a suggestion and Otto gets going. Or describe your own in the box below.",
        ),
      ).toBeDefined();
      await waitFor(() =>
        expect(scrollIntoView.mock.contexts).toContain(
          document.querySelector('[data-tour-target="first-tasks"]'),
        ),
      );
    } finally {
      HTMLElement.prototype.scrollIntoView = original;
    }
  });

  it("leaves the wallet step out where billing is off", async () => {
    billing.enabled = false;
    renderTour();
    openTour();

    expect(await screen.findByText("1 of 4")).toBeDefined();
  });

  it("stays on the same step when billing turns out to be on mid-walkthrough", async () => {
    billing.enabled = false;
    const { rerender } = renderTour();
    openTour();
    for (let i = 0; i < 3; i++)
      await userEvent.click(
        await screen.findByRole("button", { name: "Next" }),
      );
    expect(await screen.findByText("4 of 4")).toBeDefined();

    billing.enabled = true;
    rerender(
      <SidebarProvider defaultOpen={false}>
        <SidebarState />
        <Targets />
        <FirstRunTour />
      </SidebarProvider>,
    );

    expect(await screen.findByText("5 of 5")).toBeDefined();
    expect(
      screen.getByRole("dialog", { name: "Start with one task" }),
    ).toBeDefined();
  });

  it("carries on to the next step when the one showing drops out", async () => {
    const { rerender } = renderTour();
    openTour();
    for (let i = 0; i < 3; i++)
      await userEvent.click(
        await screen.findByRole("button", { name: "Next" }),
      );
    expect(
      await screen.findByRole("dialog", {
        name: "See what you have left",
      }),
    ).toBeDefined();

    billing.enabled = false;
    rerender(
      <SidebarProvider defaultOpen={false}>
        <SidebarState />
        <Targets />
        <FirstRunTour />
      </SidebarProvider>,
    );

    expect(await screen.findByText("4 of 4")).toBeDefined();
    expect(
      screen.getByRole("dialog", { name: "Start with one task" }),
    ).toBeDefined();
  });

  it("opens the collapsed sidebar, where a new account's wallet is hidden", async () => {
    renderTour();
    expect(screen.getByTestId("sidebar-state").textContent).toBe("collapsed");

    openTour();

    await waitFor(() =>
      expect(screen.getByTestId("sidebar-state").textContent).toBe("open"),
    );
  });

  it("borrows a collapsed sidebar for the walkthrough and gives it back, without saving it open", async () => {
    // Someone who keeps the sidebar collapsed and asks for "Show me around"
    // must not find their saved preference overwritten afterwards.
    document.cookie = "sidebar_open=false; path=/";
    render(
      <SidebarProvider defaultOpen={false} persistState>
        <SidebarState />
        <Targets />
        <FirstRunTour />
      </SidebarProvider>,
    );
    openTour("menu");

    await waitFor(() =>
      expect(screen.getByTestId("sidebar-state").textContent).toBe("open"),
    );
    expect(document.cookie).toContain("sidebar_open=false");

    await userEvent.click(
      await screen.findByRole("button", { name: "Skip tour" }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("sidebar-state").textContent).toBe("collapsed"),
    );
    expect(document.cookie).toContain("sidebar_open=false");
  });

  it("still opens the collapsed sidebar when its effects run twice (StrictMode, a remount)", async () => {
    render(
      <StrictMode>
        <SidebarProvider defaultOpen={false}>
          <SidebarState />
          <Targets />
          <FirstRunTour />
        </SidebarProvider>
      </StrictMode>,
    );
    openTour();

    await waitFor(() =>
      expect(screen.getByTestId("sidebar-state").textContent).toBe("open"),
    );
    await act(async () => {});
    expect(screen.getByTestId("sidebar-state").textContent).toBe("open");
  });

  it("leaves the sidebar collapsed if the person collapses it during the walkthrough", async () => {
    // Collapsing it (Ctrl/Cmd+B) is their choice and saves it; the walkthrough
    // must not reopen it over that, or give it back opened afterwards.
    document.cookie = "sidebar_open=false; path=/";
    render(
      <SidebarProvider defaultOpen={false} persistState>
        <SidebarState />
        <Targets />
        <FirstRunTour />
      </SidebarProvider>,
    );
    openTour("menu");
    await waitFor(() =>
      expect(screen.getByTestId("sidebar-state").textContent).toBe("open"),
    );

    await userEvent.keyboard("{Control>}b{/Control}");
    await waitFor(() =>
      expect(screen.getByTestId("sidebar-state").textContent).toBe("collapsed"),
    );
    await act(async () => {});
    expect(screen.getByTestId("sidebar-state").textContent).toBe("collapsed");

    await userEvent.click(screen.getByRole("button", { name: "Skip tour" }));
    await act(async () => {});
    expect(screen.getByTestId("sidebar-state").textContent).toBe("collapsed");
    expect(document.cookie).toContain("sidebar_open=false");
  });

  it("opens the phone's sidebar for the steps that live there, so the real controls are lit", async () => {
    // Pointing at the menu button and saying "open the menu" asked for a tap
    // the walkthrough's scrim and focus trap do not allow.
    const width = window.innerWidth;
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 390,
    });
    try {
      renderTour();
      openTour();
      await screen.findByRole("dialog", {
        name: "Every task gets its own chat",
      });
      await waitFor(() =>
        expect(screen.getByTestId("sheet-state").textContent).toBe("open"),
      );

      act(() => useFirstRunTourStore.getState().goTo("start"));
      await waitFor(() =>
        expect(screen.getByTestId("sheet-state").textContent).toBe("closed"),
      );

      act(() => useFirstRunTourStore.getState().goTo("files"));
      await waitFor(() =>
        expect(screen.getByTestId("sheet-state").textContent).toBe("open"),
      );

      await userEvent.click(screen.getByRole("button", { name: "Skip tour" }));
      await waitFor(() =>
        expect(screen.getByTestId("sheet-state").textContent).toBe("closed"),
      );
    } finally {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        value: width,
      });
    }
  });

  it("reports each step it shows once, and never as the emitter's source", async () => {
    renderTour();
    openTour("menu");
    await screen.findByRole("dialog", { name: "Every task gets its own chat" });
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("dialog", {
      name: "Experts take tasks off your plate",
    });
    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    await screen.findByRole("dialog", { name: "Every task gets its own chat" });
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByRole("dialog", {
      name: "Experts take tasks off your plate",
    });

    await waitFor(() => {
      const viewed = capture.mock.calls
        .filter(([event]) => event === "first_run_tour_step_viewed")
        .map(([, properties]) => properties.step_id);
      expect(viewed).toEqual(["chats", "team"]);
    });
    expect(capture).toHaveBeenCalledWith("first_run_tour_shown", {
      opened_from: "menu",
    });
    // `source` is the emitter's super property (`web`); an event property
    // of that name would overwrite it in PostHog.
    for (const [event, properties] of capture.mock.calls)
      if (String(event).startsWith("first_run_tour_"))
        expect(properties).not.toHaveProperty("source");
  });

  it("reports what a phone's step points at once the sheet has opened, not the menu before it", async () => {
    const width = window.innerWidth;
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 390,
    });
    try {
      render(
        <SidebarProvider defaultOpen={false}>
          <div {...tourTarget("menu")}>Menu</div>
          <Sidebar>
            <div {...tourTarget("chats")}>Recent chats</div>
          </Sidebar>
          <FirstRunTour />
        </SidebarProvider>,
      );
      openTour();

      await waitFor(() =>
        expect(capture).toHaveBeenCalledWith(
          "first_run_tour_step_viewed",
          expect.objectContaining({ step_id: "chats" }),
        ),
      );
      expect(capture).toHaveBeenCalledWith(
        "first_run_tour_step_viewed",
        expect.objectContaining({ step_id: "chats", target: "chats" }),
      );
    } finally {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        value: width,
      });
    }
  });

  it("stays reachable over the phone's sidebar sheet, for screen readers too", async () => {
    // The sheet is a modal dialog of its own: it hides everything outside
    // it from assistive tech and takes focus. The card has to end up above it.
    const width = window.innerWidth;
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 390,
    });
    try {
      render(
        <SidebarProvider defaultOpen={false}>
          <Sidebar>
            <div {...tourTarget("chats")}>Recent chats</div>
            <div {...tourTarget("team")}>Team</div>
          </Sidebar>
          <FirstRunTour />
        </SidebarProvider>,
      );
      openTour();

      const card = await screen.findByRole("dialog", {
        name: "Every task gets its own chat",
      });
      await waitFor(() =>
        expect(card.closest("[aria-hidden=true]")).toBeNull(),
      );
      await waitFor(() =>
        expect(
          screen.getByRole("dialog", { name: "Every task gets its own chat" }),
        ).toBeDefined(),
      );
      // The card mounts again a couple of frames after the sheet: look the
      // button up on every try, an earlier one may have been replaced.
      await waitFor(() => {
        const next = screen.getByRole("button", { name: "Next" });
        expect(next.closest("[aria-hidden=true]")).toBeNull();
        expect(document.activeElement).toBe(next);
      });
    } finally {
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        value: width,
      });
    }
  });

  it("points a phone at the menu button for anything in the sidebar", async () => {
    hidden.add("chats");
    renderTour();
    openTour();

    expect(
      await screen.findByText(
        "Ask Otto for something and it starts a new chat. They all stay here, so you can pick any of them back up. Open the menu to find it.",
      ),
    ).toBeDefined();
  });

  it("tells a screen reader where to find it on a phone too", async () => {
    hidden.add("chats");
    renderTour();
    openTour();
    await screen.findByRole("dialog", { name: "Every task gets its own chat" });

    const announcement = document.querySelector('[aria-live="polite"]');
    expect(announcement?.textContent).toBe(
      "Step 1 of 5. Every task gets its own chat. Ask Otto for something and it starts a new chat. They all stay here, so you can pick any of them back up. Open the menu to find it.",
    );
  });

  it("says the first tasks are on their way when they have not landed yet", async () => {
    renderTour(["chats", "team", "files", "wallet"]);
    openTour();
    act(() => useFirstRunTourStore.getState().goTo("start"));

    expect(
      await screen.findByText(
        "Otto is getting your first tasks ready. They'll show up here in a moment.",
      ),
    ).toBeDefined();
  });

  it("records the walkthrough as seen when it is skipped, by button or Escape", async () => {
    renderTour();
    openTour();
    await userEvent.click(
      await screen.findByRole("button", { name: "Skip tour" }),
    );

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(onboarding.current.completeStep).toHaveBeenCalledWith(
      "FIRST_RUN_TOUR",
    );

    openTour("menu");
    await screen.findByRole("dialog", { name: "Every task gets its own chat" });
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(capture).toHaveBeenCalledWith("first_run_tour_skipped", {
      opened_from: "menu",
      step_id: "chats",
      step_index: 0,
      via: null,
    });
  });

  it("hands focus to the composer when it is skipped, by button or Escape", async () => {
    // The composer mounts under the walkthrough, whose focus trap keeps it
    // from taking focus; nothing else would give it focus afterwards.
    renderTour();
    const composer = screen.getByRole("textbox", { name: "Message Otto" });
    openTour();
    await userEvent.click(
      await screen.findByRole("button", { name: "Skip tour" }),
    );
    await waitFor(() => expect(document.activeElement).toBe(composer));

    composer.blur();
    openTour("menu");
    await screen.findByRole("dialog", { name: "Every task gets its own chat" });
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(document.activeElement).toBe(composer));
  });

  it("does not focus the composer on a touch screen, where that opens the keyboard", async () => {
    const matchMedia = vi.spyOn(window, "matchMedia").mockImplementation(
      (query: string) =>
        ({
          matches: query.includes("coarse"),
          media: query,
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          addListener: vi.fn(),
          removeListener: vi.fn(),
        }) as unknown as MediaQueryList,
    );
    try {
      renderTour();
      const composer = screen.getByRole("textbox", { name: "Message Otto" });
      openTour();
      await userEvent.click(
        await screen.findByRole("button", { name: "Skip tour" }),
      );
      await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(document.activeElement).not.toBe(composer);
    } finally {
      matchMedia.mockRestore();
    }
  });

  it("keeps keyboard focus inside the walkthrough and starts it on Next", async () => {
    renderTour();
    openTour();
    const next = await screen.findByRole("button", { name: "Next" });

    await waitFor(() => expect(document.activeElement).toBe(next));
    await userEvent.tab();
    const dialog = screen.getByRole("dialog", {
      name: "Every task gets its own chat",
    });
    expect(dialog.contains(document.activeElement)).toBe(true);
    await userEvent.tab();
    expect(dialog.contains(document.activeElement)).toBe(true);
  });

  it("offers Connect your tools on the last step, after which the walkthrough is done", async () => {
    renderTour();
    openTour();
    act(() => useFirstRunTourStore.getState().goTo("start"));

    await userEvent.click(
      await screen.findByRole("button", {
        name: "Or connect your tools first",
      }),
    );

    await waitFor(() =>
      expect(useFirstRunTourStore.getState().isConnectToolsOpen).toBe(true),
    );
    expect(useFirstRunTourStore.getState().source).toBeNull();
    expect(onboarding.current.completeStep).toHaveBeenCalledWith(
      "FIRST_RUN_TOUR",
    );
    act(() => useFirstRunTourStore.getState().setConnectToolsOpen(false));
  });

  // Home shows the tasks as a list, and the greeting fresh out of onboarding
  // shows them as cards: the walkthrough points at either.
  it.each([
    ["list", SuggestedTaskList],
    ["cards", SuggestedTaskCards],
  ])(
    "starts on a task's own button, never its Hide button (%s)",
    async (_, Tasks) => {
      // A task already sent is a link, and a busy row keeps Hide enabled, so
      // the first enabled button in the list can be "Hide: …".
      const tasks: SuggestedTask[] = [
        {
          id: "0",
          title: "Chase unpaid Shopify invoices",
          prompt: "Chase every unpaid invoice.",
          icon: "bell",
          state: "sent",
          session_id: "session-1",
        },
        {
          id: "1",
          title: "Turn sales calls into CRM updates",
          prompt: "Update the CRM after each call.",
          icon: "users",
          state: "open",
        },
      ];
      function renderWithTasks(disabled: boolean) {
        return render(
          <SidebarProvider defaultOpen={false}>
            <Targets only={["chats", "team", "files", "wallet", "composer"]} />
            <Tasks
              tasks={tasks}
              sendingId={null}
              disabled={disabled}
              onSend={vi.fn()}
              onHide={vi.fn()}
            />
            <FirstRunTour />
          </SidebarProvider>,
        );
      }

      const { unmount } = renderWithTasks(false);
      openTour();
      act(() => useFirstRunTourStore.getState().goTo("start"));
      // It points at the tasks themselves, not the composer beside them.
      expect(
        await screen.findByText(
          "Pick a suggestion and Otto gets going. Or describe your own in the box below.",
        ),
      ).toBeDefined();
      await userEvent.click(
        await screen.findByRole("button", {
          name: /Start with one task|Start here/,
        }),
      );
      await waitFor(() =>
        expect(document.activeElement?.textContent).toContain(
          "Turn sales calls into CRM updates",
        ),
      );
      expect(document.activeElement?.getAttribute("aria-label")).toBeNull();
      unmount();

      // Every row busy: no task to start, so the composer.
      renderWithTasks(true);
      openTour();
      act(() => useFirstRunTourStore.getState().goTo("start"));
      await userEvent.click(
        await screen.findByRole("button", {
          name: /Start with one task|Start here/,
        }),
      );
      await waitFor(() =>
        expect(document.activeElement).toBe(
          screen.getByRole("textbox", { name: "Message Otto" }),
        ),
      );
    },
  );

  it("hands focus to the composer when Connect your tools closes", async () => {
    // The button that opened it went with the walkthrough, so there is
    // nothing to hand focus back to; keyboard users would land on <body>.
    renderTour();
    const composer = screen.getByRole("textbox", { name: "Message Otto" });
    openTour();
    act(() => useFirstRunTourStore.getState().goTo("start"));
    await userEvent.click(
      await screen.findByRole("button", {
        name: "Or connect your tools first",
      }),
    );
    await waitFor(() =>
      expect(useFirstRunTourStore.getState().isConnectToolsOpen).toBe(true),
    );

    act(() => useFirstRunTourStore.getState().setConnectToolsOpen(false));

    await waitFor(() => expect(document.activeElement).toBe(composer));
  });

  it("keeps the way out and the step counter readable (4.5:1 on the card)", async () => {
    renderTour();
    openTour();

    const skip = await screen.findByRole("button", { name: "Skip tour" });
    expect(skip.className).toContain("text-foreground-secondary");
    expect(screen.getByText("1 of 5").className).toContain(
      "text-foreground-secondary",
    );
  });

  it("jumps to the first task without smooth scrolling when motion is reduced", async () => {
    const matchMedia = vi.spyOn(window, "matchMedia").mockImplementation(
      (query: string) =>
        ({
          matches: query.includes("prefers-reduced-motion: reduce"),
          media: query,
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          addListener: vi.fn(),
          removeListener: vi.fn(),
        }) as unknown as MediaQueryList,
    );
    const scrollIntoView = vi.fn();
    const original = HTMLElement.prototype.scrollIntoView;
    HTMLElement.prototype.scrollIntoView = scrollIntoView;
    try {
      renderTour();
      openTour();
      act(() => useFirstRunTourStore.getState().goTo("start"));
      await userEvent.click(
        await screen.findByRole("button", { name: /Start here/ }),
      );

      await waitFor(() =>
        expect(scrollIntoView).toHaveBeenCalledWith(
          expect.objectContaining({ behavior: "auto" }),
        ),
      );
      expect(scrollIntoView).not.toHaveBeenCalledWith(
        expect.objectContaining({ behavior: "smooth" }),
      );
    } finally {
      HTMLElement.prototype.scrollIntoView = original;
      matchMedia.mockRestore();
    }
  });
});

function AutoStart({
  isPending,
  onFinished,
}: {
  isPending: boolean;
  onFinished: (outcome: { wasShown: boolean }) => void;
}) {
  useFirstRunTourAutoStart({ isPending, onFinished });
  return null;
}

describe("first-run walkthrough — whose it is", () => {
  afterEach(() => {
    act(() => useAuthStore.setState({ user: null }));
  });

  it("stays open through a dropped session, and closes when another account signs in", () => {
    // A failed session check drops the user too; the same person signing
    // back in keeps their place. Another account must not inherit it.
    act(() => useAuthStore.setState({ user: { id: "user-1" } as User }));
    openTour();

    act(() => useAuthStore.setState({ user: null }));
    expect(useFirstRunTourStore.getState().source).toBe("auto");
    act(() => useAuthStore.setState({ user: { id: "user-1" } as User }));
    expect(useFirstRunTourStore.getState().source).toBe("auto");

    act(() => useAuthStore.setState({ user: { id: "user-2" } as User }));
    // In the same tick as the sign-in: nothing user-2 opens can race it.
    expect(useFirstRunTourStore.getState().source).toBeNull();
  });
});

describe("first-run walkthrough — who gets it by itself", () => {
  it("opens for someone fresh out of onboarding", async () => {
    render(<AutoStart isPending onFinished={vi.fn()} />);

    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );
  });

  it("does not open for an existing user who already went through the old cards", async () => {
    onboarding.current.state = { completedSteps: ["CAPABILITY_CARDS"] };
    const onFinished = vi.fn();
    render(<AutoStart isPending onFinished={onFinished} />);

    await waitFor(() => expect(onFinished).toHaveBeenCalled());
    expect(useFirstRunTourStore.getState().source).toBeNull();
  });

  it("does not open again once recorded, on this device or another", async () => {
    onboarding.current.state = { completedSteps: ["FIRST_RUN_TOUR"] };
    render(<AutoStart isPending onFinished={vi.fn()} />);
    window.localStorage.setItem("autogpt:first-run-tour-seen", "user-1");
    onboarding.current.state = { completedSteps: [] };
    render(<AutoStart isPending onFinished={vi.fn()} />);

    await act(async () => {});
    expect(useFirstRunTourStore.getState().source).toBeNull();
  });

  it("opens straight away for someone fresh out of onboarding, before the onboarding record loads", async () => {
    // The first-run flag is set by this tab's own wizard, so it cannot be a
    // leftover from another device. Waiting for the record held the
    // walkthrough back for as long as the provider took to load after a
    // sign-up, and the user had already moved on.
    onboarding.current.state = null;
    render(<AutoStart isPending onFinished={vi.fn()} />);

    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );
  });

  it("closes again if the record then shows this person already had an intro", async () => {
    // A first-run flag can outlive its account in the tab (sessionStorage
    // survives signing out and back in as someone else).
    onboarding.current.state = null;
    const onFinished = vi.fn();
    const { rerender } = render(
      <AutoStart isPending onFinished={onFinished} />,
    );
    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );

    onboarding.current.state = { completedSteps: ["CAPABILITY_CARDS"] };
    rerender(<AutoStart isPending onFinished={onFinished} />);

    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBeNull(),
    );
    expect(onFinished).toHaveBeenCalledTimes(1);
  });

  it("leaves a walkthrough the person asked for alone when the record loads", async () => {
    onboarding.current.state = null;
    const { rerender } = render(<AutoStart isPending onFinished={vi.fn()} />);
    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );
    act(() => {
      useFirstRunTourStore.getState().close();
      useFirstRunTourStore.getState().open("menu");
    });

    onboarding.current.state = { completedSteps: ["FIRST_RUN_TOUR"] };
    rerender(<AutoStart isPending onFinished={vi.fn()} />);

    await act(async () => {});
    expect(useFirstRunTourStore.getState().source).toBe("menu");
  });

  it("does not count a walkthrough closed while the user is signed out as their first run ending", async () => {
    const onFinished = vi.fn();
    const { rerender } = render(
      <AutoStart isPending onFinished={onFinished} />,
    );
    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );

    // A session check fails: no user for a moment, and the walkthrough goes.
    auth.userID = null;
    act(() => useFirstRunTourStore.getState().close());
    // Home stays mounted through the switch, still holding its first-run flag.
    rerender(<AutoStart isPending onFinished={onFinished} />);
    await act(async () => {});

    expect(onFinished).not.toHaveBeenCalled();
  });

  it("does nothing for someone who did not just finish onboarding", async () => {
    render(<AutoStart isPending={false} onFinished={vi.fn()} />);

    await act(async () => {});
    expect(useFirstRunTourStore.getState().source).toBeNull();
  });

  it("reports the end once the walkthrough it opened closes", async () => {
    const onFinished = vi.fn();
    render(<AutoStart isPending onFinished={onFinished} />);
    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );
    expect(onFinished).not.toHaveBeenCalled();

    act(() => useFirstRunTourStore.getState().close());

    await waitFor(() => expect(onFinished).toHaveBeenCalledTimes(1));
    expect(onFinished).toHaveBeenCalledWith({ wasShown: true });
  });

  it("says the walkthrough was not shown when it never opens", async () => {
    onboarding.current.state = { completedSteps: ["CAPABILITY_CARDS"] };
    const onFinished = vi.fn();
    render(<AutoStart isPending onFinished={onFinished} />);

    await waitFor(() =>
      expect(onFinished).toHaveBeenCalledWith({ wasShown: false }),
    );
  });
});
