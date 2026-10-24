import { SidebarProvider, useSidebar } from "@/components/ui/sidebar";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-1" }, isLoggedIn: true }),
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

import { tourTarget, type TourTarget } from "../steps";
import { FirstRunTour } from "../FirstRunTour";
import { useFirstRunTourStore } from "../store";
import { useFirstRunTourAutoStart } from "../useFirstRunTourAutoStart";

// jsdom lays nothing out, so each tagged element gets a box here: present and
// on screen unless listed in `hidden` (a collapsed sidebar, a closed sheet).
const hidden = new Set<string>();
function stubLayout() {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    function (this: HTMLElement) {
      const target = this.getAttribute("data-tour-target");
      if (!target || hidden.has(target))
        return DOMRect.fromRect({ x: 0, y: 0, width: 0, height: 0 });
      return DOMRect.fromRect({ x: 10, y: 100, width: 240, height: 40 });
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
      {show("menu") && <div {...tourTarget("menu")}>Menu</div>}
      {show("first-tasks") && (
        <ol {...tourTarget("first-tasks")}>
          <li>
            <button type="button">Draft my Monday status report</button>
          </li>
        </ol>
      )}
    </div>
  );
}

function SidebarState() {
  const { open } = useSidebar();
  return <p data-testid="sidebar-state">{open ? "open" : "collapsed"}</p>;
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
      ["Work happens in chats", "1 of 5"],
      ["Experts can own a task", "2 of 5"],
      ["Results land in Files", "3 of 5"],
      ["Keep an eye on your allowance", "4 of 5"],
    ];
    for (const [title, counter] of steps) {
      expect(await screen.findByRole("dialog", { name: title })).toBeDefined();
      expect(screen.getByText(counter)).toBeDefined();
      await userEvent.click(screen.getByRole("button", { name: "Next" }));
    }

    expect(
      await screen.findByRole("dialog", { name: "Start here" }),
    ).toBeDefined();
    expect(
      screen.getByText("Pick a first task and Otto gets going on it."),
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
      source: "auto",
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
        "Your trial includes an allowance for chats and experts. See what's left here, and earn extra credits.",
      ),
    ).toBeDefined();
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
    expect(screen.getByRole("dialog", { name: "Start here" })).toBeDefined();
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
        name: "Keep an eye on your allowance",
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
    expect(screen.getByRole("dialog", { name: "Start here" })).toBeDefined();
  });

  it("opens the collapsed sidebar, where a new account's wallet is hidden", async () => {
    renderTour();
    expect(screen.getByTestId("sidebar-state").textContent).toBe("collapsed");

    openTour();

    await waitFor(() =>
      expect(screen.getByTestId("sidebar-state").textContent).toBe("open"),
    );
  });

  it("points a phone at the menu button for anything in the sidebar", async () => {
    hidden.add("chats");
    renderTour();
    openTour();

    expect(
      await screen.findByText(
        "Each task gets its own chat. They all stay here, ready to pick up. Open the menu to find it.",
      ),
    ).toBeDefined();
  });

  it("says the first tasks are on their way when they have not landed yet", async () => {
    renderTour(["chats", "team", "files", "wallet"]);
    openTour();
    act(() => useFirstRunTourStore.getState().goTo("start"));

    expect(
      await screen.findByText(
        "Otto is getting your first tasks ready. They'll show up on Home in a moment.",
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
    await screen.findByRole("dialog", { name: "Work happens in chats" });
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(capture).toHaveBeenCalledWith("first_run_tour_skipped", {
      source: "menu",
      step_id: "chats",
      step_index: 0,
      via: null,
    });
  });

  it("keeps keyboard focus inside the walkthrough and starts it on Next", async () => {
    renderTour();
    openTour();
    const next = await screen.findByRole("button", { name: "Next" });

    await waitFor(() => expect(document.activeElement).toBe(next));
    await userEvent.tab();
    const dialog = screen.getByRole("dialog", {
      name: "Work happens in chats",
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
});

function AutoStart({
  isPending,
  onFinished,
}: {
  isPending: boolean;
  onFinished: () => void;
}) {
  useFirstRunTourAutoStart({ isPending, onFinished });
  return null;
}

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
  });
});
