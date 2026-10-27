import { IconType } from "@/components/layout/Navbar/helpers";
import { ExpertChangeGroup } from "@/app/(platform)/copilot/components/ToolChain/ExpertCards";
import { EmptySession } from "@/app/(platform)/copilot/components/EmptySession/EmptySession";
import { AccountMenu } from "@/components/layout/Navbar/components/AccountMenu/AccountMenu";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart } from "ai";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PlatformChrome } from "../PlatformChrome";

// The first-run guidance as a user meets it: the real app chrome around the
// real home page, profile menu and expert proposal. Nothing here reaches
// into the walkthrough's or the tips' own modules, only what is on screen.

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "ada@example.com" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

const onboarding = vi.hoisted(() => ({
  current: {
    state: { completedSteps: [] as string[] },
    completeStep: vi.fn(),
  },
}));
vi.mock("@/providers/onboarding/onboarding-provider", async (original) => ({
  ...(await original<object>()),
  useOnboarding: () => onboarding.current,
}));

// The profile menu's feedback row talks to Tally and Sentry's replay.
vi.mock("@/components/molecules/TallyPoup/useTallyFeedback", () => ({
  useTallyFeedback: () => ({
    isFormVisible: false,
    isLoading: false,
    openFeedbackForm: vi.fn(),
  }),
}));

vi.mock("../usePlatformChrome", () => ({
  usePlatformChrome: () => ({
    showAppSidebar: true,
    showTourSidebar: false,
    showChangelog: false,
  }),
}));

// The sidebar's items, tagged the way the real sidebar tags them.
vi.mock("@/components/layout/AppSidebar/AppSidebar", () => ({
  AppSidebar: () => (
    <nav>
      <div data-tour-target="chats">Recent chats</div>
      <div data-tour-target="team">Team</div>
      <div data-tour-target="files">Files</div>
      <div data-tour-target="wallet">Wallet</div>
    </nav>
  ),
}));
vi.mock("@/components/layout/TopUpPrompt/TopUpPromptProvider", () => ({
  TopUpPromptProvider: ({ children }: { children: ReactNode }) => (
    <>{children}</>
  ),
}));
vi.mock("../../PaywallGate/PaywallGate", () => ({
  PaywallGate: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
vi.mock("../../admin/components/AdminImpersonationBanner", () => ({
  AdminImpersonationBanner: () => null,
}));
vi.mock("../../components/GlobalSearchModal/GlobalSearchOverlay", () => ({
  GlobalSearchOverlay: () => null,
}));

// jsdom lays nothing out: give every tagged element a box on screen.
function stubLayout() {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    function (this: HTMLElement) {
      return this.hasAttribute("data-tour-target")
        ? DOMRect.fromRect({ x: 10, y: 100, width: 240, height: 40 })
        : DOMRect.fromRect({ x: 0, y: 0, width: 0, height: 0 });
    },
  );
}

const WELCOME_PENDING_KEY = "autogpt:onboarding-welcome-pending";

function renderInApp(children: ReactNode) {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "", hasMemory: true });
  return render(
    <Wrapper>
      <PlatformChrome>{children}</PlatformChrome>
    </Wrapper>,
  );
}

function Home() {
  return (
    <EmptySession
      isCreatingSession={false}
      onCreateSession={() => {}}
      onSend={vi.fn()}
    />
  );
}

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
  window.sessionStorage.clear();
  window.localStorage.clear();
  onboarding.current = { state: { completedSteps: [] }, completeStep: vi.fn() };
  stubLayout();
});

afterEach(async () => {
  // The walkthrough outlives a test's render; end it the way a user would.
  const skip = screen.queryByRole("button", { name: "Skip tour" });
  if (skip) await userEvent.click(skip);
  window.history.replaceState({}, "", "/");
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("first run, as the user sees it", () => {
  it("walks someone fresh out of onboarding over the real sidebar, instead of concept cards", async () => {
    setIntroPath("A");
    renderInApp(<Home />);

    expect(
      await screen.findByRole("dialog", {
        name: "Every task gets its own chat",
      }),
    ).toBeDefined();
    expect(screen.getByText("1 of 5")).toBeDefined();
    expect(screen.queryByText("Meet your Head of AI.")).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(
      await screen.findByRole("dialog", {
        name: "Experts take tasks off your plate",
      }),
    ).toBeDefined();
  });

  it("does not come back once it is skipped", async () => {
    setIntroPath("B");
    renderInApp(<Home />);
    await screen.findByRole("dialog", { name: "Every task gets its own chat" });

    await userEvent.click(screen.getByRole("button", { name: "Skip tour" }));

    await waitFor(() =>
      expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull(),
    );
    expect(onboarding.current.completeStep).toHaveBeenCalledWith(
      "FIRST_RUN_TOUR",
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("leaves an existing user who already had the old cards alone", async () => {
    onboarding.current.state = { completedSteps: ["CAPABILITY_CARDS"] };
    setIntroPath("A");
    renderInApp(<Home />);

    await waitFor(() =>
      expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull(),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("replays from the profile menu's Show me around", async () => {
    // On Home itself it opens straight away; from another page it waits for
    // the route to change there.
    window.history.replaceState({}, "", "/home");
    renderInApp(
      <AccountMenu
        userName="Ada Lovelace"
        userEmail="ada@example.com"
        menuItemGroups={[
          {
            items: [
              {
                icon: IconType.Settings,
                text: "Settings",
                href: "/settings/account",
              },
            ],
          },
          { items: [{ icon: IconType.LogOut, text: "Log out" }] },
        ]}
      />,
    );

    await userEvent.click(screen.getByTestId("profile-popout-menu-trigger"));
    await userEvent.click(
      await screen.findByRole("button", { name: "Show me around" }),
    );

    expect(
      await screen.findByRole("dialog", {
        name: "Every task gets its own chat",
      }),
    ).toBeDefined();
  });
});

describe("contextual tips, as the user sees them", () => {
  it("says experts can own tasks the first time Otto proposes one", async () => {
    const proposal = {
      type: "tool-raise_expert",
      state: "output-available",
      toolCallId: "call-1",
      input: {},
      output: {
        type: "expert_change_proposed",
        applied: false,
        confirmation_id: "c-1",
        preview: { kind: "hire", name: "Fiona", role: "Bookkeeper" },
      },
    } as ToolUIPart;
    renderInApp(<ExpertChangeGroup parts={[proposal]} isCurrentlyStreaming />);

    expect(await screen.findByText("Experts can own tasks")).toBeDefined();
  });
});
