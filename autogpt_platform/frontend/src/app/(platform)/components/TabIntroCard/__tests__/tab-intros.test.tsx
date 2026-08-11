import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-1" } }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump",
    HIRE_EXPERTS: "hire-experts",
  },
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: true, ready: true }),
}));

const completeStep = vi.hoisted(() => vi.fn());
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: () => ({ state: { completedSteps: [] }, completeStep }),
  default: ({ children }: { children: React.ReactNode }) => children,
}));

const push = vi.hoisted(() => vi.fn());
const searchParams = vi.hoisted(() => ({ current: "" }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
  useSearchParams: () => new URLSearchParams(searchParams.current),
  usePathname: () => "/build",
}));

const startTutorial = vi.hoisted(() => vi.fn());
vi.mock("@/app/(platform)/build/components/FlowEditor/tutorial", () => ({
  startTutorial,
}));

import { AgentsTabIntro } from "@/app/(platform)/library/components/AgentsTabIntro/AgentsTabIntro";
import { BuildTabIntro } from "@/app/(platform)/build/components/BuildTabIntro/BuildTabIntro";
import { useTutorialStore } from "@/app/(platform)/build/stores/tutorialStore";
import {
  AGENTS_SECTION_ID,
  FEATURED_SECTION_ID,
} from "@/app/(platform)/marketplace/components/MarketplaceTabIntro/helpers";
import { MarketplaceTabIntro } from "@/app/(platform)/marketplace/components/MarketplaceTabIntro/MarketplaceTabIntro";

beforeEach(() => {
  window.localStorage.clear();
  capture.mockReset();
  completeStep.mockReset();
  push.mockReset();
  startTutorial.mockReset();
  searchParams.current = "";
  useTutorialStore.getState().setIsTutorialRunning(false);
});

describe("AgentsTabIntro", () => {
  it("introduces the tab as mission control and steps aside on the CTA", async () => {
    render(<AgentsTabIntro />);

    expect(await screen.findByText("Your mission control.")).toBeDefined();
    expect(
      screen.getByText(
        "What's running, what's scheduled, what needs you, and what it costs — all in one place.",
      ),
    ).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: "See my agents" }),
    );

    await waitFor(() =>
      expect(screen.queryByText("Your mission control.")).toBeNull(),
    );
    expect(completeStep).toHaveBeenCalledWith("AGENTS_TAB_INTRO");
    expect(capture).toHaveBeenCalledWith("tab_intro_cta_clicked", {
      tab: "agents",
      cta: "see_agents",
    });
  });
});

describe("MarketplaceTabIntro", () => {
  it("scrolls the featured carousel into view on the CTA", async () => {
    const featured = document.createElement("section");
    featured.id = FEATURED_SECTION_ID;
    const scrollIntoView = vi.fn();
    featured.scrollIntoView = scrollIntoView;
    document.body.appendChild(featured);

    render(<MarketplaceTabIntro />);
    expect(await screen.findByText("Agents ready to work.")).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: "Browse featured agents" }),
    );

    expect(scrollIntoView).toHaveBeenCalledTimes(1);
    expect(completeStep).toHaveBeenCalledWith("MARKETPLACE_TAB_INTRO");
    expect(capture).toHaveBeenCalledWith("tab_intro_cta_clicked", {
      tab: "marketplace",
      cta: "browse_featured",
    });
    featured.remove();
  });

  it("falls back to the full listing when nothing is featured right now", async () => {
    const listing = document.createElement("div");
    listing.id = AGENTS_SECTION_ID;
    const scrollIntoView = vi.fn();
    listing.scrollIntoView = scrollIntoView;
    document.body.appendChild(listing);

    render(<MarketplaceTabIntro />);
    await screen.findByText("Agents ready to work.");

    await userEvent.click(
      screen.getByRole("button", { name: "Browse featured agents" }),
    );

    expect(scrollIntoView).toHaveBeenCalledTimes(1);
    await waitFor(() =>
      expect(screen.queryByText("Agents ready to work.")).toBeNull(),
    );
    listing.remove();
  });

  it("still closes when there is nothing to scroll to at all", async () => {
    render(<MarketplaceTabIntro />);
    await screen.findByText("Agents ready to work.");

    await userEvent.click(
      screen.getByRole("button", { name: "Browse featured agents" }),
    );

    await waitFor(() =>
      expect(screen.queryByText("Agents ready to work.")).toBeNull(),
    );
  });
});

describe("BuildTabIntro", () => {
  it("sends the user to AutoPilot from the primary CTA", async () => {
    render(<BuildTabIntro />);
    expect(await screen.findByText("Create your own workflows.")).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: "Ask AutoPilot to build it" }),
    );

    expect(push).toHaveBeenCalledWith("/copilot");
    expect(startTutorial).not.toHaveBeenCalled();
    expect(completeStep).toHaveBeenCalledWith("BUILD_TAB_INTRO");
    expect(capture).toHaveBeenCalledWith("tab_intro_cta_clicked", {
      tab: "build",
      cta: "ask_autopilot",
    });
  });

  it("launches the existing builder tutorial from the quiet alternative", async () => {
    render(<BuildTabIntro />);
    await screen.findByText("Create your own workflows.");

    await userEvent.click(
      screen.getByRole("button", { name: "Learn to build it yourself" }),
    );

    expect(startTutorial).toHaveBeenCalledTimes(1);
    expect(useTutorialStore.getState().isTutorialRunning).toBe(true);
    expect(push).not.toHaveBeenCalled();
    expect(capture).toHaveBeenCalledWith("tab_intro_cta_clicked", {
      tab: "build",
      cta: "builder_tutorial",
    });
  });

  it("stays out of the way of a saved graph, and keeps the intro for later", () => {
    searchParams.current = "flowID=graph-1";

    render(<BuildTabIntro />);

    expect(screen.queryByText("Create your own workflows.")).toBeNull();
    expect(completeStep).not.toHaveBeenCalled();
    expect(capture).not.toHaveBeenCalled();
  });
});
