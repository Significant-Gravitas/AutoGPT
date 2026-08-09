import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const authUser = vi.hoisted(() => ({
  current: { id: "user-1" } as { id: string } | null,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: authUser.current }),
}));

const flags = vi.hoisted(() => ({ current: {} as Record<string, boolean> }));
const flagsReady = vi.hoisted(() => ({ current: true }));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ONBOARDING_TAB_INTROS: "onboarding-tab-intros" },
  useFlagStatus: (flag: string) => ({
    enabled: flags.current[flag] ?? false,
    ready: flagsReady.current,
  }),
}));

const onboarding = vi.hoisted(() => ({
  current: {
    state: null as { completedSteps: string[] } | null,
    completeStep: vi.fn(),
  },
}));
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: () => onboarding.current,
}));

import { useTabIntroCard } from "../useTabIntroCard";

const SEEN_KEY = "autogpt:tab-intro-seen:agents";

function renderGate(
  tab: "agents" | "marketplace" | "build" = "agents",
  canShow = true,
) {
  return renderHook(() => useTabIntroCard(tab, canShow));
}

beforeEach(() => {
  window.localStorage.clear();
  capture.mockReset();
  authUser.current = { id: "user-1" };
  flags.current = { "onboarding-tab-intros": true };
  flagsReady.current = true;
  onboarding.current = { state: { completedSteps: [] }, completeStep: vi.fn() };
});

describe("useTabIntroCard — when it opens", () => {
  it("opens on a first visit and reports the card as shown", async () => {
    const { result } = renderGate();

    expect(result.current.isOpen).toBe(true);
    await waitFor(() =>
      expect(capture).toHaveBeenCalledWith("tab_intro_shown", {
        tab: "agents",
      }),
    );
  });

  it("stays closed while the flag is off", () => {
    flags.current = {};

    expect(renderGate().result.current.isOpen).toBe(false);
    expect(capture).not.toHaveBeenCalled();
  });

  it("stays closed until LaunchDarkly has answered", () => {
    flagsReady.current = false;

    expect(renderGate().result.current.isOpen).toBe(false);
  });

  it("stays closed until the onboarding record has loaded", () => {
    onboarding.current.state = null;

    expect(renderGate().result.current.isOpen).toBe(false);
  });

  it("stays closed when the step was already recorded on another device", () => {
    onboarding.current.state = { completedSteps: ["AGENTS_TAB_INTRO"] };

    expect(renderGate().result.current.isOpen).toBe(false);
  });

  it("only consults its own tab's step", () => {
    onboarding.current.state = { completedSteps: ["AGENTS_TAB_INTRO"] };

    expect(renderGate("marketplace").result.current.isOpen).toBe(true);
  });

  it("stays closed when this browser already saw it for this user", () => {
    window.localStorage.setItem(SEEN_KEY, "user-1");

    expect(renderGate().result.current.isOpen).toBe(false);
  });

  it("opens for a different account on the same browser", () => {
    window.localStorage.setItem(SEEN_KEY, "someone-else");

    expect(renderGate().result.current.isOpen).toBe(true);
  });

  it("burns nothing when the tab vetoes this particular visit", () => {
    const { result } = renderGate("agents", false);

    expect(result.current.isOpen).toBe(false);
    expect(capture).not.toHaveBeenCalled();
    expect(onboarding.current.completeStep).not.toHaveBeenCalled();
    expect(window.localStorage.getItem(SEEN_KEY)).toBeNull();
  });
});

describe("useTabIntroCard — dismissal", () => {
  it("records the step, caches it locally, and never reopens", async () => {
    const { result, rerender } = renderGate();

    result.current.dismiss();
    rerender();

    expect(result.current.isOpen).toBe(false);
    expect(onboarding.current.completeStep).toHaveBeenCalledWith(
      "AGENTS_TAB_INTRO",
    );
    expect(window.localStorage.getItem(SEEN_KEY)).toBe("user-1");
    expect(capture).toHaveBeenCalledWith("tab_intro_dismissed", {
      tab: "agents",
    });
  });

  it("tags a CTA dismissal with the action that was taken", () => {
    const { result, rerender } = renderGate("build");

    result.current.takeAction("ask_autopilot");
    rerender();

    expect(result.current.isOpen).toBe(false);
    expect(onboarding.current.completeStep).toHaveBeenCalledWith(
      "BUILD_TAB_INTRO",
    );
    expect(capture).toHaveBeenCalledWith("tab_intro_cta_clicked", {
      tab: "build",
      cta: "ask_autopilot",
    });
    expect(capture).not.toHaveBeenCalledWith(
      "tab_intro_dismissed",
      expect.anything(),
    );
  });

  it("still closes when localStorage is unavailable", () => {
    const setItem = vi
      .spyOn(window.localStorage, "setItem")
      .mockImplementation(() => {
        throw new Error("quota exceeded");
      });
    const { result, rerender } = renderGate();

    result.current.dismiss();
    rerender();

    expect(result.current.isOpen).toBe(false);
    expect(onboarding.current.completeStep).toHaveBeenCalledWith(
      "AGENTS_TAB_INTRO",
    );
    setItem.mockRestore();
  });
});
