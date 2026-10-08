import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";

// DotDistortionShader paints a canvas/WebGL frame that happy-dom cannot run and
// that is purely decorative — stub it so the real chat tree can render.
vi.mock("@/components/ui/dot-distortion-shader", () => ({
  DotDistortionShader: () => null,
}));

vi.mock("@/app/(platform)/copilot/useIsMobile", () => ({
  useIsMobile: () => false,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
  };
});

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { buildTourArtifactRef } from "../helpers";
import TourChatPage from "../page";
import { DEFAULT_SCENARIO_ID, getTourScenario } from "../script/tourScenarios";
import { useTourStore } from "../tourStore";

function getSendBar() {
  return screen.getByRole("button", { name: /^Send:/i });
}

const ADVANCE_STEP_MS = 200;
// Longest turn is ~7.7s of parts — including the 5s fake run — plus the 3s
// hold before the demo completes.
const ADVANCE_TOTAL_MS = 16000;

// Timers advance in small chunks so effects that register new timers
// mid-stream get picked up (see main.test.tsx for the full rationale).
async function advanceThroughTurn() {
  for (
    let elapsed = 0;
    elapsed < ADVANCE_TOTAL_MS;
    elapsed += ADVANCE_STEP_MS
  ) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ADVANCE_STEP_MS);
    });
  }
}

// Later turns prefill the prompt bar — the visitor presses Enter to send.
async function pressEnterToSend() {
  fireEvent.keyDown(getSendBar(), { key: "Enter" });
  await advanceThroughTurn();
}

describe("Tour chat app shell", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    // Both stores are module-level state — reset between tests.
    useTourStore.setState({
      activeScenarioId: DEFAULT_SCENARIO_ID,
      runId: 0,
      isDemoComplete: false,
      watchedScenarioIds: [],
      isNudgeVisible: false,
    });
    useCopilotUIStore.getState().clearArtifactPreview();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    vi.unstubAllEnvs();
  });

  test("renders a free-trial sidebar without demo chats and only Marketplace enabled", () => {
    render(<TourChatPage />);

    expect(document.querySelector("[aria-pressed]")).toBeNull();
    expect(screen.queryByText("Recent chats")).toBeNull();
    expect(screen.queryByText("Try Otto")).toBeNull();
    for (const label of [
      "Daily brief",
      "Call prep",
      "Competitor watch",
      "Support queue",
    ]) {
      expect(screen.queryByRole("button", { name: label })).toBeNull();
    }

    expect(screen.getByText("Your AI team starts here")).toBeDefined();
    const trialCTA = screen.getByRole("link", { name: "Start free trial" });
    expect(trialCTA.getAttribute("href")).toBe("/signup");
    expect(trialCTA.getAttribute("target")).toBeNull();
    expect(screen.queryByText(/Start with Pro/i)).toBeNull();
    expect(screen.queryByText(/\$42\.50/)).toBeNull();

    // Marketplace is the only live navigation target.
    const marketplace = screen.getByRole("link", { name: "Marketplace" });
    expect(marketplace.getAttribute("href")).toBe("/marketplace");
    expect(
      screen.getByRole("link", { name: "AutoGPT" }).getAttribute("href"),
    ).toBe("/marketplace");
    for (const label of ["New Task", "Search", "Agents", "Build", "Files"]) {
      const item = screen.getByRole("button", { name: label });
      expect(item.getAttribute("aria-disabled")).toBe("true");
    }
  });

  test("finishing the demo opens the artifact panel with the mock markdown file", async () => {
    // The next/dynamic ArtifactPanel chunk can't finish loading once timers
    // are faked. Pre-open an artifact on real timers and wait for the panel
    // to actually mount (this is what forces the lazy chunk to resolve),
    // then reset and run the scripted demo under fake timers.
    vi.useRealTimers();
    act(() => {
      useCopilotUIStore
        .getState()
        .openArtifact(
          buildTourArtifactRef(getTourScenario(DEFAULT_SCENARIO_ID)),
        );
    });
    render(<TourChatPage />);
    expect(
      await screen.findByText(
        "competitor-pricing-report.md",
        {},
        { timeout: 10_000 },
      ),
    ).toBeDefined();
    act(() => {
      useCopilotUIStore.getState().clearArtifactPreview();
    });
    vi.useFakeTimers();

    // The demo mounted under real timers, so its auto-start timeout is a real
    // timer that fake-timer advancing can't reach (and it may have already
    // fired during the findByText wait). Toggling the scenario remounts
    // TourChatHost under fake timers, giving a deterministic fresh demo.
    act(() => {
      useTourStore.setState({ activeScenarioId: "daily-brief" });
    });
    act(() => {
      useTourStore.setState({ activeScenarioId: DEFAULT_SCENARIO_ID });
    });

    // First turn auto-plays; the second is sent from the prefilled bar.
    await advanceThroughTurn();
    await pressEnterToSend();

    expect(useCopilotUIStore.getState().artifactPanel.activeArtifact?.id).toBe(
      "tour-competitor-watch",
    );
    // No findBy/waitFor — RTL polling hangs under fake timers. The lazy
    // chunk is already loaded, so the panel renders synchronously once the
    // store holds the artifact.
    expect(screen.getByText("competitor-pricing-report.md")).toBeDefined();
  }, 30_000);
});
