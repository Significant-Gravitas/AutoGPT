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

import TourChatPage from "../page";
import { DEFAULT_SCENARIO_ID } from "../script/tourScenarios";
import { useTourStore } from "../tourStore";

const ADVANCE_STEP_MS = 200;
// Longest turn is ~7.7s of parts — including the 5s fake run — plus the 3s
// hold before the demo completes (see main.test.tsx for the chunking rationale).
const ADVANCE_TOTAL_MS = 16000;

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

async function playDemoThrough() {
  await advanceThroughTurn();
  fireEvent.keyDown(screen.getByRole("button", { name: /^Send:/i }), {
    key: "Enter",
  });
  await advanceThroughTurn();
}

describe("Tour demo end state", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useTourStore.setState({
      activeScenarioId: DEFAULT_SCENARIO_ID,
      runId: 0,
      isDemoComplete: false,
      watchedScenarioIds: [],
      isNudgeVisible: false,
    });
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  test("shows the end card with pricing + another-scenario CTAs after the demo", async () => {
    render(<TourChatPage />);

    expect(screen.queryByText(/Yours will too/i)).toBeNull();

    await playDemoThrough();

    expect(
      screen.getByText(/That took \d+ seconds\. Yours will too\./i),
    ).toBeDefined();
    expect(
      screen.getByText(/Start with Pro · \$42\.50\/mo · cancel anytime/i),
    ).toBeDefined();

    const pricingCta = screen
      .getByText("Make this agent yours")
      .closest("a") as HTMLAnchorElement;
    expect(pricingCta.href).toContain(
      "agpt.co/pricing?utm_source=tour&utm_medium=end_card",
    );
    expect(
      screen.getByRole("button", { name: /Watch another scenario/i }),
    ).toBeDefined();
  });

  test("marks the finished scenario as watched in the sidebar", async () => {
    render(<TourChatPage />);
    await playDemoThrough();

    expect(useTourStore.getState().watchedScenarioIds).toEqual([
      DEFAULT_SCENARIO_ID,
    ]);
    expect(screen.getByText("watched")).toBeDefined();
  });

  test("idling ~4s after the demo shows the next-scenario nudge chip", async () => {
    render(<TourChatPage />);
    await playDemoThrough();

    expect(screen.queryByText(/^Next:/)).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4100);
    });

    // Competitor watch is the default scenario, so the nudge points at the
    // next one in sidebar order: Support queue.
    expect(
      screen.getByText(/Next: Support queue — watch it in \d+s/),
    ).toBeDefined();
  });

  test("the nudge chip switches into the next scenario", async () => {
    render(<TourChatPage />);
    await playDemoThrough();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4100);
    });

    fireEvent.click(screen.getByText(/Next: Support queue/));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(useTourStore.getState().activeScenarioId).toBe("support-queue");
    // Fresh run: the end card is gone while the new scenario plays.
    expect(screen.queryByText(/Yours will too/i)).toBeNull();
  });

  test("watch another scenario starts the next unwatched demo", async () => {
    render(<TourChatPage />);
    await playDemoThrough();

    fireEvent.click(
      screen.getByRole("button", { name: /Watch another scenario/i }),
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(useTourStore.getState().activeScenarioId).toBe("support-queue");
  });
});
