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

const datafast = vi.fn();

function eventsNamed(name: string) {
  return datafast.mock.calls.filter(([eventName]) => eventName === name);
}

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

async function pressEnterToSend() {
  fireEvent.keyDown(screen.getByRole("button", { name: /^Send:/i }), {
    key: "Enter",
  });
  await advanceThroughTurn();
}

describe("Tour DataFast tracking", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    window.datafast = datafast;
    datafast.mockClear();
    sessionStorage.clear();
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

  test("fires tour_start once per session and tour_scenario_start per scenario run", async () => {
    render(<TourChatPage />);

    expect(eventsNamed("tour_start")).toHaveLength(1);
    expect(eventsNamed("tour_scenario_start")).toEqual([
      ["tour_scenario_start", { scenario: DEFAULT_SCENARIO_ID }],
    ]);

    // Switching scenario starts a new run but must not re-fire tour_start.
    fireEvent.click(screen.getByRole("button", { name: "Daily brief" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(eventsNamed("tour_start")).toHaveLength(1);
    expect(eventsNamed("tour_scenario_start")).toHaveLength(2);
    expect(eventsNamed("tour_scenario_start")[1][1]).toEqual({
      scenario: "daily-brief",
    });
  });

  test("fires tour_scenario_complete when the demo plays through", async () => {
    render(<TourChatPage />);

    await advanceThroughTurn();
    expect(eventsNamed("tour_scenario_complete")).toHaveLength(0);

    await pressEnterToSend();

    expect(eventsNamed("tour_scenario_complete")).toEqual([
      ["tour_scenario_complete", { scenario: DEFAULT_SCENARIO_ID }],
    ]);
  });

  test("end card CTAs fire tour_cta_click with their labels", async () => {
    render(<TourChatPage />);
    await advanceThroughTurn();
    await pressEnterToSend();

    fireEvent.click(screen.getByText("Make this agent yours"));
    fireEvent.click(screen.getByText("or self-host free"));
    fireEvent.click(
      screen.getByRole("button", { name: /Watch another scenario/i }),
    );

    const labels = eventsNamed("tour_cta_click").map(
      ([, metadata]) => metadata,
    );
    expect(labels).toEqual([
      { label: "pricing", placement: "end-card" },
      { label: "self-host", placement: "end-card" },
      { label: "another-scenario", placement: "end-card" },
    ]);
  });
});
