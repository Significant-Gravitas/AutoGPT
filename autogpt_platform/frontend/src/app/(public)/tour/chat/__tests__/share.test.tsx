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

const clipboardWrite = vi.fn(async (_text: string) => {});
const datafast = vi.fn();

describe("Tour share button", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboardWrite },
    });
    clipboardWrite.mockClear();
    clipboardWrite.mockImplementation(async () => {});
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

  test("copies the tour URL with share attribution and flips to 'Link copied'", async () => {
    render(<TourChatPage />);

    fireEvent.click(screen.getByRole("button", { name: /Share this demo/i }));
    await act(async () => {});

    expect(clipboardWrite).toHaveBeenCalledTimes(1);
    expect(clipboardWrite.mock.calls[0][0]).toBe(
      `${window.location.origin}/tour/chat?utm_source=share`,
    );
    expect(
      datafast.mock.calls.filter(([name]) => name === "tour_cta_click"),
    ).toEqual([["tour_cta_click", { label: "share" }]]);

    expect(screen.getByRole("button", { name: /Link copied/i })).toBeDefined();

    // The button returns to its resting label after the copied beat.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100);
    });
    expect(
      screen.getByRole("button", { name: /Share this demo/i }),
    ).toBeDefined();
  });
});
