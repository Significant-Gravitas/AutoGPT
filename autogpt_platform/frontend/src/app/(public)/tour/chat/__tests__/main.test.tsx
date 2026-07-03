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

function getSendBar() {
  return screen.getByRole("button", { name: /^Send:/i });
}

const ADVANCE_STEP_MS = 200;
// Longest turn is ~7.7s of parts — including the 5s fake run — plus the 3s
// hold before the final turn flips to the upsell.
const ADVANCE_TOTAL_MS = 13000;

// The prompt bar is prefilled and locked — the visitor only presses Enter to send.
async function pressEnterToSend() {
  fireEvent.keyDown(getSendBar(), { key: "Enter" });
  // TourStreamingText mounts mid-stream (from a setTimeout callback) and
  // registers its own setInterval — a timer created by an effect that fires
  // *during* an in-flight advanceTimersByTimeAsync call is never picked up
  // by that same call. Advancing in small chunks, each in its own act(),
  // gives React a chance to flush the mount effect and register the new
  // interval before the next chunk advances past it.
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

describe("Tour chat scripted demo", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // The scenario store is module-level state — reset between tests.
    useTourStore.setState({ activeScenarioId: DEFAULT_SCENARIO_ID });
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  test("plays the competitor watch demo through to the payoff and upsell", async () => {
    render(<TourChatPage />);

    // 1. The prompt bar is prefilled with the flagship scenario's prompt.
    expect(getSendBar()).toBeDefined();
    expect(
      screen.getByText(/Watch a competitor's pricing page/i),
    ).toBeDefined();

    // 2. Pressing Enter streams in the scripted plan turn.
    await pressEnterToSend();

    expect(screen.getByText(/break that down/i)).toBeDefined();
    expect(
      screen.getByText(/Detect changes vs\. the last snapshot/i),
    ).toBeDefined();

    // The prompt bar now prefills the second turn's prompt.
    expect(screen.getByText(/build and run it for me/i)).toBeDefined();

    // 3. Pressing Enter again builds the agent and shows the payoff artifact.
    await pressEnterToSend();

    // Agent card: block chain chips + schedule row, no raw JSON.
    expect(
      screen.getAllByText(/Competitor Pricing Watcher/i).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("Text Compare").length).toBeGreaterThan(0);
    expect(screen.getByText(/Daily · 8:00 AM/i)).toBeDefined();
    expect(screen.queryByText(/"nodes"/)).toBeNull();

    // Payoff artifact: the email that "landed" with the price diff.
    expect(screen.getByText(/what lands in your inbox/i)).toBeDefined();
    expect(screen.getByText(/Price change detected/i)).toBeDefined();
    expect(screen.getByText("$59/mo")).toBeDefined();
    expect(screen.getByText("+20.4%")).toBeDefined();

    // Upsell: Pro-first CTA with self-host secondary.
    expect(screen.getByText(/Ready to build your own/i)).toBeDefined();
    expect(screen.getByText(/Start with Pro — \$42\.50\/mo/i)).toBeDefined();
    expect(screen.getByText(/Self-host free/i)).toBeDefined();
  });

  test("scenario chips switch the demo path", async () => {
    render(<TourChatPage />);

    for (const label of [
      "Daily brief",
      "Call prep",
      "Competitor watch",
      "Support queue",
    ]) {
      expect(screen.getByRole("button", { name: label })).toBeDefined();
    }

    fireEvent.click(screen.getByRole("button", { name: "Daily brief" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // The prompt bar now prefills the selected scenario's opening prompt.
    expect(
      screen.getByText(/pull my unread emails and calendar/i),
    ).toBeDefined();
  });
});
