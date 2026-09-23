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

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import TourChatPage from "../page";
import { DEFAULT_SCENARIO_ID } from "../script/tourScenarios";
import { useTourStore } from "../tourStore";

function getSendBar() {
  return screen.getByRole("button", { name: /^Send:/i });
}

const ADVANCE_STEP_MS = 200;
// Longest turn is ~7.7s of parts — including the 5s fake run — plus the 3s
// hold before the final turn flips to the upsell.
const ADVANCE_TOTAL_MS = 16000;

// TourStreamingText mounts mid-stream (from a setTimeout callback) and
// registers its own setInterval — a timer created by an effect that fires
// *during* an in-flight advanceTimersByTimeAsync call is never picked up
// by that same call. Advancing in small chunks, each in its own act(),
// gives React a chance to flush the mount effect and register the new
// interval before the next chunk advances past it.
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

// The prompt bar is prefilled and locked — pressing Enter sends it immediately
// instead of waiting for the auto-start.
async function pressEnterToSend() {
  fireEvent.keyDown(getSendBar(), { key: "Enter" });
  await advanceThroughTurn();
}

describe("Tour chat scripted demo", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // The scenario store is module-level state — reset between tests.
    useTourStore.setState({
      activeScenarioId: DEFAULT_SCENARIO_ID,
      runId: 0,
      isDemoComplete: false,
      watchedScenarioIds: [],
      isNudgeVisible: false,
    });
    // The artifact panel lives in the shared copilot UI store (also
    // module-level) — a completed demo leaves it open across tests.
    useCopilotUIStore.getState().closeArtifactPanel({ persist: false });
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  test("auto-types the first prompt into the bar and sends it on its own", async () => {
    render(<TourChatPage />);

    // The bar starts empty.
    expect(screen.queryByText(/email me when the price changes/i)).toBeNull();
    expect(screen.queryByRole("button", { name: /^Send:/i })).toBeNull();

    // After a beat the first prompt starts typing itself into the bar.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1200);
    });
    expect(getSendBar()).toBeDefined();

    // Once typed it sends itself and the scripted turn streams in.
    await advanceThroughTurn();

    // The auto-sent user message is now in the transcript. getByText throws on
    // multiple matches, so this also guards against the turn double-firing.
    expect(screen.getByText(/email me when the price changes/i)).toBeDefined();
    expect(screen.getByText(/break that down/i)).toBeDefined();

    // Only now does the second turn's prompt prefill, ready to send.
    expect(screen.getByText(/build and run it for me/i)).toBeDefined();
    expect(getSendBar()).toBeDefined();
  });

  test("clicking the active sidebar session restarts the demo with an empty bar", async () => {
    render(<TourChatPage />);

    // Let the first turn auto-play so the second turn's prompt prefills.
    await advanceThroughTurn();
    expect(getSendBar()).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Competitor watch" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // Fresh run: transcript cleared and the bar empty/disabled again while
    // the first turn re-plays on its own.
    expect(screen.queryByRole("button", { name: /^Send:/i })).toBeNull();
    expect(screen.queryByText(/break that down/i)).toBeNull();
  });

  test("plays the competitor watch demo through to the payoff and upsell", async () => {
    render(<TourChatPage />);

    // 1. The first turn auto-plays and streams in the scripted plan turn.
    await advanceThroughTurn();

    expect(screen.getByText(/break that down/i)).toBeDefined();
    expect(
      screen.getByText(/Detect changes vs\. the last snapshot/i),
    ).toBeDefined();

    // The prompt bar now prefills the second turn's prompt.
    expect(screen.getByText(/build and run it for me/i)).toBeDefined();

    // 2. Pressing Enter builds the agent and shows the payoff artifact.
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

    // The closing line points the visitor at the artifact panel, with the
    // filename bolded (its own <strong> node, hence the split assertions).
    expect(screen.getByText('"competitor-pricing-report.md"')).toBeDefined();
    expect(
      screen.getByText(/will appear in a moment on the right side/i),
    ).toBeDefined();

    // The end card carries the upsell now — the sidebar card hides.
    expect(screen.queryByText(/Ready to build your own/i)).toBeNull();
    expect(screen.queryByText(/Start with Pro for \$42\.50\/mo/i)).toBeNull();
    expect(screen.queryByText("Self-host free")).toBeNull();
    expect(screen.queryByText(/Replay demo/i)).toBeNull();

    // Completion flips the store flag that hides the sidebar card.
    expect(useTourStore.getState().isDemoComplete).toBe(true);
  });

  test("switching scenario after a completed demo closes the artifact panel", async () => {
    render(<TourChatPage />);

    // Play the default demo through both turns so it completes and opens
    // the payoff artifact panel. (The chat column no longer dims behind it —
    // the end card has to stay legible.)
    await advanceThroughTurn();
    await pressEnterToSend();

    expect(useTourStore.getState().isDemoComplete).toBe(true);
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);

    // Picking another chat example from the sidebar must close the panel —
    // it belongs to the finished demo, not the new one.
    fireEvent.click(screen.getByRole("button", { name: "Daily brief" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
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

    // The bar stays empty while the selected scenario's first turn auto-plays.
    expect(
      screen.queryByText(/pull my unread emails and calendar/i),
    ).toBeNull();

    // The newly selected scenario auto-plays its first turn too.
    await advanceThroughTurn();

    expect(
      screen.getByText(/pull my unread emails and calendar/i),
    ).toBeDefined();
    expect(
      screen.getByText(/Love it\. Here's how I'll set that up/i),
    ).toBeDefined();
  });
});
