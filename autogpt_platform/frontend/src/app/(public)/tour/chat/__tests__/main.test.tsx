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

function getChatInput() {
  return screen.getByRole("textbox", { name: /chat message input/i });
}

// The input is prefilled and locked — the visitor only presses Enter to send.
async function pressEnterToSend() {
  fireEvent.keyDown(getChatInput(), { key: "Enter" });
  // The scripted reveal streams in over real setTimeout delays; advance fake
  // timers past the whole turn (longest turn ~2.9s) so every part is committed.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(4000);
  });
}

describe("Tour chat scripted demo", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
  });

  test("prefills the prompt, plays both turns, and opens the upsell", async () => {
    render(<TourChatPage />);

    // 1. The prompt bar is prefilled with the first scripted prompt.
    expect(getChatInput()).toBeDefined();
    expect(
      screen.getByText(/Watch a competitor's pricing page/i),
    ).toBeDefined();

    // 2. Pressing Enter streams in the scripted assistant turn.
    await pressEnterToSend();

    expect(screen.getByText(/break that down/i)).toBeDefined();
    expect(
      screen.getByText(/Detect changes vs\. the last snapshot/i),
    ).toBeDefined();

    // The prompt bar now prefills the second turn's prompt.
    expect(screen.getByText(/build and run it for me/i)).toBeDefined();

    // 3. Pressing Enter again finishes the script and opens the upsell modal.
    await pressEnterToSend();

    expect(
      screen.getAllByText(/Competitor Pricing Watcher/i).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText(/4 blocks/i)).toBeDefined();
    expect(screen.getByText(/Execution completed/i)).toBeDefined();
    expect(screen.getByText(/Ready to build your own/i)).toBeDefined();
  });

  test("sidebar navigation switches to the second chat's prompt", async () => {
    render(<TourChatPage />);

    expect(
      screen.getByText(/Watch a competitor's pricing page/i),
    ).toBeDefined();

    // Click the second chat in the sidebar.
    fireEvent.click(screen.getByText(/Summarize my weekly emails/i));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    // The prompt bar now prefills the second chat's opening prompt.
    expect(
      screen.getByText(/Summarize my unread emails every morning/i),
    ).toBeDefined();
  });
});
