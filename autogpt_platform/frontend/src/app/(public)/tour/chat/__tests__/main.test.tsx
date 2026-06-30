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

async function sendMessage(text: string) {
  const input = getChatInput();
  fireEvent.change(input, { target: { value: text } });
  const form = input.closest("form");
  if (!form) throw new Error("chat input is not inside a form");
  fireEvent.submit(form);
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

  test("plays both scripted turns and opens the upsell modal", async () => {
    render(<TourChatPage />);

    // 1. Empty chat exposes the input.
    expect(getChatInput()).toBeDefined();

    // 2. First send streams in the scripted assistant turn.
    await sendMessage("Watch competitor pricing");

    expect(screen.getByText(/break that down/i)).toBeDefined();

    // The decompose tool card renders its steps (expanded by default).
    expect(
      screen.getByText(/Detect changes vs\. the last snapshot/i),
    ).toBeDefined();

    // 3. Second send finishes the script and opens the upsell modal.
    await sendMessage("yes build it");

    // The create_agent preview card renders (accordion title + block count).
    expect(
      screen.getAllByText(/Competitor Pricing Watcher/i).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText(/4 blocks/i)).toBeDefined();

    // The run_agent card renders a completed execution.
    expect(screen.getByText(/Execution completed/i)).toBeDefined();

    expect(screen.getByText(/Ready to build your own/i)).toBeDefined();
  });
});
