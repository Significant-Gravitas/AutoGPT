import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ScreenState } from "../useBrainDumpStep";

const { trackBrainDump, device } = vi.hoisted(() => ({
  trackBrainDump: vi.fn(),
  device: { isWide: true, reducedMotion: false },
}));

vi.mock("@/services/onboarding/brain-dump-analytics", () => ({
  trackBrainDump,
}));

vi.mock("../useIsWideLayout", () => ({
  useIsWideLayout: () => device.isWide,
}));

vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  useReducedMotion: () => device.reducedMotion,
}));

import { BRAIN_DUMP_PROMPT_SETS } from "../prompts";
import { usePromptRotation } from "../usePromptRotation";

const OPERATIONS = BRAIN_DUMP_PROMPT_SETS.Operations.questions;

function renderRotation(initial: { role: string; screen: ScreenState }) {
  return renderHook((props) => usePromptRotation(props), {
    initialProps: initial,
  });
}

async function wait(ms: number) {
  await act(async () => {
    vi.advanceTimersByTime(ms);
  });
}

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  trackBrainDump.mockClear();
  device.isWide = true;
  device.reducedMotion = false;
  setVisibility("visible");
});

afterEach(() => {
  vi.useRealTimers();
  setVisibility("visible");
});

describe("usePromptRotation", () => {
  it("starts half way round the set and tracks nothing on the first render", () => {
    const { result } = renderRotation({ role: "Operations", screen: "typing" });

    expect(result.current.index).toBe(4);
    expect(result.current.question).toEqual(OPERATIONS[4]);
    expect(result.current.promptSet).toBe("Operations");
    expect(trackBrainDump).not.toHaveBeenCalled();
  });

  it("auto-advances every 20 seconds while typing", async () => {
    const { result } = renderRotation({ role: "Operations", screen: "typing" });

    await wait(19_999);
    expect(result.current.index).toBe(4);

    await wait(1);
    expect(result.current.question).toEqual(OPERATIONS[5]);
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_prompt_advanced", {
      via: "auto",
      screen: "typing",
      prompt_index: 5,
      prompt_set: "Operations",
    });

    await wait(20_000);
    expect(result.current.index).toBe(6);
  });

  it("keeps moving every 15 seconds while recording, unannounced", async () => {
    const { result } = renderRotation({
      role: "Operations",
      screen: "recording",
    });

    await wait(14_999);
    expect(result.current.index).toBe(4);

    await wait(1);
    expect(result.current.index).toBe(5);
    expect(result.current.announcement).toBe("");
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_prompt_advanced", {
      via: "auto",
      screen: "recording",
      prompt_index: 5,
      prompt_set: "Operations",
    });

    act(() => result.current.advance("click"));
    expect(result.current.index).toBe(6);
  });

  it("holds still while paused and picks up again when resumed", async () => {
    const { result } = renderRotation({
      role: "Operations",
      screen: "recording",
    });

    act(() => result.current.togglePause());
    expect(result.current.isPaused).toBe(true);
    await wait(120_000);
    expect(result.current.index).toBe(4);
    expect(vi.getTimerCount()).toBe(0);

    // "Another question" still works while paused.
    act(() => result.current.advance("click"));
    expect(result.current.index).toBe(5);

    act(() => result.current.togglePause());
    expect(result.current.isPaused).toBe(false);
    await wait(15_000);
    expect(result.current.index).toBe(6);
  });

  it("announces only a question the user asked for", async () => {
    const { result, rerender } = renderRotation({
      role: "Operations",
      screen: "typing",
    });
    expect(result.current.announcement).toBe("");

    await wait(20_000);
    expect(result.current.index).toBe(5);
    expect(result.current.announcement).toBe("");

    act(() => result.current.advance("click"));
    expect(result.current.announcement).toBe(OPERATIONS[6].text);

    // Another screen mounts its own prompt; the old click is not read again.
    rerender({ role: "Operations", screen: "recording" });
    expect(result.current.announcement).toBe("");
    rerender({ role: "Operations", screen: "typing" });

    await wait(20_000);
    expect(result.current.index).toBe(7);
    expect(result.current.announcement).toBe("");
  });

  it("rotates at rest only without the ambient cards, every 12 seconds", async () => {
    const desktop = renderRotation({ role: "Operations", screen: "rest" });
    await wait(60_000);
    expect(desktop.result.current.index).toBe(4);
    desktop.unmount();

    device.isWide = false;
    const phone = renderRotation({ role: "Operations", screen: "rest" });
    await wait(12_000);
    expect(phone.result.current.index).toBe(5);
    expect(trackBrainDump).toHaveBeenCalledWith(
      "brain_dump_prompt_advanced",
      expect.objectContaining({ screen: "rest", via: "auto" }),
    );
  });

  it("advances straight away on a click and restarts the timer", async () => {
    const { result } = renderRotation({ role: "Operations", screen: "typing" });

    await wait(15_000);
    act(() => {
      result.current.advance("click");
    });
    expect(result.current.index).toBe(5);
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_prompt_advanced", {
      via: "click",
      screen: "typing",
      prompt_index: 5,
      prompt_set: "Operations",
    });

    // 20s from the start would have been due now, but the click restarted it.
    await wait(5_000);
    expect(result.current.index).toBe(5);
    await wait(15_000);
    expect(result.current.index).toBe(6);
  });

  it("wraps from the last question back to the first", () => {
    const { result } = renderRotation({ role: "Operations", screen: "typing" });

    act(() => result.current.advance("click"));
    act(() => result.current.advance("click"));
    act(() => result.current.advance("click"));
    act(() => result.current.advance("click"));

    expect(result.current.index).toBe(0);
    expect(result.current.question).toEqual(OPERATIONS[0]);
  });

  it("does not auto-advance while the tab is hidden", async () => {
    const { result } = renderRotation({ role: "Operations", screen: "typing" });

    setVisibility("hidden");
    await wait(60_000);
    expect(result.current.index).toBe(4);

    setVisibility("visible");
    await wait(20_000);
    expect(result.current.index).toBe(5);
  });

  it.each(["processing", "failed", "recovery", "insufficient"] as const)(
    "does not auto-advance on the %s screen",
    async (screen) => {
      device.isWide = false;
      const { result } = renderRotation({ role: "Operations", screen });

      await wait(60_000);

      expect(result.current.index).toBe(4);
      expect(trackBrainDump).not.toHaveBeenCalled();
    },
  );

  it("leaves rotation to the button when motion is reduced", async () => {
    device.reducedMotion = true;
    const { result } = renderRotation({ role: "Operations", screen: "typing" });

    await wait(60_000);
    expect(result.current.index).toBe(4);

    act(() => result.current.advance("click"));
    expect(result.current.index).toBe(5);
  });

  it("starts over when the role changes", () => {
    const { result, rerender } = renderRotation({
      role: "Operations",
      screen: "typing",
    });
    act(() => result.current.advance("click"));
    expect(result.current.index).toBe(5);

    rerender({ role: "Marketing", screen: "typing" });

    expect(result.current.index).toBe(4);
    expect(result.current.promptSet).toBe("Marketing");
    expect(result.current.question).toEqual(
      BRAIN_DUMP_PROMPT_SETS.Marketing.questions[4],
    );
  });

  it("uses the Other set for a role it does not know", () => {
    const { result } = renderRotation({ role: "Dentist", screen: "typing" });

    expect(result.current.promptSet).toBe("Other");
    expect(result.current.question).toEqual(
      BRAIN_DUMP_PROMPT_SETS.Other.questions[4],
    );
  });

  it("counts the distinct prompts actually shown", async () => {
    const { result, rerender } = renderRotation({
      role: "Operations",
      screen: "rest",
    });
    // The live prompt is hidden at rest beside the ambient cards.
    expect(result.current.promptsSeen()).toBe(0);

    rerender({ role: "Operations", screen: "typing" });
    expect(result.current.promptsSeen()).toBe(1);

    act(() => result.current.advance("click"));
    await wait(20_000);
    expect(result.current.promptsSeen()).toBe(3);

    rerender({ role: "Operations", screen: "recording" });
    expect(result.current.promptsSeen()).toBe(3);
  });

  it("clears its timer when unmounted", () => {
    const { unmount } = renderRotation({
      role: "Operations",
      screen: "typing",
    });
    expect(vi.getTimerCount()).toBeGreaterThan(0);

    unmount();

    expect(vi.getTimerCount()).toBe(0);
  });
});
