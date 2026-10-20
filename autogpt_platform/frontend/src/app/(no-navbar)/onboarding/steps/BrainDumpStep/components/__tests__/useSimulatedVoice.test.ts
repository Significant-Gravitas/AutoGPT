import { act, renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useSimulatedVoice } from "../useSimulatedVoice";

afterEach(() => {
  vi.useRealTimers();
});

it("keeps stable levels, emits varied speech, and clears levels and timers on stop", () => {
  vi.useFakeTimers();
  const { result, rerender, unmount } = renderHook(
    ({ active }) => useSimulatedVoice(active),
    { initialProps: { active: false } },
  );
  const levels = result.current;
  expect(levels.map((level) => level.get())).toEqual([0, 0, 0, 0, 0]);
  rerender({ active: true });
  act(() => {
    vi.advanceTimersByTime(1200);
  });
  expect(result.current).toBe(levels);
  const speech = levels.map((level) => level.get());
  expect(speech.every((level) => level > 0 && level <= 1)).toBe(true);
  expect(new Set(speech).size).toBe(5);
  act(() => {
    vi.advanceTimersByTime(4800);
  });
  expect(levels.map((level) => level.get())).toEqual([0, 0, 0, 0, 0]);
  rerender({ active: false });
  expect(vi.getTimerCount()).toBe(0);
  rerender({ active: true });
  act(() => {
    vi.advanceTimersByTime(600);
  });
  expect(levels.some((level) => level.get() > 0)).toBe(true);
  unmount();
  expect(vi.getTimerCount()).toBe(0);
});
