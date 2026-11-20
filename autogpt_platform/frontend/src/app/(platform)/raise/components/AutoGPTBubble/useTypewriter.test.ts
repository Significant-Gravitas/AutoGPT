import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { useTypewriter } from "./useTypewriter";

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

test("renders full text immediately when disabled", () => {
  const onComplete = vi.fn();
  const { result } = renderHook(() =>
    useTypewriter("Hello world", onComplete, false),
  );

  expect(result.current.typed).toBe("Hello world");
  expect(result.current.isTyping).toBe(false);
  expect(onComplete).toHaveBeenCalledOnce();
});

test("types text character by character when enabled", () => {
  const { result } = renderHook(() => useTypewriter("Hi", undefined, true));

  expect(result.current.typed).toBe("");
  expect(result.current.isTyping).toBe(true);

  act(() => {
    vi.advanceTimersByTime(250);
  });
  expect(result.current.typed).toBe("H");

  act(() => {
    vi.advanceTimersByTime(18);
  });
  expect(result.current.typed).toBe("Hi");
  expect(result.current.isTyping).toBe(false);
});
