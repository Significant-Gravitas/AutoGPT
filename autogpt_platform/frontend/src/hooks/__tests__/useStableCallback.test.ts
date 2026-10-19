import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useStableCallback } from "../useStableCallback";

describe("useStableCallback", () => {
  it("returns the same function across renders", () => {
    const { result, rerender } = renderHook(({ fn }) => useStableCallback(fn), {
      initialProps: { fn: () => "a" },
    });
    const first = result.current;

    rerender({ fn: () => "b" });
    rerender({ fn: () => "c" });

    expect(result.current).toBe(first);
  });

  it("calls the latest implementation with the arguments and returns its result", () => {
    const first = vi.fn((a: number, b: number) => a + b);
    const latest = vi.fn((a: number, b: number) => a * b);
    const { result, rerender } = renderHook(({ fn }) => useStableCallback(fn), {
      initialProps: { fn: first },
    });
    const stable = result.current;

    expect(stable(2, 3)).toBe(5);
    rerender({ fn: latest });

    expect(stable(2, 3)).toBe(6);
    expect(first).toHaveBeenCalledTimes(1);
    expect(latest).toHaveBeenCalledWith(2, 3);
  });

  it("passes a rejected promise through to the caller", async () => {
    const { result } = renderHook(() =>
      useStableCallback(async () => {
        throw new Error("send failed");
      }),
    );

    await expect(result.current()).rejects.toThrow("send failed");
  });
});
