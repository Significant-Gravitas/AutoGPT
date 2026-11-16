import { renderHook } from "@testing-library/react";
import { expect, test } from "vitest";
import { useReadEveryRender } from "../useReadEveryRender";

test("reads the outside value again on every render, with the same inputs", () => {
  const outside = { status: "idle" };
  const { result, rerender } = renderHook(
    ({ key }) => useReadEveryRender(() => `${key}:${outside.status}`),
    { initialProps: { key: "expert-1" } },
  );
  expect(result.current).toBe("expert-1:idle");

  outside.status = "pending";
  rerender({ key: "expert-1" });

  expect(result.current).toBe("expert-1:pending");
});

test("does not render on its own when the outside value changes", () => {
  const outside = { status: "idle" };
  let renders = 0;
  const { result } = renderHook(() => {
    renders += 1;
    return useReadEveryRender(() => outside.status);
  });
  const rendersAfterMount = renders;

  outside.status = "done";

  expect(renders).toBe(rendersAfterMount);
  expect(result.current).toBe("idle");
});

test("returns a fresh object on every render without looping", () => {
  const outside = { status: "idle" };
  const { result, rerender } = renderHook(() =>
    useReadEveryRender(() => ({ status: outside.status })),
  );
  expect(result.current).toEqual({ status: "idle" });

  outside.status = "pending";
  rerender();

  expect(result.current).toEqual({ status: "pending" });
});
