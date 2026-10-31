import { fireEvent, renderHook, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useBottomScrollShadow } from "./useBottomScrollShadow";

function scrollElement(scrollHeight: number) {
  const element = document.createElement("div");
  Object.defineProperties(element, {
    scrollHeight: { value: scrollHeight, writable: true, configurable: true },
    clientHeight: { value: 100, configurable: true },
  });
  return element;
}

test("reattaches to remounted lists and stops observing detached lists", () => {
  const first = scrollElement(300);
  const second = scrollElement(600);
  const remove = vi.spyOn(first, "removeEventListener");
  const { result, rerender, unmount } = renderHook(
    ({ element }: { element: HTMLElement | null }) =>
      useBottomScrollShadow(element),
    { initialProps: { element: first as HTMLElement | null } },
  );
  expect(result.current).toBe(true);
  first.scrollTop = 200;
  fireEvent.scroll(first);
  expect(result.current).toBe(false);
  rerender({ element: null });
  expect(result.current).toBe(false);
  expect(remove).toHaveBeenCalledWith("scroll", expect.any(Function));
  rerender({ element: second });
  expect(result.current).toBe(true);
  fireEvent.scroll(first);
  expect(result.current).toBe(true);
  second.scrollTop = 500;
  fireEvent.scroll(second);
  expect(result.current).toBe(false);
  unmount();
  remove.mockRestore();
});

test("updates when filtering replaces children without resizing the list", async () => {
  const element = scrollElement(100);
  const { result } = renderHook(() => useBottomScrollShadow(element));
  expect(result.current).toBe(false);
  Object.defineProperty(element, "scrollHeight", { value: 300 });
  element.appendChild(document.createElement("div"));
  await waitFor(() => expect(result.current).toBe(true));
});
