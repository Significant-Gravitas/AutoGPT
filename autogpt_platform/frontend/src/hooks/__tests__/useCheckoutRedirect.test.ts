import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";
import { useCheckoutRedirect } from "../useCheckoutRedirect";

describe("useCheckoutRedirect", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function startRedirect() {
    const hook = renderHook(() => useCheckoutRedirect());
    act(() => hook.result.current.markRedirecting());
    return hook;
  }

  test("stays busy until the page leaves", () => {
    const { result } = startRedirect();
    act(() => {
      vi.advanceTimersByTime(14_999);
    });
    expect(result.current.isRedirecting).toBe(true);
  });

  test("lets go after 15 seconds if the page never leaves", () => {
    const { result } = startRedirect();
    act(() => {
      vi.advanceTimersByTime(15_000);
    });
    expect(result.current.isRedirecting).toBe(false);
  });

  test("lets go when Back restores the page from the back/forward cache", () => {
    const { result } = startRedirect();
    restoreFromBackForwardCache();
    expect(result.current.isRedirecting).toBe(false);
  });

  test("stays busy on a pageshow that is not a back/forward cache restore", () => {
    const { result } = startRedirect();
    act(() => {
      window.dispatchEvent(new Event("pageshow"));
    });
    expect(result.current.isRedirecting).toBe(true);
  });

  test("stops listening once unmounted", () => {
    const removeListener = vi.spyOn(window, "removeEventListener");
    const { unmount } = startRedirect();
    unmount();
    expect(removeListener).toHaveBeenCalledWith(
      "pageshow",
      expect.any(Function),
    );
    expect(vi.getTimerCount()).toBe(0);
  });
});
