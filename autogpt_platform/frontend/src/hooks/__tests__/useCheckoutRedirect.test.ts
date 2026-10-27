import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";
import {
  resetCheckoutRedirect,
  useCheckoutRedirect,
} from "../useCheckoutRedirect";

describe("useCheckoutRedirect", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    resetCheckoutRedirect();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  function renderTwoButtons() {
    const first = renderHook(() => useCheckoutRedirect());
    const second = renderHook(() => useCheckoutRedirect());
    return { first, second };
  }

  test("a redirect from one button makes every button on the page busy", () => {
    const { first, second } = renderTwoButtons();

    act(() => first.result.current.markRedirecting());

    expect(first.result.current.isRedirecting).toBe(true);
    expect(second.result.current.isRedirecting).toBe(true);
    expect(first.result.current.isRedirectingHere).toBe(true);
    expect(second.result.current.isRedirectingHere).toBe(false);
  });

  test("a button mounted during a redirect starts out busy", () => {
    const { first } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());

    const late = renderHook(() => useCheckoutRedirect());

    expect(late.result.current.isRedirecting).toBe(true);
  });

  test("stays busy after the button that started it unmounts", () => {
    const { first, second } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());

    first.unmount();

    expect(second.result.current.isRedirecting).toBe(true);
  });

  test("listens for pageshow once, however many buttons are on the page", () => {
    const addListener = vi.spyOn(window, "addEventListener");
    const removeListener = vi.spyOn(window, "removeEventListener");
    const { first, second } = renderTwoButtons();

    act(() => first.result.current.markRedirecting());
    act(() => second.result.current.markRedirecting());

    const pageshow = addListener.mock.calls.filter(
      ([type]) => type === "pageshow",
    );
    expect(pageshow).toHaveLength(1);

    restoreFromBackForwardCache();

    expect(removeListener).toHaveBeenCalledWith("pageshow", pageshow[0][1]);
  });

  test("stays busy until the page leaves", () => {
    const { first, second } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());

    act(() => {
      vi.advanceTimersByTime(19_999);
    });

    expect(first.result.current.isRedirecting).toBe(true);
    expect(second.result.current.isRedirecting).toBe(true);
  });

  test("lets every button go after 20 seconds if the page never leaves", () => {
    const { first, second } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());

    act(() => {
      vi.advanceTimersByTime(20_000);
    });

    expect(first.result.current.isRedirecting).toBe(false);
    expect(second.result.current.isRedirecting).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });

  test("lets every button go when Back restores the page from the back/forward cache", () => {
    const { first, second } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());

    restoreFromBackForwardCache();

    expect(first.result.current.isRedirecting).toBe(false);
    expect(second.result.current.isRedirecting).toBe(false);
    expect(first.result.current.isRedirectingHere).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });

  test("stays busy on a pageshow that is not a back/forward cache restore", () => {
    const { first } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());

    act(() => {
      window.dispatchEvent(new Event("pageshow"));
    });

    expect(first.result.current.isRedirecting).toBe(true);
  });

  test("a later redirect from another button is not this button's", () => {
    const { first, second } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());
    restoreFromBackForwardCache();

    act(() => second.result.current.markRedirecting());

    expect(first.result.current.isRedirectingHere).toBe(false);
    expect(second.result.current.isRedirectingHere).toBe(true);
  });

  test("resetCheckoutRedirect clears the redirect, its timer and its listener", () => {
    const removeListener = vi.spyOn(window, "removeEventListener");
    const { first } = renderTwoButtons();
    act(() => first.result.current.markRedirecting());

    act(() => resetCheckoutRedirect());

    expect(first.result.current.isRedirecting).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
    expect(removeListener).toHaveBeenCalledWith(
      "pageshow",
      expect.any(Function),
    );
  });
});

// The store outlives a test, so the global test setup resets it after each
// one; without that, a redirect left by one test would disable the buttons of
// the next.
describe("useCheckoutRedirect between tests", () => {
  test("a test can leave a redirect running", () => {
    const { result } = renderHook(() => useCheckoutRedirect());
    act(() => result.current.markRedirecting());
    expect(result.current.isRedirecting).toBe(true);
  });

  test("the next test starts with no redirect", () => {
    const { result } = renderHook(() => useCheckoutRedirect());
    expect(result.current.isRedirecting).toBe(false);
  });
});
