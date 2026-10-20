import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "../changelog-constants";
import { useChangelog } from "../useChangelog";

const LATEST = "v0-6-63";
const INDEX_MD = `
| Date | Highlights |
| ---- | ---------- |
| [May 7 – June 10](/docs/platform/changelog/changelog/${LATEST}.md) | Copilot upgrades, new blocks |
| [Apr 1 – May 6](/docs/platform/changelog/changelog/v0-6-58.md) | Marketplace redesign |
`;

beforeEach(() => {
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
  );
});

afterEach(() => {
  server.resetHandlers();
  vi.useRealTimers();
});

describe("useChangelog", () => {
  it("loads the latest release from the index", async () => {
    const { result } = renderHook(() => useChangelog());

    await waitFor(() => expect(result.current.latestEntry?.slug).toBe(LATEST));
    expect(result.current.latestEntry?.url).toContain(`/changelog/${LATEST}`);
  });

  it("does not surface a release the user has already seen", async () => {
    window.localStorage.setItem(STORAGE_KEY, LATEST);
    const { result } = renderHook(() => useChangelog());

    await waitFor(() => expect(result.current.latestEntry?.slug).toBe(LATEST));
    // Give the reveal delay a chance to elapse — it must stay hidden.
    await new Promise((resolve) => setTimeout(resolve, 1800));
    expect(result.current.isVisible).toBe(false);
  });

  it("does not mark a release seen if unmounted before it appears", async () => {
    const { result, unmount } = renderHook(() => useChangelog());
    await waitFor(() => expect(result.current.latestEntry?.slug).toBe(LATEST));

    expect(result.current.isVisible).toBe(false);
    unmount();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it.each(["visible", "fading"])(
    "does not repeat a release after unmounting while %s",
    async (phase) => {
      vi.useFakeTimers();
      const { result, unmount } = renderHook(() => useChangelog());
      await vi.waitFor(() => expect(result.current.isVisible).toBe(true), {
        timeout: 3000,
      });

      if (phase === "fading") act(() => result.current.dismiss());
      unmount();

      expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
      const revisit = renderHook(() => useChangelog());
      await vi.waitFor(() =>
        expect(revisit.result.current.latestEntry?.slug).toBe(LATEST),
      );
      await act(() => vi.advanceTimersByTimeAsync(2000));
      expect(revisit.result.current.isVisible).toBe(false);
    },
  );

  it("dismiss hides the popup after fading out", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useChangelog());
    await vi.waitFor(() => expect(result.current.isVisible).toBe(true), {
      timeout: 3000,
    });

    act(() => result.current.dismiss());
    expect(result.current.isFading).toBe(true);
    expect(result.current.isVisible).toBe(true);
    await act(() => vi.advanceTimersByTimeAsync(500));
    expect(result.current.isVisible).toBe(false);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
  });
});
