import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "../changelog-constants";
import { useChangelog } from "../useChangelog";

const LATEST = "v0-6-63";
const INDEX_MD = `
| Date | Highlights |
| ---- | ---------- |
| [May 7 – June 10](${LATEST}.md) | Copilot upgrades, new blocks |
| [Apr 1 – May 6](v0-6-58.md) | Marketplace redesign |
`;

beforeEach(() => {
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
  );
});

afterEach(() => {
  server.resetHandlers();
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

  it("dismiss marks the latest release seen after fading out", async () => {
    const { result } = renderHook(() => useChangelog());
    await waitFor(() => expect(result.current.latestEntry?.slug).toBe(LATEST));

    act(() => result.current.pauseAutoDismiss());
    act(() => result.current.resumeAutoDismiss());
    act(() => result.current.dismiss());

    await waitFor(
      () => expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST),
      { timeout: 1500 },
    );
  });
});
