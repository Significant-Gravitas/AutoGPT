import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "../changelog-constants";
import { useChangelog } from "../useChangelog";

const LATEST = "v0-6-63";
const PREV = "v0-6-58";
const INDEX_MD = `
| Release | Highlights |
| --- | --- |
| [May 7 – June 10, 2026](/docs/platform/changelog/changelog/${LATEST}.md) | Copilot upgrades |
| [Apr 1 – May 6, 2026](/docs/platform/changelog/changelog/${PREV}.md) | Marketplace redesign |
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
  it("parses the changelog index into entries", async () => {
    const { result } = renderHook(() => useChangelog());

    await waitFor(() => expect(result.current.allEntries).toHaveLength(2));
    expect(result.current.latestEntry?.slug).toBe(LATEST);
    expect(result.current.latestEntry?.url).toContain(`/changelog/${LATEST}`);
  });

  it("openFullChangelog shows the modal and marks the latest release seen", async () => {
    const { result } = renderHook(() => useChangelog());
    await waitFor(() => expect(result.current.latestEntry?.slug).toBe(LATEST));

    act(() => result.current.openFullChangelog());

    expect(result.current.showFullChangelog).toBe(true);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
  });

  it("closeFullChangelog hides the modal", async () => {
    const { result } = renderHook(() => useChangelog());
    await waitFor(() => expect(result.current.allEntries).toHaveLength(2));

    act(() => result.current.openFullChangelog());
    act(() => result.current.closeFullChangelog());

    expect(result.current.showFullChangelog).toBe(false);
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
