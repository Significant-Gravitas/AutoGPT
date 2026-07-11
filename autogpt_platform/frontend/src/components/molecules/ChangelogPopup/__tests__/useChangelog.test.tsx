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
| [May 7 – June 10, 2026](${LATEST}.md) | Copilot upgrades |
| [Apr 1 – May 6, 2026](${PREV}.md) | Marketplace redesign |
`;

const ENTRY_MD: Record<string, string> = {
  [LATEST]: "# Latest release\n{% hint %}kept",
  [PREV]: "# Previous release",
};

beforeEach(() => {
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, ({ request }) => {
      const slug = new URL(request.url).searchParams.get("slug");
      return HttpResponse.text(slug ? (ENTRY_MD[slug] ?? "") : INDEX_MD);
    }),
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
  });

  it("openFullChangelog loads + cleans the markdown and marks the release seen", async () => {
    const { result } = renderHook(() => useChangelog());
    await waitFor(() => expect(result.current.allEntries).toHaveLength(2));

    act(() => result.current.openFullChangelog(result.current.allEntries[0]));
    expect(result.current.showFullChangelog).toBe(true);

    await waitFor(() =>
      expect(result.current.entryMarkdown).toContain("# Latest release"),
    );
    expect(result.current.entryMarkdown).not.toContain("{%");
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
  });

  it("selectEntry switches the rendered release", async () => {
    const { result } = renderHook(() => useChangelog());
    await waitFor(() => expect(result.current.allEntries).toHaveLength(2));

    act(() => result.current.openFullChangelog(result.current.allEntries[0]));
    act(() => result.current.selectEntry(result.current.allEntries[1]));

    await waitFor(() =>
      expect(result.current.entryMarkdown).toContain("# Previous release"),
    );
  });

  it("closeFullChangelog resets the modal state", async () => {
    const { result } = renderHook(() => useChangelog());
    await waitFor(() => expect(result.current.allEntries).toHaveLength(2));

    act(() => result.current.openFullChangelog());
    act(() => result.current.closeFullChangelog());

    expect(result.current.showFullChangelog).toBe(false);
    expect(result.current.selectedEntry).toBeNull();
    expect(result.current.entryMarkdown).toBeNull();
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
