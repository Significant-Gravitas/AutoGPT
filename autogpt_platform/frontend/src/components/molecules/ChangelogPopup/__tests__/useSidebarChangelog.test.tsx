import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "../changelog-constants";
import { useSidebarChangelog } from "../useSidebarChangelog";

const LATEST = "v0-6-63";
const PREV = "v0-6-58";
const INDEX_MD = `
| Release | Highlights |
| --- | --- |
| [May 7 – June 10, 2026](https://agpt.co/docs/platform/changelog/changelog/${LATEST}) | Copilot upgrades |
| [Apr 1 – May 6, 2026](https://agpt.co/docs/platform/changelog/changelog/${PREV}) | Marketplace redesign |
`;

function mdUrl(slug: string) {
  return `https://agpt.co/docs/platform/changelog/changelog/${slug}.md`;
}

beforeEach(() => {
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
    http.get(mdUrl(LATEST), () =>
      HttpResponse.text("# Latest release\n{% hint %}kept"),
    ),
    http.get(mdUrl(PREV), () => HttpResponse.text("# Previous release")),
  );
});

afterEach(() => {
  server.resetHandlers();
});

describe("useSidebarChangelog", () => {
  it("loads entries and flags an unseen latest release", async () => {
    const { result } = renderHook(() => useSidebarChangelog());

    await waitFor(() => expect(result.current.entries).toHaveLength(2));
    expect(result.current.hasUnseen).toBe(true);
  });

  it("treats an already-seen latest release as seen", async () => {
    window.localStorage.setItem(STORAGE_KEY, LATEST);
    const { result } = renderHook(() => useSidebarChangelog());

    await waitFor(() => expect(result.current.entries).toHaveLength(2));
    expect(result.current.hasUnseen).toBe(false);
  });

  it("open() selects the latest entry, loads its markdown, and marks it seen", async () => {
    const { result } = renderHook(() => useSidebarChangelog());
    await waitFor(() => expect(result.current.entries).toHaveLength(2));

    act(() => result.current.open());

    expect(result.current.isOpen).toBe(true);
    expect(result.current.selectedEntry?.slug).toBe(LATEST);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
    expect(result.current.hasUnseen).toBe(false);

    await waitFor(() => expect(result.current.isLoadingMarkdown).toBe(false));
    expect(result.current.entryMarkdown).toContain("# Latest release");
    // GitBook liquid tags are stripped by the shared cleaner.
    expect(result.current.entryMarkdown).not.toContain("{%");
  });

  it("selectEntry() switches the rendered release", async () => {
    const { result } = renderHook(() => useSidebarChangelog());
    await waitFor(() => expect(result.current.entries).toHaveLength(2));

    act(() => result.current.selectEntry(result.current.entries[1]));
    expect(result.current.selectedEntry?.slug).toBe(PREV);

    await waitFor(() =>
      expect(result.current.entryMarkdown).toContain("# Previous release"),
    );
  });

  it("close() resets the modal state", async () => {
    const { result } = renderHook(() => useSidebarChangelog());
    await waitFor(() => expect(result.current.entries).toHaveLength(2));

    act(() => result.current.open());
    act(() => result.current.close());

    expect(result.current.isOpen).toBe(false);
    expect(result.current.selectedEntry).toBeNull();
    expect(result.current.entryMarkdown).toBeNull();
  });
});
