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

  it("open() opens the modal and marks the latest release seen", async () => {
    const { result } = renderHook(() => useSidebarChangelog());
    await waitFor(() => expect(result.current.entries).toHaveLength(2));

    act(() => result.current.open());

    expect(result.current.isOpen).toBe(true);
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST);
    expect(result.current.hasUnseen).toBe(false);
  });

  it("close() closes the modal", async () => {
    const { result } = renderHook(() => useSidebarChangelog());
    await waitFor(() => expect(result.current.entries).toHaveLength(2));

    act(() => result.current.open());
    act(() => result.current.close());

    expect(result.current.isOpen).toBe(false);
  });

  it("open() is a no-op until the index has loaded", () => {
    const { result } = renderHook(() => useSidebarChangelog());

    expect(result.current.entries).toHaveLength(0);
    act(() => result.current.open());

    expect(result.current.isOpen).toBe(false);
  });
});
