import { server } from "@/mocks/mock-server";
import {
  focusManager,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  SESSION_LIST_QUERY_KEY,
  SESSION_LIST_REFETCH_INTERVAL_MS,
  useSessionList,
} from "../useSessionList";

// The app's query client marks data fresh for 60 s (`lib/react-query`), which
// is what decides whether a window focus refetches the older pages.
const APP_STALE_TIME_MS = 60_000;

let listed: { id: string; title: string }[] = [];
let listRequests: string[] = [];

function makeClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: APP_STALE_TIME_MS },
    },
  });
}

function renderSessionList(client: QueryClient) {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return renderHook(() => useSessionList(), { wrapper: Wrapper });
}

async function settle() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  listed = Array.from({ length: 75 }, (_, i) => ({
    id: `s${i}`,
    title: `Chat ${i}`,
  }));
  listRequests = [];
  server.use(
    http.get("*/api/chat/sessions", ({ request }) => {
      const url = new URL(request.url);
      const offset = Number(url.searchParams.get("offset") ?? "0");
      const limit = Number(url.searchParams.get("limit") ?? "50");
      listRequests.push(String(offset));
      return HttpResponse.json({
        sessions: listed.slice(offset, offset + limit).map((s) => ({
          ...s,
          is_processing: false,
          created_at: "2026-01-05T09:00:00Z",
          updated_at: "2026-01-05T09:00:00Z",
        })),
        total: listed.length,
      });
    }),
  );
});

afterEach(() => {
  focusManager.setFocused(undefined);
  vi.useRealTimers();
  server.resetHandlers();
});

describe("useSessionList — older pages", () => {
  it("still refreshes the older pages on focus after first-page polls", async () => {
    const client = makeClient();
    const { result } = renderSessionList(client);
    await waitFor(() => expect(result.current.sessions).toHaveLength(50));
    await act(async () => {
      await result.current.loadMore();
    });
    await waitFor(() => expect(result.current.sessions).toHaveLength(75));
    await settle();
    const loadedAt = client.getQueryState(
      SESSION_LIST_QUERY_KEY,
    )?.dataUpdatedAt;

    await act(async () => {
      vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    });
    await waitFor(() => expect(listRequests.at(-1)).toBe("0"));
    await settle();
    // Patching page 0 must not mark the pages it never fetched as fresh.
    expect(client.getQueryState(SESSION_LIST_QUERY_KEY)?.dataUpdatedAt).toBe(
      loadedAt,
    );

    // An older chat is renamed; page 0 is unchanged, so the poll alone
    // would never pick it up.
    listed[60] = { ...listed[60], title: "Renamed older chat" };
    listRequests = [];
    act(() => {
      focusManager.setFocused(false);
      focusManager.setFocused(true);
    });

    await waitFor(() =>
      expect(result.current.sessions.find((s) => s.id === "s60")?.title).toBe(
        "Renamed older chat",
      ),
    );
    expect(listRequests).toEqual(["0", "50"]);
  });

  it("keeps a first-page update that lands while an older page is loading", async () => {
    listed = Array.from({ length: 120 }, (_, i) => ({
      id: `s${i}`,
      title: `Chat ${i}`,
    }));
    let releaseThirdPage: () => void = () => {};
    const thirdPageHeld = new Promise<void>((resolve) => {
      releaseThirdPage = resolve;
    });
    server.use(
      http.get("*/api/chat/sessions", async ({ request }) => {
        const url = new URL(request.url);
        const offset = Number(url.searchParams.get("offset") ?? "0");
        const limit = Number(url.searchParams.get("limit") ?? "50");
        listRequests.push(String(offset));
        const page = listed.slice(offset, offset + limit);
        if (offset === 100) await thirdPageHeld;
        return HttpResponse.json({
          sessions: page.map((s) => ({
            ...s,
            is_processing: false,
            created_at: "2026-01-05T09:00:00Z",
            updated_at: "2026-01-05T09:00:00Z",
          })),
          total: listed.length,
        });
      }),
    );
    const client = makeClient();
    const { result } = renderSessionList(client);
    await waitFor(() => expect(result.current.sessions).toHaveLength(50));
    await act(async () => {
      await result.current.loadMore();
    });
    await waitFor(() => expect(result.current.sessions).toHaveLength(100));
    await settle();

    // "Load older chats" is in flight when the first-page poll comes back
    // with a renamed chat.
    let loadingThirdPage: Promise<unknown> = Promise.resolve();
    act(() => {
      loadingThirdPage = result.current.loadMore();
    });
    await waitFor(() => expect(listRequests).toContain("100"));
    listed[3] = { ...listed[3], title: "Renamed on page one" };
    listRequests = [];
    await act(async () => {
      vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    });
    await waitFor(() => expect(listRequests).toContain("0"));
    await settle();

    await act(async () => {
      releaseThirdPage();
      await loadingThirdPage;
    });
    await waitFor(() => expect(result.current.sessions).toHaveLength(120));
    await settle();

    expect(result.current.sessions.find((s) => s.id === "s3")?.title).toBe(
      "Renamed on page one",
    );
  });

  it("drops a first-page poll that a newer refetch of the list overtook", async () => {
    let holdNextFirstPage = false;
    let releaseFirstPage: () => void = () => {};
    server.use(
      http.get("*/api/chat/sessions", async ({ request }) => {
        const url = new URL(request.url);
        const offset = Number(url.searchParams.get("offset") ?? "0");
        const limit = Number(url.searchParams.get("limit") ?? "50");
        listRequests.push(String(offset));
        // The page is read when the request arrives, as the server would.
        const body = {
          sessions: listed.slice(offset, offset + limit).map((s) => ({
            ...s,
            is_processing: false,
            created_at: "2026-01-05T09:00:00Z",
            updated_at: "2026-01-05T09:00:00Z",
          })),
          total: listed.length,
        };
        if (offset === 0 && holdNextFirstPage) {
          holdNextFirstPage = false;
          await new Promise<void>((resolve) => {
            releaseFirstPage = resolve;
          });
        }
        return HttpResponse.json(body);
      }),
    );
    const client = makeClient();
    const { result } = renderSessionList(client);
    await waitFor(() => expect(result.current.sessions).toHaveLength(50));
    await act(async () => {
      await result.current.loadMore();
    });
    await waitFor(() => expect(result.current.sessions).toHaveLength(75));
    await settle();

    // The poll's request leaves, then the chat is renamed and the rename's
    // invalidation refetches the list before the poll's response arrives.
    holdNextFirstPage = true;
    listRequests = [];
    await act(async () => {
      vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    });
    await waitFor(() => expect(listRequests).toEqual(["0"]));
    listed[3] = { ...listed[3], title: "Renamed after the poll left" };
    await act(async () => {
      await client.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
    });
    await waitFor(() =>
      expect(result.current.sessions.find((s) => s.id === "s3")?.title).toBe(
        "Renamed after the poll left",
      ),
    );

    await act(async () => {
      releaseFirstPage();
    });
    await settle();

    expect(result.current.sessions.find((s) => s.id === "s3")?.title).toBe(
      "Renamed after the poll left",
    );
  });

  it("does not poll the first page straight away when the list remounts", async () => {
    const client = makeClient();
    const first = renderSessionList(client);
    await waitFor(() => expect(first.result.current.sessions).toHaveLength(50));
    await act(async () => {
      await first.result.current.loadMore();
    });
    await waitFor(() => expect(first.result.current.sessions).toHaveLength(75));
    await settle();
    first.unmount();

    // Back within the list's own 60 s staleTime, so nothing is due yet.
    await act(async () => {
      vi.advanceTimersByTime(15_000);
    });
    listRequests = [];
    const second = renderSessionList(client);
    expect(second.result.current.sessions).toHaveLength(75);
    await settle();

    expect(listRequests).toEqual([]);
  });
});

describe("useSessionList — whether the list loaded", () => {
  it("tells an empty list apart from one that failed to load", async () => {
    listed = [];
    const { result: empty } = renderSessionList(makeClient());
    await waitFor(() => expect(empty.current.isListLoaded).toBe(true));
    expect(empty.current.sessions).toEqual([]);

    server.use(
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    const { result: failed } = renderSessionList(makeClient());
    await settle();
    expect(failed.current.sessions).toEqual([]);
    expect(failed.current.isListLoaded).toBe(false);
  });
});
