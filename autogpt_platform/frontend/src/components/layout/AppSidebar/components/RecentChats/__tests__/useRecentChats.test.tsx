import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";

import { useRecentChats } from "../useRecentChats";

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: unknown[]) => toastMock(...args),
  };
});

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <NuqsTestingAdapter>{children}</NuqsTestingAdapter>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

function sessionsHandler() {
  return http.get("*/api/chat/sessions", () =>
    HttpResponse.json({
      sessions: [
        {
          id: "s1",
          title: "Chat one",
          is_processing: false,
          created_at: "2026-06-30T00:00:00Z",
          updated_at: "2026-06-30T00:00:00Z",
        },
      ],
      total: 1,
    }),
  );
}

beforeEach(() => {
  toastMock.mockClear();
  server.use(sessionsHandler());
});

afterEach(() => {
  server.resetHandlers();
});

describe("useRecentChats — rename", () => {
  it("enters and exits editing mode via startRename/cancelRename", async () => {
    const { result } = renderHook(() => useRecentChats(), {
      wrapper: makeWrapper(),
    });

    act(() => result.current.startRename("s1", "Chat one"));
    expect(result.current.editingSessionId).toBe("s1");
    expect(result.current.editingTitle).toBe("Chat one");

    act(() => result.current.cancelRename());
    expect(result.current.editingSessionId).toBeNull();
  });

  it("submits a trimmed title and clears editing on success", async () => {
    server.use(
      http.patch("*/api/chat/sessions/:id/title", () => HttpResponse.json({})),
    );
    const { result } = renderHook(() => useRecentChats(), {
      wrapper: makeWrapper(),
    });

    act(() => result.current.startRename("s1", "Chat one"));
    act(() => result.current.setEditingTitle("  Renamed  "));
    act(() => result.current.submitRename("s1"));

    await waitFor(() => {
      expect(result.current.editingSessionId).toBeNull();
    });
  });

  it("just exits editing when the submitted title is blank", () => {
    const { result } = renderHook(() => useRecentChats(), {
      wrapper: makeWrapper(),
    });

    act(() => result.current.startRename("s1", "Chat one"));
    act(() => result.current.setEditingTitle("   "));
    act(() => result.current.submitRename("s1"));

    expect(result.current.editingSessionId).toBeNull();
  });

  it("toasts and exits editing when the rename request fails", async () => {
    server.use(
      http.patch("*/api/chat/sessions/:id/title", () =>
        HttpResponse.json({ detail: "nope" }, { status: 422 }),
      ),
    );
    const { result } = renderHook(() => useRecentChats(), {
      wrapper: makeWrapper(),
    });

    act(() => result.current.startRename("s1", "Chat one"));
    act(() => result.current.setEditingTitle("Renamed"));
    act(() => result.current.submitRename("s1"));

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to rename chat",
          variant: "destructive",
        }),
      );
    });
    expect(result.current.editingSessionId).toBeNull();
  });
});

describe("useRecentChats — export", () => {
  it("toasts success after a successful export", async () => {
    server.use(
      http.get("*/api/chat/sessions/:id", () =>
        HttpResponse.json({ messages: [], has_more_messages: false }),
      ),
    );
    const { result } = renderHook(() => useRecentChats(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.exportChat("s1", "Chat one");
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Chat exported" }),
    );
    expect(result.current.exportingIds.has("s1")).toBe(false);
  });

  it("toasts a destructive error when the export fetch fails", async () => {
    server.use(
      http.get("*/api/chat/sessions/:id", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    const { result } = renderHook(() => useRecentChats(), {
      wrapper: makeWrapper(),
    });

    await act(async () => {
      await result.current.exportChat("s1", "Chat one");
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Export failed",
        variant: "destructive",
      }),
    );
  });
});
