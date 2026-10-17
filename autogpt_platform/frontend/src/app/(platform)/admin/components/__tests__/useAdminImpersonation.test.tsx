import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { getPostV2NotifyImpersonationStartMockHandler200 } from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { ImpersonationCookie, ImpersonationState } from "@/lib/impersonation";
import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";

import { useAdminImpersonation } from "../useAdminImpersonation";

const NOTIFY_URL =
  "http://localhost:3000/api/proxy/api/admin/impersonation/notify";
const TARGET_USER_ID = "target-user-id";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, useToast: () => ({ toast: toastMock }) };
});

const reloadMock = vi.fn();
let setSpy: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  // Impersonation is recorded against the signed-in admin who starts it.
  useAuthStore.setState({ user: { id: "admin-1", role: "admin" } as User });
  toastMock.mockClear();
  reloadMock.mockClear();
  // jsdom can't navigate; intercept the reload the hook fires on a successful swap.
  Object.defineProperty(window.location, "reload", {
    configurable: true,
    value: reloadMock,
  });
  // Avoid real sessionStorage/cookie writes; assert call instead.
  setSpy = vi.spyOn(ImpersonationState, "set").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useAdminImpersonation", () => {
  test("blocks the swap and shows a destructive toast when the audit alert fails (502)", async () => {
    server.use(
      http.post(NOTIFY_URL, () => new HttpResponse(null, { status: 502 })),
    );

    const { result } = renderHook(() => useAdminImpersonation(), { wrapper });

    await act(async () => {
      void result.current.startImpersonating(TARGET_USER_ID);
    });

    await waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "destructive" }),
      );
    });
    // The swap must NOT happen when the audit alert is not delivered.
    expect(setSpy).not.toHaveBeenCalled();
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("proceeds with the swap when the audit alert is delivered", async () => {
    server.use(
      getPostV2NotifyImpersonationStartMockHandler200({ alerted: true }),
    );

    const { result } = renderHook(() => useAdminImpersonation(), { wrapper });

    await act(async () => {
      void result.current.startImpersonating(TARGET_USER_ID);
    });

    await waitFor(() => expect(reloadMock).toHaveBeenCalledTimes(1));
    // Recorded against the admin who chose it, so no other account adopts it.
    expect(setSpy).toHaveBeenCalledWith(TARGET_USER_ID, "admin-1");
    expect(toastMock).not.toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  test("rejects an empty user id without calling the API or swapping", async () => {
    const apiHit = vi.fn();
    server.use(
      http.post(NOTIFY_URL, () => {
        apiHit();
        return new HttpResponse(JSON.stringify({ alerted: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );

    const { result } = renderHook(() => useAdminImpersonation(), { wrapper });

    await act(async () => {
      void result.current.startImpersonating("   ");
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: expect.stringMatching(/user id is required/i),
        variant: "destructive",
      }),
    );
    expect(apiHit).not.toHaveBeenCalled();
    expect(setSpy).not.toHaveBeenCalled();
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("logs no audit alert when no admin is signed in to own the swap", async () => {
    useAuthStore.setState({ user: null });
    const apiHit = vi.fn();
    server.use(
      http.post(NOTIFY_URL, () => {
        apiHit();
        return new HttpResponse(JSON.stringify({ alerted: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );

    const { result } = renderHook(() => useAdminImpersonation(), { wrapper });

    await act(async () => {
      void result.current.startImpersonating(TARGET_USER_ID);
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't start impersonation",
        variant: "destructive",
      }),
    );
    expect(apiHit).not.toHaveBeenCalled();
    expect(setSpy).not.toHaveBeenCalled();
    expect(reloadMock).not.toHaveBeenCalled();
  });

  test("shows the active impersonation once the admin is known after a reload", async () => {
    // The reload that starts impersonation comes back before auth has loaded.
    useAuthStore.setState({ user: null });
    // Stored directly: this file stubs ImpersonationState.set.
    ImpersonationCookie.set({ target: TARGET_USER_ID, owner: "admin-1" });

    const { result } = renderHook(() => useAdminImpersonation(), { wrapper });
    expect(result.current.isImpersonating).toBe(false);

    act(() => {
      useAuthStore.setState({ user: { id: "admin-1", role: "admin" } as User });
    });

    expect(result.current.impersonatedUserId).toBe(TARGET_USER_ID);
    expect(result.current.isImpersonating).toBe(true);
  });

  test("drops a re-entrant second start so only one audit alert is sent", async () => {
    let calls = 0;
    server.use(
      http.post(NOTIFY_URL, async () => {
        calls++;
        await delay(10);
        return new HttpResponse(JSON.stringify({ alerted: true }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );

    const { result } = renderHook(() => useAdminImpersonation(), { wrapper });

    await act(async () => {
      // Fire twice before the first resolves; the re-entry guard drops the 2nd.
      void result.current.startImpersonating(TARGET_USER_ID);
      void result.current.startImpersonating(TARGET_USER_ID);
    });

    await waitFor(() => expect(reloadMock).toHaveBeenCalled());
    expect(calls).toBe(1);
  });
});
