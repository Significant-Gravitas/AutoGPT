import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { getPostV2NotifyImpersonationStartMockHandler200 } from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { ImpersonationState } from "@/lib/impersonation";
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
      result.current.startImpersonating(TARGET_USER_ID);
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
      result.current.startImpersonating(TARGET_USER_ID);
    });

    await waitFor(() => expect(reloadMock).toHaveBeenCalledTimes(1));
    expect(setSpy).toHaveBeenCalledWith(TARGET_USER_ID);
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
      result.current.startImpersonating("   ");
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
});
