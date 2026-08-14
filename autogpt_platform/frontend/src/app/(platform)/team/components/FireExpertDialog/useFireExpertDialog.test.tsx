import {
  getGetExpertQueryKey,
  getListExpertIdentitiesQueryKey,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import {
  getArchiveExpertMockHandler,
  getGetExpertDetachPreviewMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { useFireExpertDialog } from "./useFireExpertDialog";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn() }),
}));

function makeClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
}

function makeWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

function containsKey(calls: unknown[][], key: readonly unknown[]) {
  return calls.some(([arg]) => {
    const queryKey = (arg as { queryKey?: unknown })?.queryKey;
    return JSON.stringify(queryKey) === JSON.stringify(key);
  });
}

describe("useFireExpertDialog invalidation", () => {
  it("invalidates roster, per-expert, home and schedule caches on a successful fire", async () => {
    server.use(
      getGetExpertDetachPreviewMockHandler({
        schedule_names: [],
        trigger_names: [],
      }),
      getArchiveExpertMockHandler(),
    );

    const client = makeClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");
    const onFired = vi.fn();

    const { result } = renderHook(
      () =>
        useFireExpertDialog({
          expertId: "expert-maria",
          expertName: "Maria",
          open: true,
          onClose: vi.fn(),
          onFired,
        }),
      { wrapper: makeWrapper(client) },
    );

    await waitFor(() => expect(result.current.isPreviewReady).toBe(true));

    act(() => result.current.handleFire());

    await waitFor(() => expect(onFired).toHaveBeenCalled());

    const calls = invalidateSpy.mock.calls;
    expect(containsKey(calls, ["/api/experts"])).toBe(true);
    expect(containsKey(calls, getListExpertIdentitiesQueryKey())).toBe(true);
    expect(containsKey(calls, getGetExpertQueryKey("expert-maria"))).toBe(true);
    expect(containsKey(calls, getGetHomeDashboardQueryKey())).toBe(true);
    expect(
      containsKey(calls, getGetV1ListExecutionSchedulesForAUserQueryKey()),
    ).toBe(true);
    expect(containsKey(calls, ["/api/library/agents"])).toBe(true);
  });

  it("does not fire before the preview has resolved", async () => {
    const archiveSpy = vi.fn();
    server.use(
      getGetExpertDetachPreviewMockHandler({
        schedule_names: [],
        trigger_names: [],
      }),
      getArchiveExpertMockHandler(archiveSpy),
    );

    const client = makeClient();
    const { result } = renderHook(
      () =>
        useFireExpertDialog({
          expertId: "expert-maria",
          expertName: "Maria",
          open: true,
          onClose: vi.fn(),
        }),
      { wrapper: makeWrapper(client) },
    );

    // Preview not yet ready on first render — a click must be a no-op.
    expect(result.current.isPreviewReady).toBe(false);
    act(() => result.current.handleFire());
    expect(archiveSpy).not.toHaveBeenCalled();
  });

  it("does not fire while a cached preview is being retried", async () => {
    const archiveSpy = vi.fn();
    server.use(
      getGetExpertDetachPreviewMockHandler({
        schedule_names: ["Weekly report"],
        trigger_names: [],
      }),
      getArchiveExpertMockHandler(archiveSpy),
    );

    const client = makeClient();
    const { result } = renderHook(
      () =>
        useFireExpertDialog({
          expertId: "expert-maria",
          expertName: "Maria",
          open: true,
          onClose: vi.fn(),
        }),
      { wrapper: makeWrapper(client) },
    );
    await waitFor(() => expect(result.current.isPreviewReady).toBe(true));

    let releaseRetry: () => void = () => undefined;
    const retryGate = new Promise<void>((resolve) => {
      releaseRetry = resolve;
    });
    server.use(
      http.get("*/api/experts/:expertId/detach-preview", async () => {
        await retryGate;
        return HttpResponse.json({
          schedule_names: ["Weekly report"],
          trigger_names: [],
        });
      }),
    );

    let retryPromise: Promise<unknown> | undefined;
    act(() => {
      retryPromise = result.current.retryPreview();
    });
    await waitFor(() => expect(result.current.isPreviewLoading).toBe(true));

    act(() => result.current.handleFire());
    expect(archiveSpy).not.toHaveBeenCalled();

    await act(async () => {
      releaseRetry();
      await retryPromise;
    });
    await waitFor(() => expect(result.current.isPreviewLoading).toBe(false));

    act(() => result.current.handleFire());
    await waitFor(() => expect(archiveSpy).toHaveBeenCalledTimes(1));
  });
});
