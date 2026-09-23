import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { ReactNode, StrictMode, useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  resetReportedAssignmentsForTests,
  useReportAssignment,
} from "../useReportAssignment";

const auth = vi.hoisted(() => ({
  user: { id: "user-1" } as { id: string } | null,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: auth.user, isUserLoading: false }),
}));

interface Props {
  children: ReactNode;
}

function Wrapper({ children }: Props) {
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { mutations: { retry: false } } }),
  );
  return (
    <StrictMode>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </StrictMode>
  );
}

function useAssignment(variant = "control") {
  useReportAssignment({
    experimentKey: "pricing-test",
    variant,
    isResolved: true,
    source: "posthog",
  });
}

beforeEach(() => {
  auth.user = { id: "user-1" };
  resetReportedAssignmentsForTests();
});

describe("useReportAssignment retry", () => {
  it("retries a transient failure without remounting or changing inputs", async () => {
    const bodies: unknown[] = [];
    server.use(
      http.post("*/api/experiments/assignments", async ({ request }) => {
        bodies.push(await request.json());
        if (bodies.length === 1) return new HttpResponse(null, { status: 503 });
        return HttpResponse.json({ variant: "control" });
      }),
    );

    renderHook(() => useAssignment(), { wrapper: Wrapper });
    await waitFor(() => expect(bodies).toHaveLength(2), { timeout: 2500 });
    expect(bodies[0]).toEqual(bodies[1]);

    renderHook(() => useAssignment(), { wrapper: Wrapper });
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(bodies).toHaveLength(2);
  });

  it("limits retries and allows a later mount to try again", async () => {
    const attempts = vi.fn();
    server.use(
      http.post("*/api/experiments/assignments", () => {
        attempts();
        return new HttpResponse(null, { status: 503 });
      }),
    );
    const first = renderHook(() => useAssignment(), { wrapper: Wrapper });
    await waitFor(() => expect(attempts).toHaveBeenCalledTimes(3), {
      timeout: 4500,
    });
    await new Promise((resolve) => setTimeout(resolve, 100));
    first.unmount();

    server.use(
      http.post("*/api/experiments/assignments", () => {
        attempts();
        return HttpResponse.json({ variant: "control" });
      }),
    );
    renderHook(() => useAssignment(), { wrapper: Wrapper });
    await waitFor(() => expect(attempts).toHaveBeenCalledTimes(4));
  });

  it("replaces an in-flight report without letting its late failure erase the new claim", async () => {
    const bodies: unknown[] = [];
    let releaseFirst: () => void = () => {};
    const firstResponse = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const firstFinished = vi.fn();
    server.use(
      http.post("*/api/experiments/assignments", async ({ request }) => {
        bodies.push(await request.json());
        if (bodies.length === 1) {
          await firstResponse;
          firstFinished();
          return new HttpResponse(null, { status: 503 });
        }
        return HttpResponse.json({ variant: "control" });
      }),
    );

    const first = renderHook(() => useAssignment(), { wrapper: Wrapper });
    await waitFor(() => expect(bodies).toHaveLength(1));
    first.unmount();
    renderHook(() => useAssignment(), { wrapper: Wrapper });
    try {
      await waitFor(() => expect(bodies).toHaveLength(2));
    } finally {
      releaseFirst();
    }
    await waitFor(() => expect(firstFinished).toHaveBeenCalledOnce());
    await new Promise((resolve) => setTimeout(resolve, 30));

    renderHook(() => useAssignment(), { wrapper: Wrapper });
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(bodies).toHaveLength(2);
  });

  it("does not retry rejected input", async () => {
    const attempts = vi.fn();
    server.use(
      http.post("*/api/experiments/assignments", () => {
        attempts();
        return new HttpResponse(null, { status: 422 });
      }),
    );
    renderHook(() => useAssignment(), { wrapper: Wrapper });
    await waitFor(() => expect(attempts).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => setTimeout(resolve, 1200));
    expect(attempts).toHaveBeenCalledTimes(1);
  });

  it("cancels a scheduled retry on unmount", async () => {
    const attempts = vi.fn();
    server.use(
      http.post("*/api/experiments/assignments", () => {
        attempts();
        return new HttpResponse(null, { status: 503 });
      }),
    );
    const { unmount } = renderHook(() => useAssignment(), { wrapper: Wrapper });
    await waitFor(() => expect(attempts).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => setTimeout(resolve, 20));
    unmount();
    await new Promise((resolve) => setTimeout(resolve, 1200));
    expect(attempts).toHaveBeenCalledTimes(1);
  });

  it("does not retry the previous user's variant after an account switch", async () => {
    const bodies: unknown[] = [];
    server.use(
      http.post("*/api/experiments/assignments", async ({ request }) => {
        bodies.push(await request.json());
        if (bodies.length === 1) return new HttpResponse(null, { status: 503 });
        return HttpResponse.json({ variant: "treatment" });
      }),
    );
    const { rerender } = renderHook(({ variant }) => useAssignment(variant), {
      initialProps: { variant: "control" },
      wrapper: Wrapper,
    });
    await waitFor(() => expect(bodies).toHaveLength(1));
    await new Promise((resolve) => setTimeout(resolve, 20));

    auth.user = { id: "user-2" };
    rerender({ variant: "treatment" });
    await waitFor(() => expect(bodies).toHaveLength(2));
    await new Promise((resolve) => setTimeout(resolve, 1200));
    expect(bodies).toEqual([
      { experiment_key: "pricing-test", variant: "control", source: "posthog" },
      {
        experiment_key: "pricing-test",
        variant: "treatment",
        source: "posthog",
      },
    ]);
  });
});
