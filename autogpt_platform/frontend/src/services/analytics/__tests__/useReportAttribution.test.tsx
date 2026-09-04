import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  resetAttributionReportForTests,
  useReportAttribution,
} from "../useReportAttribution";

const REPORTED_KEY = "agpt_attribution_reported";

const auth = vi.hoisted(() => ({
  user: null as { id: string } | null,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: auth.user, isUserLoading: false }),
}));

vi.mock("@/services/analytics/attribution-payload", () => ({
  buildAttributionPayload: () => ({ anonymous_id: "anon-1" }),
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function captureReports(status = 200) {
  const bodies: unknown[] = [];
  server.use(
    http.post("*/api/analytics/attribution", async ({ request }) => {
      bodies.push(await request.json());
      if (status !== 200) return new HttpResponse(null, { status });
      return HttpResponse.json({ user_id: "user-1", anonymous_id: "anon-1" });
    }),
  );
  return bodies;
}

beforeEach(() => {
  window.localStorage.clear();
  auth.user = { id: "user-1" };
  resetAttributionReportForTests();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useReportAttribution", () => {
  it("reports once per user and remembers a successful report across loads", async () => {
    const bodies = captureReports();

    const { rerender } = renderHook(() => useReportAttribution(), { wrapper });
    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({ anonymous_id: "anon-1" });
    await waitFor(() =>
      expect(window.localStorage.getItem(REPORTED_KEY)).toBe("user-1"),
    );

    rerender();
    resetAttributionReportForTests();
    renderHook(() => useReportAttribution(), { wrapper });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(bodies).toHaveLength(1);
  });

  it("does nothing while signed out", async () => {
    const bodies = captureReports();
    auth.user = null;

    renderHook(() => useReportAttribution(), { wrapper });
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(bodies).toHaveLength(0);
    expect(window.localStorage.getItem(REPORTED_KEY)).toBeNull();
  });

  it("does not remember a failed report, so it can be retried", async () => {
    const bodies = captureReports(500);

    renderHook(() => useReportAttribution(), { wrapper });
    await waitFor(() => expect(bodies).toHaveLength(1));
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(window.localStorage.getItem(REPORTED_KEY)).toBeNull();

    const retried = captureReports();
    renderHook(() => useReportAttribution(), { wrapper });
    await waitFor(() => expect(retried).toHaveLength(1));
  });
});
