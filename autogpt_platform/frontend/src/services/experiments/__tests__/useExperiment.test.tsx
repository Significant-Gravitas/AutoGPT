import { server } from "@/mocks/mock-server";
import { environment } from "@/services/environment";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  resetReportedAssignmentsForTests,
  useExperiment,
} from "../useExperiment";

const postHog = vi.hoisted(() => ({
  variant: undefined as string | boolean | undefined,
}));

const auth = vi.hoisted(() => ({
  user: null as { id: string } | null,
}));

vi.mock("@posthog/react", () => ({
  useFeatureFlagVariantKey: () => postHog.variant,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: auth.user, isUserLoading: false }),
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function captureAssignments() {
  const bodies: unknown[] = [];
  server.use(
    http.post("*/api/experiments/assignments", async ({ request }) => {
      bodies.push(await request.json());
      return HttpResponse.json({
        experiment_key: "pricing-test",
        variant: "yearly-pro",
        source: "posthog",
        assigned_at: "2026-09-02T00:00:00Z",
      });
    }),
  );
  return bodies;
}

beforeEach(() => {
  postHog.variant = undefined;
  auth.user = { id: "user-1" };
  resetReportedAssignmentsForTests();
  vi.spyOn(environment, "isPostHogEnabled").mockReturnValue("phc_test");
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useExperiment", () => {
  it("is unresolved while PostHog has not loaded the flag", () => {
    const { result } = renderHook(() => useExperiment("pricing-test"), {
      wrapper,
    });

    expect(result.current.isResolved).toBe(false);
    expect(result.current.variant).toBeNull();
  });

  it("resolves immediately when PostHog is disabled", () => {
    vi.spyOn(environment, "isPostHogEnabled").mockReturnValue(false);

    const { result } = renderHook(() => useExperiment("pricing-test"), {
      wrapper,
    });

    expect(result.current.isResolved).toBe(true);
    expect(result.current.variant).toBeNull();
  });

  it("reports a string variant to the backend once per user and experiment", async () => {
    const bodies = captureAssignments();
    postHog.variant = "yearly-pro";

    const first = renderHook(() => useExperiment("pricing-test"), { wrapper });
    expect(first.result.current).toEqual({
      variant: "yearly-pro",
      isResolved: true,
    });
    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      experiment_key: "pricing-test",
      variant: "yearly-pro",
      source: "posthog",
    });

    renderHook(() => useExperiment("pricing-test"), { wrapper });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(bodies).toHaveLength(1);
  });

  it("does not report users who are not enrolled or not signed in", async () => {
    const bodies = captureAssignments();

    postHog.variant = false;
    renderHook(() => useExperiment("pricing-test"), { wrapper });

    postHog.variant = "control";
    auth.user = null;
    renderHook(() => useExperiment("pricing-test"), { wrapper });

    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(bodies).toHaveLength(0);
  });
});
