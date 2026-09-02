import { server } from "@/mocks/mock-server";
import { environment } from "@/services/environment";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  resetExposuresForTests,
  useLaunchDarklyExperiment,
} from "../useLaunchDarklyExperiment";
import { resetReportedAssignmentsForTests } from "../useReportAssignment";

const ld = vi.hoisted(() => ({
  flags: {} as Record<string, unknown>,
  client: null as null | { waitForInitialization: () => Promise<void> },
}));

const auth = vi.hoisted(() => ({
  user: null as { id: string } | null,
}));

const capture = vi.hoisted(() => vi.fn());

vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => ld.flags,
  useLDClient: () => ld.client ?? undefined,
}));

vi.mock("posthog-js", () => ({ default: { capture } }));

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
        experiment_key: "onboarding-copy",
        variant: "b",
        source: "launchdarkly",
        assigned_at: "2026-09-02T00:00:00Z",
      });
    }),
  );
  return bodies;
}

beforeEach(() => {
  ld.flags = {};
  ld.client = { waitForInitialization: () => Promise.resolve() };
  auth.user = { id: "user-1" };
  capture.mockReset();
  resetExposuresForTests();
  resetReportedAssignmentsForTests();
  vi.spyOn(environment, "areFeatureFlagsEnabled").mockReturnValue(true);
  vi.spyOn(environment, "isPostHogEnabled").mockReturnValue("phc_test");
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useLaunchDarklyExperiment", () => {
  it("resolves after the LaunchDarkly client initialises", async () => {
    ld.flags = { "onboarding-copy": "b" };

    const { result } = renderHook(
      () => useLaunchDarklyExperiment("onboarding-copy"),
      { wrapper },
    );

    expect(result.current.isResolved).toBe(false);
    await waitFor(() => expect(result.current.isResolved).toBe(true));
    expect(result.current.variant).toBe("b");
  });

  it("reports the arm to the backend and to PostHog once", async () => {
    const bodies = captureAssignments();
    ld.flags = { "onboarding-copy": "b" };

    renderHook(() => useLaunchDarklyExperiment("onboarding-copy"), {
      wrapper,
    });

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      experiment_key: "onboarding-copy",
      variant: "b",
      source: "launchdarkly",
    });
    expect(capture).toHaveBeenCalledTimes(1);
    expect(capture).toHaveBeenCalledWith("experiment_exposed", {
      experiment_key: "onboarding-copy",
      variant: "b",
      provider: "launchdarkly",
      "$feature/onboarding-copy": "b",
    });

    renderHook(() => useLaunchDarklyExperiment("onboarding-copy"), {
      wrapper,
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(bodies).toHaveLength(1);
    expect(capture).toHaveBeenCalledTimes(1);
  });

  it("treats boolean or missing flags as not enrolled", async () => {
    const bodies = captureAssignments();
    ld.flags = { "onboarding-copy": true };

    const { result } = renderHook(
      () => useLaunchDarklyExperiment("onboarding-copy"),
      { wrapper },
    );

    await waitFor(() => expect(result.current.isResolved).toBe(true));
    expect(result.current.variant).toBeNull();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(bodies).toHaveLength(0);
    expect(capture).not.toHaveBeenCalled();
  });

  it("resolves immediately when feature flags are disabled", () => {
    vi.spyOn(environment, "areFeatureFlagsEnabled").mockReturnValue(false);
    ld.client = null;

    const { result } = renderHook(
      () => useLaunchDarklyExperiment("onboarding-copy"),
      { wrapper },
    );

    expect(result.current).toEqual({ variant: null, isResolved: true });
  });
});
