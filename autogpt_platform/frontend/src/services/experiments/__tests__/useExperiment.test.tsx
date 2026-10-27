import { getGetExperimentsGetExperimentAssignmentMockHandler } from "@/app/api/__generated__/endpoints/experiments/experiments.msw";
import { Experiment } from "@/app/api/__generated__/models/experiment";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { resetPostHogCaptureQueueForTests } from "@/services/analytics/posthog-capture";
import { environment } from "@/services/environment";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useExperiment } from "../useExperiment";

const posthogJS = vi.hoisted(() => ({
  __loaded: true,
  capturing: true,
  capture: vi.fn(),
  is_capturing() {
    return this.capturing;
  },
}));
vi.mock("posthog-js", () => ({ default: posthogJS }));

const PRICING = Experiment.SUBSCRIPTION_PRICING;

beforeEach(() => {
  signIn("user-1");
  posthogJS.capture.mockClear();
  posthogJS.capturing = true;
  sessionStorage.clear();
  vi.spyOn(environment, "isPostHogEnabled").mockReturnValue("phc_test");
});

afterEach(() => {
  resetPostHogCaptureQueueForTests();
  removeCookiebot();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  useAuthStore.setState({ user: null });
});

describe("useExperiment", () => {
  it("serves the arm the backend assigned and reports the exposure", async () => {
    serveArms(["yearly-pro"]);

    const { result } = renderHook(() => useExperiment(PRICING), {
      wrapper: withQueryClient(),
    });

    expect(result.current.isResolved).toBe(false);
    await waitFor(() =>
      expect(result.current).toEqual({
        variant: "yearly-pro",
        isResolved: true,
      }),
    );
    expect(posthogJS.capture).toHaveBeenCalledWith("$feature_flag_called", {
      $feature_flag: PRICING,
      $feature_flag_response: "yearly-pro",
    });
  });

  // On a full page load the arm can arrive before PostHog has opted in,
  // when posthog-js would drop the event.
  it("holds the exposure of a consented user until PostHog captures", async () => {
    configureCookiebot();
    installCookiebot({ statistics: true });
    posthogJS.capturing = false;
    serveArms(["yearly-pro"]);

    const { result } = renderHook(() => useExperiment(PRICING), {
      wrapper: withQueryClient(),
    });
    await waitFor(() => expect(result.current.variant).toBe("yearly-pro"));
    expect(posthogJS.capture).not.toHaveBeenCalled();

    posthogJS.capturing = true;

    await waitFor(() =>
      expect(posthogJS.capture).toHaveBeenCalledExactlyOnceWith(
        "$feature_flag_called",
        { $feature_flag: PRICING, $feature_flag_response: "yearly-pro" },
        { timestamp: expect.any(Date) },
      ),
    );
  });

  it("reports the exposure once per tab, however often the page remounts", async () => {
    serveArms(["yearly-pro", "yearly-pro"]);

    const first = renderHook(() => useExperiment(PRICING), {
      wrapper: withQueryClient(),
    });
    await waitFor(() => expect(posthogJS.capture).toHaveBeenCalledTimes(1));
    first.unmount();
    const second = renderHook(() => useExperiment(PRICING), {
      wrapper: withQueryClient(),
    });
    await waitFor(() =>
      expect(second.result.current.variant).toBe("yearly-pro"),
    );

    expect(posthogJS.capture).toHaveBeenCalledTimes(1);
  });

  it("reports each user's exposure, even when they share an arm", async () => {
    serveArms(["control", "control"]);

    renderHook(() => useExperiment(PRICING), { wrapper: withQueryClient() });
    await waitFor(() => expect(posthogJS.capture).toHaveBeenCalledTimes(1));
    act(() => signIn("user-2"));

    await waitFor(() => expect(posthogJS.capture).toHaveBeenCalledTimes(2));
  });

  it("resolves to no arm, and reports nothing, for a user it did not enrol", async () => {
    serveArms([null]);

    const { result } = renderHook(() => useExperiment(PRICING), {
      wrapper: withQueryClient(),
    });

    await waitFor(() => expect(result.current.isResolved).toBe(true));
    expect(result.current.variant).toBeNull();
    expect(posthogJS.capture).not.toHaveBeenCalled();
  });

  it("asks nothing for a signed-out visitor", async () => {
    const requests = serveArms(["yearly-pro"]);
    useAuthStore.setState({ user: null, isUserLoading: false });

    const { result } = renderHook(() => useExperiment(PRICING), {
      wrapper: withQueryClient(),
    });

    expect(result.current).toEqual({ variant: null, isResolved: true });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(requests.count).toBe(0);
  });

  it("reads each signed-in user's own arm", async () => {
    const requests = serveArms(["control", "yearly-pro"]);

    const { result } = renderHook(() => useExperiment(PRICING), {
      wrapper: withQueryClient(),
    });
    await waitFor(() => expect(result.current.variant).toBe("control"));
    act(() => signIn("user-2"));

    await waitFor(() => expect(result.current.variant).toBe("yearly-pro"));
    expect(requests.count).toBe(2);
  });
});

function serveArms(arms: Array<string | null>) {
  const requests = { count: 0 };
  server.use(
    getGetExperimentsGetExperimentAssignmentMockHandler(() => {
      const variant = arms[requests.count++];
      return variant === null
        ? null
        : {
            experiment_key: PRICING,
            variant,
            source: "backend",
            assigned_at: new Date("2026-10-11T00:00:00Z"),
          };
    }),
  );
  return requests;
}

function signIn(userID: string) {
  useAuthStore.setState({
    user: { id: userID, email: `${userID}@example.com`, user_metadata: {} },
    isUserLoading: false,
  });
}

function withQueryClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}
