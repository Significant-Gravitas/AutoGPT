import { beforeEach, describe, expect, test, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import OnboardingProvider from "../onboarding-provider";

const routerReplace = vi.fn();
let mockPathname = "/signup";
let mockSearchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: routerReplace,
    push: vi.fn(),
    refresh: vi.fn(),
  }),
  useSearchParams: () => mockSearchParams,
  usePathname: () => mockPathname,
}));

let mockIsLoggedIn = true;
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: mockIsLoggedIn,
    isUserLoading: false,
    user: mockIsLoggedIn ? { id: "test-user", email: "u@example.com" } : null,
  }),
}));

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    onWebSocketMessage: () => () => {},
    connectWebSocket: () => {},
  }),
}));

vi.mock("@/hooks/useOnboardingTimezoneDetection", () => ({
  useOnboardingTimezoneDetection: () => undefined,
}));

let mockIsCompleted = false;
let mockCompletedStatus = 200;
const completedCallCount = { value: 0 };
vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1CheckIfOnboardingIsCompleted: () => {
    completedCallCount.value += 1;
    return Promise.resolve({
      status: mockCompletedStatus,
      data: { is_completed: mockIsCompleted },
    });
  },
  getV1OnboardingState: () =>
    Promise.resolve({ status: 200, data: { completedSteps: [] } }),
  patchV1UpdateOnboardingState: () => Promise.resolve({ status: 200 }),
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
}));

describe("OnboardingProvider routing — logged-in user", () => {
  beforeEach(() => {
    routerReplace.mockClear();
    mockSearchParams = new URLSearchParams();
    mockPathname = "/signup";
    mockIsLoggedIn = true;
    mockIsCompleted = false;
    mockCompletedStatus = 200;
    completedCallCount.value = 0;
  });

  test("incomplete user on /signup is redirected to /onboarding", async () => {
    mockPathname = "/signup";
    mockIsCompleted = false;

    render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
  });

  test("completed user on /signup is redirected to /copilot", async () => {
    mockPathname = "/signup";
    mockIsCompleted = true;

    render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/copilot"));
  });

  test("completed user on /login is redirected to /copilot", async () => {
    mockPathname = "/login";
    mockIsCompleted = true;

    render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/copilot"));
  });

  test("a safe ?next= deep link defers to the auth page — no redirect from provider", async () => {
    mockPathname = "/login";
    mockSearchParams = new URLSearchParams({ next: "/library" });
    mockIsCompleted = true;

    render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    // Wait long enough for any pending async redirect to land
    await new Promise((r) => setTimeout(r, 30));
    expect(routerReplace).not.toHaveBeenCalled();
    // The completion check should also be skipped — we never hit the API.
    expect(completedCallCount.value).toBe(0);
  });

  test("an unsafe ?next= value is treated as no deep link and the provider routes normally", async () => {
    // `sanitizeAuthNext` drops absolute URLs as null. Without the alignment
    // fix, OnboardingProvider would defer (assuming the auth page handles it)
    // while the auth page silently drops the value — deadlocking the user.
    mockPathname = "/signup";
    mockSearchParams = new URLSearchParams({ next: "https://phishing.site" });
    mockIsCompleted = false;

    render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
  });

  test("incomplete user on /reset-password stays put (recovery session must set password first)", async () => {
    // A Supabase recovery link signs the user in and lands them on
    // /reset-password. Bouncing them to /onboarding would skip the password
    // change entirely, turning the reset link into a one-time sign-in link.
    mockPathname = "/reset-password";
    mockIsCompleted = false;

    render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    await new Promise((r) => setTimeout(r, 30));
    expect(routerReplace).not.toHaveBeenCalled();
    expect(completedCallCount.value).toBe(0);
  });

  test("a persistently failing completion check stops retrying as the user navigates", async () => {
    // An account with no platform User row 500s this check on every call. The
    // init guard used to be re-armed on every failure, so each navigation shot
    // off another doomed request and the user generated 5xx for as long as they
    // browsed. Retries are bounded now: enough for a transient blip to recover,
    // not enough to keep a broken account hammering the endpoint.
    mockCompletedStatus = 500;
    mockPathname = "/copilot";

    const { rerender } = render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    await waitFor(() => expect(completedCallCount.value).toBe(1));

    for (const path of [
      "/library",
      "/marketplace",
      "/copilot",
      "/library",
      "/profile",
      "/marketplace",
    ]) {
      mockPathname = path;
      rerender(
        <OnboardingProvider>
          <div data-testid="child" />
        </OnboardingProvider>,
      );
      await new Promise((r) => setTimeout(r, 10));
    }

    // The first attempt plus MAX_INIT_ATTEMPTS retries, and then silence —
    // not one request per navigation.
    expect(completedCallCount.value).toBe(4);
  });

  test("incomplete user already on /onboarding stays put", async () => {
    mockPathname = "/onboarding";
    mockIsCompleted = false;

    render(
      <OnboardingProvider>
        <div data-testid="child" />
      </OnboardingProvider>,
    );

    // Give async work time to resolve, then assert no redirect.
    await new Promise((r) => setTimeout(r, 30));
    expect(routerReplace).not.toHaveBeenCalled();
  });
});
