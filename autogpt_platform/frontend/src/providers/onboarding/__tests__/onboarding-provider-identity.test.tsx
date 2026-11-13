import { act, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import type { WebSocketNotification } from "@/lib/autogpt-server-api";
import OnboardingProvider, { useOnboarding } from "../onboarding-provider";

// Stable across renders, like the real module-level `toast` and the memoised
// BackendAPI instance.
const { onboardingState, notificationHandlers, toast, api } = vi.hoisted(() => {
  const handlers: Array<(n: WebSocketNotification) => void> = [];
  return {
    onboardingState: {
      completedSteps: ["WELCOME"] as string[],
      walletShown: false,
    },
    notificationHandlers: handlers,
    toast: () => {},
    api: {
      onWebSocketMessage: (
        _type: string,
        handler: (n: WebSocketNotification) => void,
      ) => {
        handlers.push(handler);
        return () => {};
      },
      connectWebSocket: () => {},
    },
  };
});

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => "/marketplace",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false }),
}));

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => api,
}));

vi.mock("@/hooks/useOnboardingTimezoneDetection", () => ({
  useOnboardingTimezoneDetection: () => undefined,
}));

const getOnboardingState = vi.fn(() =>
  // A fresh object every call, as each response is.
  Promise.resolve({ status: 200, data: structuredClone(onboardingState) }),
);
vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: { is_completed: true } }),
  getV1OnboardingState: () => getOnboardingState(),
  patchV1UpdateOnboardingState: () => Promise.resolve({ status: 200 }),
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
}));

const contextValues: Array<ReturnType<typeof useOnboarding>> = [];
function Consumer() {
  contextValues.push(useOnboarding());
  return null;
}
const consumer = <Consumer />;

async function mountLoaded() {
  const view = render(<OnboardingProvider>{consumer}</OnboardingProvider>);
  await waitFor(() =>
    expect(contextValues[contextValues.length - 1]?.state).not.toBeNull(),
  );
  return view;
}

async function pushOnboardingNotification() {
  const calls = getOnboardingState.mock.calls.length;
  await act(async () => {
    notificationHandlers[notificationHandlers.length - 1]({
      type: "onboarding",
    } as WebSocketNotification);
  });
  await waitFor(() =>
    expect(getOnboardingState.mock.calls.length).toBe(calls + 1),
  );
  await act(async () => undefined);
}

describe("OnboardingProvider context identity", () => {
  beforeEach(() => {
    contextValues.length = 0;
    notificationHandlers.length = 0;
    getOnboardingState.mockClear();
    onboardingState.completedSteps = ["WELCOME"];
  });

  test("keeps the context value across a provider re-render with unchanged state", async () => {
    const view = await mountLoaded();
    const value = contextValues[contextValues.length - 1];
    const renders = contextValues.length;

    view.rerender(<OnboardingProvider>{consumer}</OnboardingProvider>);

    expect(contextValues.length).toBe(renders);
    expect(contextValues[contextValues.length - 1]).toBe(value);
  });

  test("an onboarding notification with identical data does not re-render consumers", async () => {
    await mountLoaded();
    const renders = contextValues.length;

    await pushOnboardingNotification();

    expect(contextValues.length).toBe(renders);
  });

  test("an onboarding notification with new data still reaches consumers", async () => {
    await mountLoaded();
    onboardingState.completedSteps = ["WELCOME", "MARKETPLACE_VISIT"];

    await pushOnboardingNotification();

    expect(
      contextValues[contextValues.length - 1].state?.completedSteps,
    ).toEqual(["WELCOME", "MARKETPLACE_VISIT"]);
  });
});
