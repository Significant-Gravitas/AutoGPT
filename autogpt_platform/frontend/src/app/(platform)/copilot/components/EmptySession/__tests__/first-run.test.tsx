import { useFirstRunTourStore } from "@/app/(platform)/components/FirstRunTour/store";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

const onboarding = vi.hoisted(() => ({
  current: {
    state: { completedSteps: [] as string[] },
    completeStep: vi.fn(),
  },
}));
vi.mock("@/providers/onboarding/onboarding-provider", async (original) => ({
  ...(await original<object>()),
  useOnboarding: () => onboarding.current,
}));

import { EmptySession } from "../EmptySession";

const WELCOME_PENDING_KEY = "autogpt:onboarding-welcome-pending";

function renderHome() {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "", hasMemory: true });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={vi.fn()}
      />
    </Wrapper>,
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
  window.localStorage.clear();
  onboarding.current = { state: { completedSteps: [] }, completeStep: vi.fn() };
});

afterEach(() => {
  act(() => useFirstRunTourStore.getState().close());
});

describe("home — the first visit after onboarding", () => {
  it("starts the guided walkthrough instead of the old concept cards", async () => {
    setIntroPath("A");
    renderHome();

    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );
    expect(screen.queryByText("Meet your Head of AI.")).toBeNull();
    expect(screen.queryByTestId("onboarding-welcome-overlay")).toBeNull();
  });

  it("clears the first-run flag once the walkthrough ends, so a refresh does not bring it back", async () => {
    setIntroPath("B");
    renderHome();
    await waitFor(() =>
      expect(useFirstRunTourStore.getState().source).toBe("auto"),
    );
    expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBe("1");

    act(() => useFirstRunTourStore.getState().close());

    await waitFor(() =>
      expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull(),
    );
  });

  it("does not start it for an existing user who already had the old cards", async () => {
    onboarding.current.state = { completedSteps: ["CAPABILITY_CARDS"] };
    setIntroPath("A");
    renderHome();

    await waitFor(() =>
      expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull(),
    );
    expect(useFirstRunTourStore.getState().source).toBeNull();
  });
});
