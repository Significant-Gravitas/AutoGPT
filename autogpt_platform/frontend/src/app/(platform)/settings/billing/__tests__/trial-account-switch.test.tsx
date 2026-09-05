import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
  getPostTrialsStartTrialCheckoutMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { TrialCheckoutResponse } from "@/app/api/__generated__/models/trialCheckoutResponse";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

beforeEach(() => setTrialUser("user-a"));
afterEach(() => {
  setTrialUser(null);
  vi.restoreAllMocks();
});

function offerForCurrentUser() {
  return trialResponse({
    eligible: true,
    active: false,
    status: null,
    offer: {
      ...trialOffer,
      duration_days: useAuthStore.getState().user?.id === "user-b" ? 10 : 7,
    },
  });
}

describe("trial account isolation", () => {
  it("hides user A's status immediately when switching to user B", async () => {
    const pending = deferredTrialResponse<ReturnType<typeof trialResponse>>();
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(() =>
        useAuthStore.getState().user?.id === "user-a"
          ? trialResponse()
          : pending.promise,
      ),
    );
    render(<TrialCard />);
    await screen.findByRole("button", { name: "Cancel trial" });
    act(() => setTrialUser("user-b"));
    expect(screen.queryByText("Your trial")).toBeNull();
    expect(screen.queryByRole("button", { name: "Cancel trial" })).toBeNull();
    pending.resolve(offerForCurrentUser());
    await screen.findByRole("button", { name: /start 10-day trial/i });
    act(() => setTrialUser(null));
    expect(screen.queryByRole("region", { name: "AutoGPT trial" })).toBeNull();
  });

  it("never redirects user B to a checkout requested by user A", async () => {
    const pending = deferredTrialResponse<TrialCheckoutResponse>();
    const checkout = vi.fn(() => pending.promise);
    const assign = vi
      .spyOn(window.location, "assign")
      .mockImplementation(() => {});
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(offerForCurrentUser),
      getPostTrialsStartTrialCheckoutMockHandler200(checkout),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: /start 7-day trial/i }),
    );
    await waitFor(() => expect(checkout).toHaveBeenCalledOnce());
    act(() => setTrialUser("user-b"));
    await screen.findByRole("button", { name: /start 10-day trial/i });
    pending.resolve({ url: "https://checkout.stripe.com/c/pay/user_a" });
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: /start 10-day trial/i })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(assign).not.toHaveBeenCalled();
  });

  it("does not apply user A's late cancellation result to user B", async () => {
    const pending = deferredTrialResponse<ReturnType<typeof trialResponse>>();
    const cancel = vi.fn(() => pending.promise);
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(() =>
        trialResponse({
          allowance_used_percent:
            useAuthStore.getState().user?.id === "user-a" ? 42.4 : 15,
        }),
      ),
      getPostTrialsCancelTrialMockHandler200(cancel),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Cancel trial" }),
    );
    await waitFor(() => expect(cancel).toHaveBeenCalledOnce());
    act(() => setTrialUser("user-b"));
    await screen.findByText(/allowance used: 15%/i);
    pending.resolve(trialResponse({ cancel_at_period_end: true }));
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: "Cancel trial" })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(screen.queryByText(/Cancellation confirmed/)).toBeNull();
    expect(screen.getByText(/allowance used: 15%/i)).toBeDefined();
  });
});
