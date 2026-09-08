import CreditsPage from "@/app/(platform)/profile/(user)/credits/page";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const useCreditsMock = vi.hoisted(() => vi.fn());
const routerPush = vi.hoisted(() => vi.fn());
const fulfillCheckout = vi.hoisted(() => vi.fn(() => Promise.resolve()));
const navigationState = vi.hoisted(() => ({ topup: null as string | null }));

vi.mock("@/hooks/useCredits", () => ({ default: useCreditsMock }));
vi.mock("@/lib/autogpt-server-api/context", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/autogpt-server-api/context")>();
  return { ...actual, useBackendAPI: () => ({ fulfillCheckout }) };
});
vi.mock("next/navigation", () => ({
  usePathname: () => "/settings/credits",
  useRouter: () => ({ push: routerPush, replace: vi.fn() }),
  useSearchParams: () =>
    new URLSearchParams(
      navigationState.topup ? { topup: navigationState.topup } : undefined,
    ),
}));
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetCopilotUsage: () => ({ data: null, isSuccess: false }),
}));
vi.mock(
  "@/app/(platform)/profile/(user)/credits/components/SubscriptionTierSection/SubscriptionTierSection",
  () => ({ SubscriptionTierSection: () => null }),
);
vi.mock("@/app/(platform)/profile/(user)/credits/RefundModal", () => ({
  RefundModal: () => null,
}));

beforeEach(() => {
  navigationState.topup = null;
  useAuthStore.setState({ user: { id: "user-a" } as User });
  useCreditsMock.mockReturnValue({
    requestTopUp: vi.fn(),
    autoTopUpConfig: null,
    updateAutoTopUpConfig: vi.fn(),
    transactionHistory: { transactions: [], next_transaction_time: null },
    fetchTransactionHistory: vi.fn(),
    formatCredits: vi.fn(() => "$1.00"),
    refundTopUp: vi.fn(),
    refundRequests: [],
  });
});

afterEach(() => {
  useAuthStore.setState({ user: null });
  vi.clearAllMocks();
});

test("remounts only the billing page state for a new identity", async () => {
  render(<CreditsPage />);

  expect(useCreditsMock).toHaveBeenLastCalledWith({
    identityKey: "user-a",
    fetchInitialAutoTopUpConfig: true,
    fetchInitialRefundRequests: true,
    fetchInitialTransactionHistory: true,
  });

  const topUpAmount = screen.getByLabelText("Top-up amount (USD), minimum $5:");
  await userEvent.clear(topUpAmount);
  await userEvent.type(topUpAmount, "99");
  expect((topUpAmount as HTMLInputElement).value).toBe("99");

  act(() => {
    useAuthStore.setState({ user: { id: "user-b" } as User });
  });

  await waitFor(() => {
    expect(
      (
        screen.getByLabelText(
          "Top-up amount (USD), minimum $5:",
        ) as HTMLInputElement
      ).value,
    ).toBe("5");
    expect(useCreditsMock).toHaveBeenLastCalledWith({
      identityKey: "user-b",
      fetchInitialAutoTopUpConfig: true,
      fetchInitialRefundRequests: true,
      fetchInitialTransactionHistory: true,
    });
  });
});

test("disables initial billing reads while logged out", () => {
  useAuthStore.setState({ user: null });

  render(<CreditsPage />);

  expect(useCreditsMock).toHaveBeenLastCalledWith({
    identityKey: null,
    fetchInitialAutoTopUpConfig: false,
    fetchInitialRefundRequests: false,
    fetchInitialTransactionHistory: false,
  });
});

test("fulfills checkout once for the first authenticated identity", async () => {
  navigationState.topup = "success";
  useAuthStore.setState({ user: null });

  render(<CreditsPage />);
  expect(fulfillCheckout).not.toHaveBeenCalled();

  act(() => {
    useAuthStore.setState({ user: { id: "user-a" } as User });
  });
  await waitFor(() => expect(fulfillCheckout).toHaveBeenCalledTimes(1));

  act(() => {
    useAuthStore.setState({ user: { id: "user-b" } as User });
  });
  await act(async () => {
    await Promise.resolve();
  });

  expect(fulfillCheckout).toHaveBeenCalledTimes(1);
});
