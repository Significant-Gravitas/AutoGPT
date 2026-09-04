import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import useCredits from "../useCredits";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";

let authenticatedUserId: string | null = "user-a";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: authenticatedUserId ? { id: authenticatedUserId } : null,
    isLoggedIn: Boolean(authenticatedUserId),
  }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}));

describe("useCredits hook authentication and identity scoping", () => {
  beforeEach(() => {
    authenticatedUserId = "user-a";
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("does not emit billing requests and exposes empty state when logged out", async () => {
    authenticatedUserId = null;
    const getUserCreditSpy = vi
      .spyOn(AutoGPTServerAPI.prototype, "getUserCredit")
      .mockResolvedValue({ credits: 500 });
    const getAutoTopUpConfigSpy = vi
      .spyOn(AutoGPTServerAPI.prototype, "getAutoTopUpConfig")
      .mockResolvedValue({ amount: 10, threshold: 5 });
    const getTransactionHistorySpy = vi
      .spyOn(AutoGPTServerAPI.prototype, "getTransactionHistory")
      .mockResolvedValue({ transactions: [], next_transaction_time: null });
    const getRefundRequestsSpy = vi
      .spyOn(AutoGPTServerAPI.prototype, "getRefundRequests")
      .mockResolvedValue([]);

    const { result } = renderHook(() =>
      useCredits({
        fetchInitialCredits: true,
        fetchInitialAutoTopUpConfig: true,
        fetchInitialTransactionHistory: true,
        fetchInitialRefundRequests: true,
      }),
    );

    // Call manual fetch triggers as well
    act(() => {
      result.current.fetchCredits();
      result.current.fetchAutoTopUpConfig();
      result.current.fetchTransactionHistory();
      result.current.fetchRefundRequests();
    });

    expect(getUserCreditSpy).not.toHaveBeenCalled();
    expect(getAutoTopUpConfigSpy).not.toHaveBeenCalled();
    expect(getTransactionHistorySpy).not.toHaveBeenCalled();
    expect(getRefundRequestsSpy).not.toHaveBeenCalled();

    expect(result.current.credits).toBeNull();
    expect(result.current.autoTopUpConfig).toBeNull();
    expect(result.current.transactionHistory).toEqual({
      transactions: [],
      next_transaction_time: null,
    });
    expect(result.current.refundRequests).toEqual([]);
  });

  test("populates all state values for the authenticated identity", async () => {
    authenticatedUserId = "user-a";
    vi.spyOn(AutoGPTServerAPI.prototype, "getUserCredit").mockResolvedValue({
      credits: 1250,
    });
    vi.spyOn(
      AutoGPTServerAPI.prototype,
      "getAutoTopUpConfig",
    ).mockResolvedValue({ amount: 20, threshold: 5 });
    vi.spyOn(
      AutoGPTServerAPI.prototype,
      "getTransactionHistory",
    ).mockResolvedValue({
      transactions: [
        {
          transaction_key: "tx-1",
          amount: 500,
          type: "TOP_UP",
          created_at: new Date("2026-08-29T10:00:00Z"),
        } as any,
      ],
      next_transaction_time: new Date("2026-08-29T10:00:00Z"),
    });
    vi.spyOn(AutoGPTServerAPI.prototype, "getRefundRequests").mockResolvedValue(
      [{ id: "ref-1", status: "PENDING" } as any],
    );

    const { result } = renderHook(() =>
      useCredits({
        fetchInitialCredits: true,
        fetchInitialAutoTopUpConfig: true,
        fetchInitialTransactionHistory: true,
        fetchInitialRefundRequests: true,
      }),
    );

    await waitFor(() => {
      expect(result.current.credits).toBe(1250);
      expect(result.current.autoTopUpConfig).toEqual({
        amount: 20,
        threshold: 5,
      });
      expect(result.current.transactionHistory.transactions).toHaveLength(1);
      expect(result.current.refundRequests).toHaveLength(1);
    });
  });

  test("immediately clears and refetches with fresh cursor on account switch without remounting", async () => {
    let currentId = "user-a";
    const historyCalls: Array<{ cursor: Date | null | undefined }> = [];

    vi.spyOn(AutoGPTServerAPI.prototype, "getUserCredit").mockImplementation(
      async () => ({
        credits: currentId === "user-a" ? 100 : 200,
      }),
    );
    vi.spyOn(
      AutoGPTServerAPI.prototype,
      "getAutoTopUpConfig",
    ).mockImplementation(async () => ({
      amount: currentId === "user-a" ? 10 : 30,
      threshold: 5,
    }));
    vi.spyOn(
      AutoGPTServerAPI.prototype,
      "getTransactionHistory",
    ).mockImplementation(async (cursor) => {
      historyCalls.push({ cursor });
      return {
        transactions: [
          {
            transaction_key: `tx-${currentId}`,
            amount: 100,
            type: "TOP_UP",
            created_at: new Date("2026-08-29"),
          } as any,
        ],
        next_transaction_time: new Date("2026-08-29T12:00:00Z"),
      };
    });
    vi.spyOn(
      AutoGPTServerAPI.prototype,
      "getRefundRequests",
    ).mockImplementation(async () => [{ id: `refund-${currentId}` } as any]);

    const { result, rerender } = renderHook(
      ({ identityKey }) =>
        useCredits({
          identityKey,
          fetchInitialCredits: true,
          fetchInitialAutoTopUpConfig: true,
          fetchInitialTransactionHistory: true,
          fetchInitialRefundRequests: true,
        }),
      {
        initialProps: { identityKey: "user-a" as string | null },
      },
    );

    await waitFor(() => {
      expect(result.current.credits).toBe(100);
      expect(result.current.autoTopUpConfig?.amount).toBe(10);
      expect(
        result.current.transactionHistory.transactions[0].transaction_key,
      ).toBe("tx-user-a");
      expect(result.current.refundRequests[0].id).toBe("refund-user-a");
    });

    expect(historyCalls).toEqual([{ cursor: null }]);

    // Switch account to user-b
    currentId = "user-b";
    rerender({ identityKey: "user-b" });

    // Stale user-a data should be immediately hidden
    expect(result.current.credits).toBeNull();
    expect(result.current.autoTopUpConfig).toBeNull();
    expect(result.current.transactionHistory.transactions).toHaveLength(0);
    expect(result.current.refundRequests).toHaveLength(0);

    // Wait for user-b data to resolve from a fresh cursor
    await waitFor(() => {
      expect(result.current.credits).toBe(200);
      expect(result.current.autoTopUpConfig?.amount).toBe(30);
      expect(
        result.current.transactionHistory.transactions[0].transaction_key,
      ).toBe("tx-user-b");
      expect(result.current.refundRequests[0].id).toBe("refund-user-b");
    });

    // Verify user-b started from fresh cursor (cursor: null)
    expect(historyCalls).toHaveLength(2);
    expect(historyCalls[1]).toEqual({ cursor: null });

    // Now log out (identityKey: null)
    rerender({ identityKey: null });

    expect(result.current.credits).toBeNull();
    expect(result.current.autoTopUpConfig).toBeNull();
    expect(result.current.transactionHistory).toEqual({
      transactions: [],
      next_transaction_time: null,
    });
    expect(result.current.refundRequests).toEqual([]);
  });

  test("ignores inflight responses started for a previous identity", async () => {
    let resolveCreditsUserA: (value: { credits: number }) => void;
    const creditsPromiseUserA = new Promise<{ credits: number }>((resolve) => {
      resolveCreditsUserA = resolve;
    });

    vi.spyOn(AutoGPTServerAPI.prototype, "getUserCredit").mockImplementation(
      async () => {
        if (authenticatedUserId === "user-a") {
          return creditsPromiseUserA;
        }
        return { credits: 200 };
      },
    );

    const { result, rerender } = renderHook(
      ({ identityKey }) =>
        useCredits({
          identityKey,
          fetchInitialCredits: true,
        }),
      {
        initialProps: { identityKey: "user-a" as string | null },
      },
    );

    // User switches identity to user-b while user-a's request is inflight
    authenticatedUserId = "user-b";
    rerender({ identityKey: "user-b" });

    // Now user-a's delayed request completes
    await act(async () => {
      resolveCreditsUserA({ credits: 9999 });
    });

    // Credits should resolve to user-b's data, NOT user-a's 9999
    await waitFor(() => {
      expect(result.current.credits).toBe(200);
    });
  });
});
