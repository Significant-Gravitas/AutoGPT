import {
  getGetV1GetAutoTopUpMockHandler,
  getGetV1GetCreditHistoryMockHandler,
  getGetV1GetRefundRequestsMockHandler,
  getGetV1GetUserCreditsMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import useCredits from "@/hooks/useCredits";
import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { HttpResponse, http } from "msw";
import { describe, expect, test, vi } from "vitest";

const routerPush = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useParams: () => ({}),
  usePathname: () => "/marketplace",
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: routerPush,
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

type Props = {
  identityKey?: string | null;
};

function refundRequest(userId: string) {
  return {
    id: `refund-${userId}`,
    user_id: userId,
    transaction_key: `transaction-${userId}`,
    amount: 100,
    reason: "test",
    status: "PENDING",
    created_at: new Date("2026-08-29T00:00:00Z"),
    updated_at: new Date("2026-08-29T00:00:00Z"),
  };
}

describe("useCredits identity scope", () => {
  test("resets and refetches every dataset from a fresh transaction cursor", async () => {
    let responseUser = "user-a";
    const transactionCursors: Array<string | null> = [];

    server.use(
      getGetV1GetUserCreditsMockHandler(() => ({
        credits: responseUser === "user-a" ? 100 : 200,
      })),
      getGetV1GetAutoTopUpMockHandler(() => ({
        amount: responseUser === "user-a" ? 500 : 1000,
        threshold: responseUser === "user-a" ? 100 : 200,
      })),
      getGetV1GetCreditHistoryMockHandler(({ request }) => {
        const cursor = new URL(request.url).searchParams.get(
          "transaction_time",
        );
        transactionCursors.push(cursor);
        return {
          transactions: cursor ? [] : [{ user_id: responseUser }],
          next_transaction_time: cursor
            ? null
            : new Date("2026-08-28T00:00:00Z"),
        };
      }),
      getGetV1GetRefundRequestsMockHandler(() => [refundRequest(responseUser)]),
    );

    const view = renderHook(
      ({ identityKey }: Props) =>
        useCredits({
          identityKey,
          fetchInitialCredits: true,
          fetchInitialAutoTopUpConfig: true,
          fetchInitialTransactionHistory: true,
          fetchInitialRefundRequests: true,
        }),
      { initialProps: { identityKey: "user-a" } as Props },
    );

    await waitFor(() => {
      expect(view.result.current.credits).toBe(100);
      expect(view.result.current.autoTopUpConfig?.amount).toBe(500);
      expect(
        view.result.current.transactionHistory.transactions[0]?.user_id,
      ).toBe("user-a");
      expect(view.result.current.refundRequests[0]?.user_id).toBe("user-a");
    });

    act(() => {
      view.result.current.fetchTransactionHistory();
    });
    await waitFor(() => expect(transactionCursors).toHaveLength(2));
    expect(transactionCursors[1]).not.toBeNull();

    responseUser = "user-b";
    view.rerender({ identityKey: "user-b" });

    expect(view.result.current.credits).toBeNull();
    expect(view.result.current.autoTopUpConfig).toBeNull();
    expect(view.result.current.transactionHistory.transactions).toEqual([]);
    expect(view.result.current.refundRequests).toEqual([]);

    await waitFor(() => {
      expect(view.result.current.credits).toBe(200);
      expect(view.result.current.autoTopUpConfig?.amount).toBe(1000);
      expect(
        view.result.current.transactionHistory.transactions[0]?.user_id,
      ).toBe("user-b");
      expect(view.result.current.refundRequests[0]?.user_id).toBe("user-b");
    });
    expect(transactionCursors[2]).toBeNull();
  });

  test("does not request protected datasets for a logged-out identity", async () => {
    const requests = {
      credits: 0,
      autoTopUp: 0,
      transactions: 0,
      refunds: 0,
    };

    server.use(
      getGetV1GetUserCreditsMockHandler(() => {
        requests.credits += 1;
        return { credits: 0 };
      }),
      getGetV1GetAutoTopUpMockHandler(() => {
        requests.autoTopUp += 1;
        return { amount: 0, threshold: 0 };
      }),
      getGetV1GetCreditHistoryMockHandler(() => {
        requests.transactions += 1;
        return { transactions: [], next_transaction_time: null };
      }),
      getGetV1GetRefundRequestsMockHandler(() => {
        requests.refunds += 1;
        return [];
      }),
    );

    for (const identityKey of [null, undefined]) {
      const view = renderHook(() =>
        useCredits({
          identityKey,
          fetchInitialCredits: true,
          fetchInitialAutoTopUpConfig: true,
          fetchInitialTransactionHistory: true,
          fetchInitialRefundRequests: true,
        }),
      );
      view.unmount();
    }
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(requests).toEqual({
      credits: 0,
      autoTopUp: 0,
      transactions: 0,
      refunds: 0,
    });
  });

  test("does not run a retained fetch callback after logout", async () => {
    let creditRequests = 0;
    server.use(
      getGetV1GetUserCreditsMockHandler(() => {
        creditRequests += 1;
        return { credits: 100 };
      }),
    );
    const view = renderHook(
      ({ identityKey }: Props) => useCredits({ identityKey }),
      { initialProps: { identityKey: "user-a" } as Props },
    );
    const retainedFetch = view.result.current.fetchCredits;

    view.rerender({ identityKey: null });
    act(() => {
      retainedFetch();
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(creditRequests).toBe(0);
  });

  test("rejects a stale response after an A to logout to A transition", async () => {
    let creditRequests = 0;
    let resolveOldRequest: ((value: { credits: number }) => void) | undefined;

    server.use(
      getGetV1GetUserCreditsMockHandler(() => {
        creditRequests += 1;
        if (creditRequests === 1) {
          return new Promise<{ credits: number }>((resolve) => {
            resolveOldRequest = resolve;
          });
        }
        return { credits: 200 };
      }),
    );

    const view = renderHook(
      ({ identityKey }: Props) =>
        useCredits({ identityKey, fetchInitialCredits: true }),
      { initialProps: { identityKey: "user-a" } as Props },
    );

    await waitFor(() => expect(creditRequests).toBe(1));
    view.rerender({ identityKey: null });
    view.rerender({ identityKey: "user-a" });

    await waitFor(() => {
      expect(creditRequests).toBe(2);
      expect(view.result.current.credits).toBe(200);
    });

    await act(async () => {
      resolveOldRequest?.({ credits: 100 });
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(view.result.current.credits).toBe(200);
  });

  test("does not navigate to a stale checkout after identity changes", async () => {
    let releaseRequest: (() => void) | undefined;
    routerPush.mockClear();
    server.use(
      http.post("/api/proxy/api/credits", async () => {
        await new Promise<void>((resolve) => {
          releaseRequest = resolve;
        });
        return HttpResponse.json({
          checkout_url: "https://checkout.example/old-user",
        });
      }),
    );

    const view = renderHook(
      ({ identityKey }: Props) => useCredits({ identityKey }),
      { initialProps: { identityKey: "user-a" } as Props },
    );
    let request: Promise<void> | undefined;

    act(() => {
      request = view.result.current.requestTopUp(500);
    });
    await waitFor(() => expect(releaseRequest).toBeDefined());

    view.rerender({ identityKey: "user-b" });
    await act(async () => {
      releaseRequest?.();
      await request;
    });

    expect(routerPush).not.toHaveBeenCalled();
  });

  test("does not navigate to a checkout after the hook unmounts", async () => {
    let releaseRequest: (() => void) | undefined;
    routerPush.mockClear();
    server.use(
      http.post("/api/proxy/api/credits", async () => {
        await new Promise<void>((resolve) => {
          releaseRequest = resolve;
        });
        return HttpResponse.json({
          checkout_url: "https://checkout.example/unmounted",
        });
      }),
    );

    const view = renderHook(() => useCredits({ identityKey: "user-a" }));
    let request: Promise<void> | undefined;
    act(() => {
      request = view.result.current.requestTopUp(500);
    });
    await waitFor(() => expect(releaseRequest).toBeDefined());

    view.unmount();
    releaseRequest?.();
    await request;

    expect(routerPush).not.toHaveBeenCalled();
  });

  test("reports a stale auto-top-up update as ignored", async () => {
    let releaseRequest: (() => void) | undefined;
    let configReads = 0;
    server.use(
      http.post("/api/proxy/api/credits/auto-top-up", async () => {
        await new Promise<void>((resolve) => {
          releaseRequest = resolve;
        });
        return HttpResponse.json({ amount: 1000, threshold: 500 });
      }),
      http.get("/api/proxy/api/credits/auto-top-up", () => {
        configReads += 1;
        return HttpResponse.json({ amount: 1000, threshold: 500 });
      }),
    );

    const view = renderHook(
      ({ identityKey }: Props) => useCredits({ identityKey }),
      { initialProps: { identityKey: "user-a" } as Props },
    );
    let update: Promise<boolean> | undefined;
    act(() => {
      update = view.result.current.updateAutoTopUpConfig(1000, 500);
    });
    await waitFor(() => expect(releaseRequest).toBeDefined());

    view.rerender({ identityKey: "user-b" });
    releaseRequest?.();

    await expect(update).resolves.toBe(false);
    expect(configReads).toBe(0);
  });

  test("completes billing mutations for the current identity", async () => {
    routerPush.mockClear();
    server.use(
      http.post("/api/proxy/api/credits/auto-top-up", () =>
        HttpResponse.json({ amount: 1000, threshold: 500 }),
      ),
      http.get("/api/proxy/api/credits/auto-top-up", () =>
        HttpResponse.json({ amount: 1000, threshold: 500 }),
      ),
      http.post("/api/proxy/api/credits", () =>
        HttpResponse.json({ checkout_url: "https://checkout.example/current" }),
      ),
      http.get("/api/proxy/api/credits/manage", () =>
        HttpResponse.json({ url: "https://billing.example/current" }),
      ),
      http.post("/api/proxy/api/credits/:transactionKey/refund", () =>
        HttpResponse.json(250),
      ),
      http.get("/api/proxy/api/credits", () =>
        HttpResponse.json({ credits: 750 }),
      ),
      http.get("/api/proxy/api/credits/transactions", () =>
        HttpResponse.json({
          transactions: [],
          next_transaction_time: null,
        }),
      ),
    );

    const view = renderHook(() => useCredits({ identityKey: "user-a" }));
    let updated = false;
    await act(async () => {
      updated = await view.result.current.updateAutoTopUpConfig(1000, 500);
    });
    expect(updated).toBe(true);

    await act(async () => {
      await view.result.current.requestTopUp(500);
      await view.result.current.openBillingPortal();
    });
    expect(routerPush).toHaveBeenNthCalledWith(
      1,
      "https://checkout.example/current",
    );
    expect(routerPush).toHaveBeenNthCalledWith(
      2,
      "https://billing.example/current",
    );

    let refundedAmount: number | null = null;
    await act(async () => {
      refundedAmount = await view.result.current.refundTopUp(
        "transaction-a",
        "test",
      );
    });
    expect(refundedAmount).toBe(250);
    expect(view.result.current.credits).toBe(750);
    expect(view.result.current.transactionHistory.transactions).toEqual([]);
  });

  test("rejects billing mutations without an authenticated identity", async () => {
    const view = renderHook(() => useCredits({ identityKey: null }));

    await expect(
      view.result.current.updateAutoTopUpConfig(1000, 500),
    ).rejects.toThrow("Authentication required");
    await expect(view.result.current.requestTopUp(500)).rejects.toThrow(
      "Authentication required",
    );
    await expect(view.result.current.openBillingPortal()).rejects.toThrow(
      "Authentication required",
    );
    await expect(
      view.result.current.refundTopUp("transaction-a", "test"),
    ).rejects.toThrow("Authentication required");
  });

  test("propagates current-identity billing mutation failures", async () => {
    server.use(
      http.post("/api/proxy/api/credits/auto-top-up", () =>
        HttpResponse.json({ detail: "failed" }, { status: 500 }),
      ),
      http.post("/api/proxy/api/credits", () =>
        HttpResponse.json({ detail: "failed" }, { status: 500 }),
      ),
      http.get("/api/proxy/api/credits/manage", () =>
        HttpResponse.json({ detail: "failed" }, { status: 500 }),
      ),
      http.post("/api/proxy/api/credits/:transactionKey/refund", () =>
        HttpResponse.json({ detail: "failed" }, { status: 500 }),
      ),
    );

    const view = renderHook(() => useCredits({ identityKey: "user-a" }));

    await expect(
      view.result.current.updateAutoTopUpConfig(1000, 500),
    ).rejects.toThrow();
    await expect(view.result.current.requestTopUp(500)).rejects.toThrow();
    await expect(view.result.current.openBillingPortal()).rejects.toThrow();
    await expect(
      view.result.current.refundTopUp("transaction-a", "test"),
    ).rejects.toThrow();
  });

  test("does not return a previous identity's refund amount", async () => {
    let releaseRequest: (() => void) | undefined;
    let followUpReads = 0;
    server.use(
      http.post("/api/proxy/api/credits/:transactionKey/refund", async () => {
        await new Promise<void>((resolve) => {
          releaseRequest = resolve;
        });
        return HttpResponse.json(250);
      }),
      http.get("/api/proxy/api/credits", () => {
        followUpReads += 1;
        return HttpResponse.json({ credits: 1000 });
      }),
      http.get("/api/proxy/api/credits/transactions", () => {
        followUpReads += 1;
        return HttpResponse.json({
          transactions: [],
          next_transaction_time: null,
        });
      }),
    );

    const view = renderHook(
      ({ identityKey }: Props) => useCredits({ identityKey }),
      { initialProps: { identityKey: "user-a" } as Props },
    );
    let refund: Promise<number | null> | undefined;
    act(() => {
      refund = view.result.current.refundTopUp("transaction-a", "test");
    });
    await waitFor(() => expect(releaseRequest).toBeDefined());

    view.rerender({ identityKey: "user-b" });
    releaseRequest?.();

    await expect(refund).resolves.toBeNull();
    expect(followUpReads).toBe(0);
  });
});
