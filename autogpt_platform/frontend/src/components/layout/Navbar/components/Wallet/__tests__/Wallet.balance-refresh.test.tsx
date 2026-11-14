import { act, render, screen } from "@/tests/integrations/test-utils";
import BackendAPI, {
  type GraphExecution,
  type UserOnboarding,
  type WebSocketNotification,
} from "@/lib/autogpt-server-api";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { server } from "@/mocks/mock-server";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Wallet } from "../Wallet";
import { BALANCE_REFRESH_DELAY_MS } from "../useRefreshBalanceOnSpend";

// SECRT-2758: the balance in the open wallet stayed as it was after a run or
// a chat turn spent from it, until the wallet was closed and opened again.

const socketHandlers = vi.hoisted(
  () => new Map<string, Set<(data: unknown) => void>>(),
);
const backendAPI = vi.hoisted(() => ({
  onWebSocketMessage(method: string, handler: (data: unknown) => void) {
    const handlers = socketHandlers.get(method) ?? new Set();
    handlers.add(handler);
    socketHandlers.set(method, handlers);
    return () => handlers.delete(handler);
  },
  connectWebSocket() {},
}));

vi.mock("@/lib/autogpt-server-api/context", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/autogpt-server-api/context")>();
  return { ...actual, useBackendAPI: () => backendAPI };
});

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => children,
  useOnboarding: () => ({
    state: {
      completedSteps: ["ONBOARDING_COMPLETE"],
      walletShown: true,
      notified: [],
      rewardedFor: [],
      usageReason: null,
      integrations: [],
      otherIntegrations: null,
      selectedStoreListingVersionId: null,
      agentInput: null,
      onboardingAgentExecutionId: null,
      lastRunAt: null,
      consecutiveRunDays: 0,
      agentRuns: 0,
    } satisfies UserOnboarding,
    updateState: vi.fn(),
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: vi.fn(() => true) };
});

vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({ isPaymentEnabled: true, isResolved: true }),
}));

let balance = 979;

beforeEach(() => {
  useAuthStore.setState({ user: { id: "user-a" } as User });
  socketHandlers.clear();
  balance = 979;
  server.use(
    http.get("*/api/internal/credits", () =>
      HttpResponse.json({ credits: balance }),
    ),
  );
});

afterEach(() => {
  useAuthStore.setState({ user: null });
  vi.restoreAllMocks();
  if (vi.isFakeTimers()) vi.useRealTimers();
});

function emit(method: string, data: unknown) {
  socketHandlers.get(method)?.forEach((handler) => handler(data));
}

function runUpdate(status: GraphExecution["status"]) {
  return { id: "exec-1", graph_id: "graph-1", status };
}

const chatTurnEnded: WebSocketNotification = {
  type: "copilot_completion",
  event: "session_completed",
  session_id: "session-1",
  status: "completed",
};

// Opens the wallet as the sidebar shows it, then puts the refresh delay on a
// clock the test advances and counts balance reads from that point on.
async function openWallet() {
  render(<Wallet compact />);
  await userEvent.click(
    await screen.findByRole("button", { name: "Usage and credits" }),
  );
  await screen.findByText("$9.79");
  const balanceReads = vi.spyOn(BackendAPI.prototype, "getUserCredit");
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  return balanceReads;
}

describe("Wallet balance after spending", () => {
  it("re-reads the balance once when a burst of run updates ends in a finished run", async () => {
    const balanceReads = await openWallet();

    balance = 929;
    emit("graph_execution_event", runUpdate("RUNNING"));
    emit("graph_execution_event", runUpdate("COMPLETED"));
    // The same run arrives once per subscribed channel.
    emit("graph_execution_event", runUpdate("COMPLETED"));
    emit("graph_execution_event", runUpdate("FAILED"));
    vi.advanceTimersByTime(BALANCE_REFRESH_DELAY_MS * 2);

    expect(balanceReads).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
    expect(await screen.findByText("$9.29")).toBeDefined();
  });

  it("re-reads the balance when a chat turn ends", async () => {
    const balanceReads = await openWallet();

    balance = 954;
    emit("notification", chatTurnEnded);
    vi.advanceTimersByTime(BALANCE_REFRESH_DELAY_MS);

    expect(balanceReads).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
    expect(await screen.findByText("$9.54")).toBeDefined();
  });

  it("keeps showing the last balance when the re-read fails", async () => {
    const balanceReads = await openWallet();

    server.use(
      http.get(
        "*/api/internal/credits",
        () => new HttpResponse(null, { status: 500 }),
      ),
    );
    emit("notification", chatTurnEnded);
    vi.advanceTimersByTime(BALANCE_REFRESH_DELAY_MS);

    expect(balanceReads).toHaveBeenCalledTimes(1);
    await act(async () => {
      expect(await balanceReads.mock.results[0].value).toEqual({
        credits: null,
      });
    });
    vi.useRealTimers();
    expect(screen.getByText("$9.79")).toBeDefined();
  });

  it("does not re-read the balance for a run still going or an unrelated notification", async () => {
    const balanceReads = await openWallet();

    emit("graph_execution_event", runUpdate("QUEUED"));
    emit("graph_execution_event", runUpdate("RUNNING"));
    emit("notification", {
      type: "onboarding",
      event: "increment_runs",
      step: null,
    });
    vi.advanceTimersByTime(BALANCE_REFRESH_DELAY_MS * 2);

    expect(balanceReads).not.toHaveBeenCalled();
  });
});
