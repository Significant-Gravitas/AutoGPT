import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";
import { render, waitFor } from "@/tests/integrations/test-utils";
import { usePaidResetReturn } from "../components/SubscriptionTab/AutopilotUsageCard/usePaidResetReturn";

const mockSearchParams = { current: new URLSearchParams() };
const mockRouterReplace = vi.fn();
vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useSearchParams: () => mockSearchParams.current,
    useRouter: () => ({
      push: vi.fn(),
      replace: mockRouterReplace,
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => "/settings/billing",
    useParams: () => ({}),
  };
});

afterEach(() => {
  mockSearchParams.current = new URLSearchParams();
  mockRouterReplace.mockReset();
});

function Returning() {
  // No waiting between attempts in tests.
  usePaidResetReturn(1);
  return null;
}

function onConfirm(answers: Array<"granted" | "pending" | "error">) {
  const sessions: string[] = [];
  const usageReads = { count: 0 };
  server.use(
    http.post("*/api/chat/usage/paid-reset/confirm", async ({ request }) => {
      const body = (await request.json()) as { session_id: string };
      sessions.push(body.session_id);
      const answer = answers[Math.min(sessions.length, answers.length) - 1];
      if (answer === "error")
        return HttpResponse.json({ detail: "down" }, { status: 503 });
      return HttpResponse.json({
        status: answer,
        resets_at: answer === "granted" ? "2030-11-12T08:30:56Z" : null,
        usage: null,
      });
    }),
    http.get("*/api/chat/usage", () => {
      usageReads.count += 1;
      return HttpResponse.json({ tier: "MAX", daily: null, weekly: null });
    }),
  );
  return sessions;
}

describe("returning from paying for a usage reset by card", () => {
  it("confirms the payment with the session Stripe sent back", async () => {
    const sessions = onConfirm(["granted"]);
    mockSearchParams.current = new URLSearchParams({
      usage_reset: "success",
      session_id: "cs_test_paid",
    });

    render(<Returning />);

    await waitFor(() => expect(sessions[0]).toBe("cs_test_paid"));
    // The query string is dropped, so a reload doesn't ask again.
    expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing");
  });

  it("keeps asking for a moment while Stripe hasn’t confirmed yet", async () => {
    const sessions = onConfirm(["pending", "error", "granted"]);
    mockSearchParams.current = new URLSearchParams({
      usage_reset: "success",
      session_id: "cs_test_slow",
    });

    render(<Returning />);

    await waitFor(() => expect(sessions).toHaveLength(3));
  });

  it("gives up asking after a few tries and leaves it to Stripe’s own confirmation", async () => {
    const sessions = onConfirm(["pending"]);
    mockSearchParams.current = new URLSearchParams({
      usage_reset: "success",
      session_id: "cs_test_unpaid",
    });

    render(<Returning />);

    await waitFor(() => expect(sessions).toHaveLength(4));
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(sessions).toHaveLength(4);
  });

  it("confirms nothing after a cancelled payment", async () => {
    const sessions = onConfirm(["granted"]);
    mockSearchParams.current = new URLSearchParams({ usage_reset: "cancel" });

    render(<Returning />);

    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
    expect(sessions).toEqual([]);
  });

  it("confirms nothing without a session, and does nothing on a normal visit", async () => {
    const sessions = onConfirm(["granted"]);
    mockSearchParams.current = new URLSearchParams({ usage_reset: "success" });

    render(<Returning />);

    await waitFor(() => expect(mockRouterReplace).toHaveBeenCalled());
    expect(sessions).toEqual([]);
    mockRouterReplace.mockReset();
    mockSearchParams.current = new URLSearchParams();
    render(<Returning />);
    expect(mockRouterReplace).not.toHaveBeenCalled();
  });
});
