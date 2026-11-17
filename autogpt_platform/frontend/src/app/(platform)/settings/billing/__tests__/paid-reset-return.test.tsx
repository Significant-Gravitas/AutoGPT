import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  rememberReturnPath,
  takeReturnPath,
} from "@/app/(platform)/copilot/components/UsageLimits/PaidReset/helpers";
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

const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

afterEach(() => {
  mockSearchParams.current = new URLSearchParams();
  mockRouterReplace.mockReset();
  toast.mockReset();
  takeReturnPath();
});

function Returning() {
  // The usage read-out is on the page, so refreshing it reads it again.
  useGetV2GetCopilotUsage();
  // No waiting between attempts in tests.
  usePaidResetReturn(1);
  return null;
}

function onConfirm(answers: Array<"granted" | "pending" | "error">) {
  const seen = { sessions: [] as string[], usageReads: 0 };
  server.use(
    http.post("*/api/chat/usage/paid-reset/confirm", async ({ request }) => {
      const body = (await request.json()) as { session_id: string };
      seen.sessions.push(body.session_id);
      const answer =
        answers[Math.min(seen.sessions.length, answers.length) - 1];
      if (answer === "error")
        return HttpResponse.json({ detail: "down" }, { status: 503 });
      return HttpResponse.json({
        status: answer,
        resets_at: answer === "granted" ? "2030-11-12T08:30:56Z" : null,
        usage: null,
      });
    }),
    http.get("*/api/chat/usage", () => {
      seen.usageReads += 1;
      return HttpResponse.json({ tier: "MAX", daily: null, weekly: null });
    }),
  );
  return seen;
}

function titles() {
  return toast.mock.calls.map(([shown]) => shown.title);
}

describe("returning from paying for a usage reset by card", () => {
  it("confirms the payment, says the limits are reset and reads usage again", async () => {
    const seen = onConfirm(["granted"]);
    mockSearchParams.current = new URLSearchParams({
      usage_reset: "success",
      session_id: "cs_test_paid",
    });

    render(<Returning />);

    await waitFor(() => expect(titles()).toContain("Your limits are reset"));
    expect(seen.sessions[0]).toBe("cs_test_paid");
    // Granted on the first answer: nothing more is asked.
    expect(new Set(seen.sessions).size).toBe(1);
    // Once on mount and again for the reset.
    await waitFor(() => expect(seen.usageReads).toBeGreaterThanOrEqual(2));
    // The query string is dropped, so a reload doesn't ask again.
    expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing");
  });

  it("keeps asking for a moment while Stripe hasn’t confirmed yet", async () => {
    const seen = onConfirm(["pending", "error", "granted"]);
    mockSearchParams.current = new URLSearchParams({
      usage_reset: "success",
      session_id: "cs_test_slow",
    });

    render(<Returning />);

    await waitFor(() => expect(titles()).toContain("Your limits are reset"));
    expect(seen.sessions).toHaveLength(3);
  });

  it("stops asking after a few tries, says so, and keeps reading usage for a late confirmation", async () => {
    const seen = onConfirm(["pending"]);
    mockSearchParams.current = new URLSearchParams({
      usage_reset: "success",
      session_id: "cs_test_unpaid",
    });

    render(<Returning />);

    await waitFor(() => expect(titles()).toContain("Confirming your payment"));
    expect(titles()).not.toContain("Your limits are reset");
    // Usage is read on mount, when the asking ends, and three times after.
    await waitFor(() => expect(seen.usageReads).toBeGreaterThanOrEqual(5));
    // The asking itself ended at four tries.
    expect(seen.sessions).toHaveLength(4);
  });

  it("confirms nothing after a cancelled payment", async () => {
    const seen = onConfirm(["granted"]);
    mockSearchParams.current = new URLSearchParams({ usage_reset: "cancel" });

    render(<Returning />);

    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
    expect(titles()).toEqual(["Payment cancelled"]);
    expect(seen.sessions).toEqual([]);
  });

  it("confirms nothing without a session, and does nothing on a normal visit", async () => {
    const seen = onConfirm(["granted"]);
    mockSearchParams.current = new URLSearchParams({ usage_reset: "success" });

    render(<Returning />);

    await waitFor(() => expect(mockRouterReplace).toHaveBeenCalled());
    expect(seen.sessions).toEqual([]);
    mockRouterReplace.mockReset();
    mockSearchParams.current = new URLSearchParams();
    render(<Returning />);
    expect(mockRouterReplace).not.toHaveBeenCalled();
  });

  it("goes back to the chat the customer paid from", async () => {
    onConfirm(["granted"]);
    rememberReturnPath("/copilot?sessionId=abc");
    mockSearchParams.current = new URLSearchParams({
      usage_reset: "success",
      session_id: "cs_test_paid",
    });

    render(<Returning />);

    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/copilot?sessionId=abc"),
    );
  });

  it.each(["https://elsewhere.test/x", "//elsewhere.test/x", "copilot"])(
    "never leaves the site for a remembered %s",
    async (path) => {
      onConfirm(["granted"]);
      rememberReturnPath(path);
      mockSearchParams.current = new URLSearchParams({ usage_reset: "cancel" });

      render(<Returning />);

      await waitFor(() =>
        expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
      );
    },
  );
});
