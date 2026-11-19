import { getGetV2GetUserRateLimitQueryKey } from "@/app/api/__generated__/endpoints/admin/admin";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { useAutoPilotUsage } from "../components/AutoPilotUsageSection/useAutoPilotUsage";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

const USER_A = "11111111-1111-4111-8111-111111111111";
const USER_B = "22222222-2222-4222-8222-222222222222";

// The page keeps one usage section mounted across lookups, so a result that
// lands after the admin has moved on must still go to the user it was for.
describe("useAutoPilotUsage after switching users mid-request", () => {
  it("stores a usage reset under the user it was made for", async () => {
    const release = holdResponses(
      "*/api/copilot/admin/rate_limit/reset",
      (userID) => usage(userID, 0),
    );
    const { result, rerender, queryClient } = renderUsageFor(USER_B);
    await waitFor(() => expect(result.current.rateLimit?.user_id).toBe(USER_B));
    rerender({ userId: USER_A });
    await waitFor(() => expect(result.current.rateLimit?.user_id).toBe(USER_A));

    act(() => result.current.resetUsage("daily", () => {}));
    rerender({ userId: USER_B });
    release();

    await waitFor(() =>
      expect(
        cachedUsage(queryClient, USER_A)?.daily_cost_used_microdollars,
      ).toBe(0),
    );
    expect(cachedUsage(queryClient, USER_B)?.user_id).toBe(USER_B);
    expect(result.current.rateLimit?.user_id).toBe(USER_B);
  });

  it("refreshes the plan of the user it was changed for", async () => {
    const release = holdResponses(
      "*/api/copilot/admin/rate_limit/tier",
      (userID) => ({
        user_id: userID,
        tier: "MAX",
      }),
    );
    const { result, rerender, queryClient } = renderUsageFor(USER_B);
    await waitFor(() => expect(result.current.rateLimit?.user_id).toBe(USER_B));
    rerender({ userId: USER_A });
    await waitFor(() => expect(result.current.rateLimit?.user_id).toBe(USER_A));

    act(() => result.current.changePlan("MAX", () => {}));
    rerender({ userId: USER_B });
    release();

    await waitFor(() =>
      expect(queryClient.getQueryState(usageKey(USER_A))?.isInvalidated).toBe(
        true,
      ),
    );
  });
});

function renderUsageFor(userId: string) {
  server.use(
    http.get("*/api/copilot/admin/rate_limit", ({ request }) =>
      HttpResponse.json(
        usage(
          new URL(request.url).searchParams.get("user_id") ?? "",
          1_200_000,
        ),
      ),
    ),
  );
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }
  const rendered = renderHook(
    (props: { userId: string }) => useAutoPilotUsage(props.userId),
    { wrapper: Wrapper, initialProps: { userId } },
  );
  return { ...rendered, queryClient };
}

// Answers each POST to `path` only once `release()` is called.
function holdResponses(path: string, answer: (userID: string) => object) {
  let release = () => {};
  const held = new Promise<void>((resolve) => (release = resolve));
  server.use(
    http.post(path, async ({ request }) => {
      const body = (await request.json()) as { user_id: string };
      await held;
      return HttpResponse.json(answer(body.user_id));
    }),
  );
  return () => release();
}

function usageKey(userID: string) {
  return getGetV2GetUserRateLimitQueryKey({ user_id: userID });
}

function cachedUsage(queryClient: QueryClient, userID: string) {
  return queryClient.getQueryData<{ data: UserRateLimitResponse }>(
    usageKey(userID),
  )?.data;
}

function usage(userID: string, dailyUsed: number): UserRateLimitResponse {
  return {
    user_id: userID,
    user_email: `${userID}@example.com`,
    daily_cost_limit_microdollars: 5_000_000,
    weekly_cost_limit_microdollars: 25_000_000,
    daily_cost_used_microdollars: dailyUsed,
    weekly_cost_used_microdollars: 4_800_000,
    tier: "PRO",
    tier_multipliers: { PRO: 3 },
  };
}
