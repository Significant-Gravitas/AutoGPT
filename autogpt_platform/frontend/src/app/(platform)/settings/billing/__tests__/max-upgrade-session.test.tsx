import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  deferredTrialResponse,
  setTrialUser,
} from "@/tests/integrations/trial-fixtures";
import { useTierUpgradePreview } from "../components/SubscriptionTab/YourPlanCard/useTierUpgradePreview";
import { maxUpgradePreview } from "./max-upgrade.fixture";

beforeEach(() => setTrialUser("user-a"));
afterEach(() => setTrialUser(null));

function renderReview() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return renderHook(() => useTierUpgradePreview(true, "monthly"), {
    wrapper: Wrapper,
  });
}

it("discards a late quote from the previous account and clears the amount on logout", async () => {
  const pending = deferredTrialResponse<ReturnType<typeof maxUpgradePreview>>();
  const requested = vi.fn();
  server.use(
    http.post("*/api/credits/subscription/preview", async () => {
      requested();
      return HttpResponse.json(
        requested.mock.calls.length === 1
          ? await pending.promise
          : maxUpgradePreview({
              amount_due: 28000,
              preview_id: "b".repeat(32),
            }),
      );
    }),
  );
  const { result } = renderReview();
  await waitFor(() => expect(requested).toHaveBeenCalledOnce());
  act(() => setTrialUser("user-b"));
  expect(result.current.preview).toBeUndefined();
  await waitFor(() => expect(result.current.preview?.amount_due).toBe(28000));
  await act(async () => pending.resolve(maxUpgradePreview()));
  expect(result.current.preview?.preview_id).toBe("b".repeat(32));
  act(() => setTrialUser(null));
  expect(result.current.preview).toBeUndefined();
});

it("withdraws an expired charge and requires another review", async () => {
  let calls = 0;
  server.use(
    http.post("*/api/credits/subscription/preview", () => {
      const expires =
        ++calls === 1
          ? Math.ceil(Date.now() / 1000) + 1
          : Math.floor(Date.now() / 1000) + 900;
      return HttpResponse.json(maxUpgradePreview({ expires_at: expires }));
    }),
  );
  const { result } = renderReview();
  await waitFor(() => expect(result.current.preview).toBeDefined());
  await waitFor(() => expect(result.current.error).toMatch(/expired/), {
    timeout: 2500,
  });
  expect(result.current.preview).toBeUndefined();
  act(() => result.current.retry());
  await waitFor(() => expect(result.current.preview).toBeDefined());
  expect(result.current.error).toBe("");
});

it("never presents an already expired response as payable", async () => {
  server.use(
    http.post("*/api/credits/subscription/preview", () =>
      HttpResponse.json(
        maxUpgradePreview({ expires_at: Math.floor(Date.now() / 1000) - 1 }),
      ),
    ),
  );
  const { result } = renderReview();
  await waitFor(() => expect(result.current.error).toMatch(/expired/));
  expect(result.current.preview).toBeUndefined();
});
