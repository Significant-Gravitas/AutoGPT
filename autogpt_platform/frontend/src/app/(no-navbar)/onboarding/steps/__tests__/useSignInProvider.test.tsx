import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { useSignInProvider } from "../AccessStep/useSignInProvider";

function renderLookup() {
  const client = new QueryClient();
  function wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return renderHook(() => useSignInProvider(true), { wrapper });
}

afterEach(() => {
  vi.useRealTimers();
});

describe("useSignInProvider", () => {
  it("finds the provider this account signed in with", async () => {
    server.use(
      http.get("*/api/auth/sign-in-providers", () =>
        HttpResponse.json({ providers: ["credential", "microsoft"] }),
      ),
    );
    const { result } = renderLookup();

    await waitFor(() => expect(result.current.isReady).toBe(true));
    expect(result.current.provider).toBe("microsoft");
  });

  it("lets onboarding go on when the lookup hangs", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    let lookups = 0;
    server.use(
      http.get("*/api/auth/sign-in-providers", async () => {
        lookups += 1;
        await delay("infinite");
        return HttpResponse.json({ providers: ["microsoft"] });
      }),
    );
    const { result } = renderLookup();
    expect(result.current.isReady).toBe(false);

    // Each attempt gives up after 5 s, and the one retry waits 1 s first.
    // A second at a time, so each failure settles before the next timer.
    for (let second = 0; second < 12; second += 1) {
      await act(() => vi.advanceTimersByTimeAsync(1000));
    }

    expect(result.current.isReady).toBe(true);
    expect(result.current.provider).toBeNull();
    expect(lookups).toBe(2);
  });
});
