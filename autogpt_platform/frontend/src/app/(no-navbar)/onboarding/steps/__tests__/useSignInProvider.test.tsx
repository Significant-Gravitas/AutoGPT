import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
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

let releaseLookups = () => {};
let heldLookups: Promise<unknown>[] = [];

afterEach(async () => {
  vi.useRealTimers();
  releaseLookups();
  await Promise.all(heldLookups);
  heldLookups = [];
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

  it("lets onboarding go on 5 seconds into a lookup that hangs", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    let lookups = 0;
    // Held until teardown, past the lookup's timeout.
    const released = new Promise<void>((resolve) => {
      releaseLookups = resolve;
    });
    server.use(
      http.get("*/api/auth/sign-in-providers", () => {
        lookups += 1;
        const reply = released.then(() =>
          HttpResponse.json({ providers: ["microsoft"] }),
        );
        heldLookups.push(reply);
        return reply;
      }),
    );
    const { result } = renderLookup();
    expect(result.current.isReady).toBe(false);

    // A second at a time, so the timeout's failure settles in between.
    for (let second = 0; second < 4; second += 1) {
      await act(() => vi.advanceTimersByTimeAsync(1000));
    }
    expect(result.current.isReady).toBe(false);
    for (let second = 0; second < 2; second += 1) {
      await act(() => vi.advanceTimersByTimeAsync(1000));
    }

    // A lookup that timed out would only hang again, so it isn't retried.
    expect(result.current.isReady).toBe(true);
    expect(result.current.provider).toBeNull();
    expect(lookups).toBe(1);
  });

  it("tries once more after a lookup that fails quickly", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    let lookups = 0;
    server.use(
      http.get("*/api/auth/sign-in-providers", () => {
        lookups += 1;
        return lookups === 1
          ? HttpResponse.json({}, { status: 503 })
          : HttpResponse.json({ providers: ["microsoft"] });
      }),
    );
    const { result } = renderLookup();

    for (let second = 0; second < 3; second += 1) {
      await act(() => vi.advanceTimersByTimeAsync(1000));
    }

    expect(result.current.isReady).toBe(true);
    expect(result.current.provider).toBe("microsoft");
    expect(lookups).toBe(2);
  });
});
