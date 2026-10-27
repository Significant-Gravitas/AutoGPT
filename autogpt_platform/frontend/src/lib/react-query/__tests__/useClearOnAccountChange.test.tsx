import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { useClearOnAccountChange } from "../useClearOnAccountChange";

afterEach(() => {
  act(() => useAuthStore.setState({ user: null }));
});

function signIn(id: string | null) {
  act(() => useAuthStore.setState({ user: id ? ({ id } as User) : null }));
}

describe("useClearOnAccountChange", () => {
  it("refetches what is on screen for the next account instead of leaving it pending", async () => {
    // clear() drops queries without telling mounted observers, which then
    // wait forever; the next account's screen must load its own data.
    const queryClient = new QueryClient();
    signIn("user-1");
    let fetches = 0;
    function useChats() {
      return useQuery({
        queryKey: ["copilot", "session-list"],
        queryFn: async () => {
          fetches += 1;
          return [`chat for fetch ${fetches}`];
        },
      });
    }
    const { result } = renderHook(
      () => {
        useClearOnAccountChange(queryClient);
        return useChats();
      },
      {
        wrapper: ({ children }: { children: ReactNode }) => (
          <QueryClientProvider client={queryClient}>
            {children}
          </QueryClientProvider>
        ),
      },
    );
    await waitFor(() =>
      expect(result.current.data).toEqual(["chat for fetch 1"]),
    );

    signIn("user-2");

    await waitFor(() =>
      expect(result.current.data).toEqual(["chat for fetch 2"]),
    );
  });

  it("keeps the cache when the page's first account arrives", () => {
    // Every page load starts signed out and then learns its session; that is
    // not an account change, and must not throw away what is cached.
    const queryClient = new QueryClient();
    renderHook(() => useClearOnAccountChange(queryClient));
    queryClient.setQueryData(["flags"], { on: true });

    signIn("user-1");

    expect(queryClient.getQueryData(["flags"])).toEqual({ on: true });
  });

  it("drops the last account's cached data when another account signs in on the tab", () => {
    // A sign-out in another tab or a failed session check never runs this
    // tab's logOut(), so its cache (the chat list, among the rest) would
    // otherwise greet the next account.
    const queryClient = new QueryClient();
    signIn("user-1");
    renderHook(() => useClearOnAccountChange(queryClient));
    queryClient.setQueryData(["copilot", "session-list"], ["user-1's chat"]);

    signIn(null);
    expect(queryClient.getQueryData(["copilot", "session-list"])).toEqual([
      "user-1's chat",
    ]);
    signIn("user-1");
    expect(queryClient.getQueryData(["copilot", "session-list"])).toEqual([
      "user-1's chat",
    ]);

    signIn("user-2");
    expect(
      queryClient.getQueryData(["copilot", "session-list"]),
    ).toBeUndefined();
  });
});
