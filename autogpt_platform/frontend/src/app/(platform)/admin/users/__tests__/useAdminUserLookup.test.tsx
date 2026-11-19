import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { ReactNode } from "react";
import { describe, expect, it } from "vitest";
import { useAdminUserLookup } from "../useAdminUserLookup";

const FIRST_LAST = authUser({
  id: "11111111-1111-4111-8111-111111111111",
  email: "first_last@example.com",
});
const FIRSTXLAST = authUser({
  id: "22222222-2222-4222-8222-222222222222",
  email: "firstxlast@example.com",
});

describe("useAdminUserLookup", () => {
  it("looks an email up by exact match, so `_` matches only itself", async () => {
    const requests = mockListUsers([FIRSTXLAST, FIRST_LAST]);
    const { result } = renderLookup();

    act(() => result.current.search("  First_Last@Example.com "));

    await waitFor(() => expect(result.current.user).not.toBeNull());
    expect(result.current.user?.id).toBe(FIRST_LAST.id);
    // React Query may refetch under load; every request must be the exact one.
    expect(requests.length).toBeGreaterThan(0);
    for (const params of requests) {
      expect(params.get("filterField")).toBe("email");
      expect(params.get("filterOperator")).toBe("eq");
      expect(params.get("filterValue")).toBe("first_last@example.com");
      expect(params.get("searchValue")).toBeNull();
    }
  });

  it("reports an email with no exact match as not found", async () => {
    mockListUsers([FIRSTXLAST]);
    const { result } = renderLookup();

    act(() => result.current.search("first_last@example.com"));

    await waitFor(() => expect(result.current.isNotFound).toBe(true));
    expect(result.current.user).toBeNull();
  });

  it("looks an ID up through get-user", async () => {
    const requestedIds: (string | null)[] = [];
    server.use(
      http.get("*/api/auth/admin/get-user", ({ request }) => {
        requestedIds.push(new URL(request.url).searchParams.get("id"));
        return HttpResponse.json({ ...FIRST_LAST, banned: true });
      }),
    );
    const { result } = renderLookup();

    act(() => result.current.search(FIRST_LAST.id));

    await waitFor(() => expect(result.current.user).not.toBeNull());
    expect([...new Set(requestedIds)]).toEqual([FIRST_LAST.id]);
    expect(result.current.user?.banned).toBe(true);
    expect(result.current.user?.createdAt).toBeInstanceOf(Date);
  });

  it("reports an unknown ID as not found rather than as an error", async () => {
    server.use(
      http.get("*/api/auth/admin/get-user", () =>
        HttpResponse.json(
          { code: "USER_NOT_FOUND", message: "User not found" },
          { status: 404 },
        ),
      ),
    );
    const { result } = renderLookup();

    act(() => result.current.search("33333333-3333-4333-8333-333333333333"));

    await waitFor(() => expect(result.current.isNotFound).toBe(true));
    expect(result.current.error).toBeNull();
  });

  it("surfaces a refused lookup as an error", async () => {
    server.use(
      http.get("*/api/auth/admin/get-user", () =>
        HttpResponse.json(
          { code: "FORBIDDEN", message: "You are not allowed to get user" },
          { status: 403 },
        ),
      ),
    );
    const { result } = renderLookup();

    act(() => result.current.search(FIRST_LAST.id));

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.message).toBe(
      "You are not allowed to get user",
    );
    expect(result.current.isNotFound).toBe(false);
  });
});

function renderLookup() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <NuqsTestingAdapter hasMemory>{children}</NuqsTestingAdapter>
      </QueryClientProvider>
    );
  }
  return renderHook(() => useAdminUserLookup(), { wrapper: Wrapper });
}

// Answers like the real endpoint for the exact filter it is sent, and with
// every row for anything looser, as a LIKE with `_` in it would.
function mockListUsers(users: ReturnType<typeof authUser>[]) {
  const requests: URLSearchParams[] = [];
  server.use(
    http.get("*/api/auth/admin/list-users", ({ request }) => {
      const params = new URL(request.url).searchParams;
      requests.push(params);
      const exact =
        params.get("filterField") === "email" &&
        params.get("filterOperator") === "eq";
      const matches = exact
        ? users.filter((user) => user.email === params.get("filterValue"))
        : users;
      return HttpResponse.json({ users: matches, total: matches.length });
    }),
  );
  return requests;
}

function authUser({ id, email }: { id: string; email: string }) {
  return {
    id,
    email,
    name: "Test User",
    image: null,
    role: "user",
    emailVerified: true,
    createdAt: "2026-01-02T03:04:05.000Z",
    updatedAt: "2026-01-02T03:04:05.000Z",
    banned: false,
    banReason: null,
    banExpires: null,
  };
}
