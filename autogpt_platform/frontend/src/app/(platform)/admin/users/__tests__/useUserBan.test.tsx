import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { type AdminUser, parseLookupQuery, toAdminUser } from "../helpers";
import { getAdminUserLookupQueryKey } from "../useAdminUserLookup";
import { useUserBan } from "../useUserBan";

const mockToast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

const RECORD = {
  id: "11111111-1111-4111-8111-111111111111",
  email: "first_last@example.com",
  name: "Test User",
  image: null,
  role: "user",
  emailVerified: true,
  createdAt: "2026-01-02T03:04:05.000Z",
  banned: false,
  banReason: null,
  banExpires: null,
};
const USER = toAdminUser(RECORD);

beforeEach(() => mockToast.mockReset());

describe("useUserBan", () => {
  it("bans through Better Auth with the reason and expiry, and shows the banned user", async () => {
    const bodies: unknown[] = [];
    server.use(
      http.post("*/api/auth/admin/ban-user", async ({ request }) => {
        bodies.push(await request.json());
        return HttpResponse.json({
          user: {
            ...RECORD,
            banned: true,
            banReason: "spam",
            banExpires: new Date(Date.now() + 7 * 24 * 3600 * 1000),
          },
        });
      }),
    );
    const { result, queryClient } = renderBan(USER);

    act(() => result.current.banUser({ reason: " spam ", duration: "7d" }));

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      userId: USER.id,
      banReason: "spam",
      banExpiresIn: 7 * 24 * 60 * 60,
    });
    await waitFor(() =>
      expect(cachedUser(queryClient, USER.email)?.banned).toBe(true),
    );
    expect(cachedUser(queryClient, USER.id)?.banReason).toBe("spam");
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "success" }),
    );
  });

  it("sends no expiry for a permanent ban", async () => {
    const bodies: Record<string, unknown>[] = [];
    server.use(
      http.post("*/api/auth/admin/ban-user", async ({ request }) => {
        bodies.push((await request.json()) as Record<string, unknown>);
        return HttpResponse.json({ user: { ...RECORD, banned: true } });
      }),
    );
    const { result } = renderBan(USER);

    act(() =>
      result.current.banUser({ reason: "spam", duration: "permanent" }),
    );

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({ userId: USER.id, banReason: "spam" });
  });

  it("refuses a ban without a reason and sends nothing", async () => {
    const bodies: unknown[] = [];
    server.use(
      http.post("*/api/auth/admin/ban-user", async ({ request }) => {
        bodies.push(await request.json());
        return HttpResponse.json({ user: { ...RECORD, banned: true } });
      }),
    );
    const { result } = renderBan(USER);

    act(() => result.current.banUser({ reason: "   ", duration: "1h" }));

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "destructive" }),
      ),
    );
    expect(bodies).toEqual([]);
  });

  it("shows why Better Auth refused a ban and leaves the user unbanned", async () => {
    server.use(
      http.post("*/api/auth/admin/ban-user", () =>
        HttpResponse.json(
          {
            code: "YOU_CANNOT_BAN_YOURSELF",
            message: "You cannot ban yourself",
          },
          { status: 400 },
        ),
      ),
    );
    const { result, queryClient } = renderBan(USER);

    act(() => result.current.banUser({ reason: "spam", duration: "1h" }));

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
          description: "You cannot ban yourself",
        }),
      ),
    );
    expect(cachedUser(queryClient, USER.id)?.banned).toBe(false);
  });

  it("unbans through Better Auth and shows the unbanned user", async () => {
    const bodies: unknown[] = [];
    server.use(
      http.post("*/api/auth/admin/unban-user", async ({ request }) => {
        bodies.push(await request.json());
        return HttpResponse.json({ user: RECORD });
      }),
    );
    const banned: AdminUser = { ...USER, banned: true, banReason: "spam" };
    const { result, queryClient } = renderBan(banned);

    act(() => result.current.unbanUser());

    await waitFor(() => expect(bodies).toEqual([{ userId: USER.id }]));
    await waitFor(() =>
      expect(cachedUser(queryClient, USER.id)?.banned).toBe(false),
    );
  });
});

function renderBan(user: AdminUser) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  for (const lookup of [user.id, user.email]) {
    queryClient.setQueryData(
      getAdminUserLookupQueryKey(parseLookupQuery(lookup)),
      user,
    );
  }
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }
  const rendered = renderHook(() => useUserBan(user), { wrapper: Wrapper });
  return { ...rendered, queryClient };
}

function cachedUser(queryClient: QueryClient, lookup: string) {
  return queryClient.getQueryData<AdminUser>(
    getAdminUserLookupQueryKey(parseLookupQuery(lookup)),
  );
}
