import { beforeEach, describe, expect, it, vi } from "vitest";

const getServerSessionMock = vi.fn();
const listUserAccountsMock = vi.fn();

vi.mock("@/lib/auth/server/getServerSession", () => ({
  getServerSession: () => getServerSessionMock(),
}));

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      listUserAccounts: (...args: unknown[]) => listUserAccountsMock(...args),
    },
  },
}));

import { GET } from "../route";

function request() {
  return new Request("http://localhost:3000/api/auth/sign-in-providers");
}

beforeEach(() => {
  getServerSessionMock.mockReset();
  listUserAccountsMock.mockReset();
});

describe("GET /api/auth/sign-in-providers", () => {
  it("names each provider once and nothing else about the accounts", async () => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    listUserAccountsMock.mockResolvedValue([
      { providerId: "credential", accountId: "user-1", accessToken: "x" },
      { providerId: "microsoft", accountId: "object-id", idToken: "y" },
      { providerId: "microsoft", accountId: "tenant:other", idToken: "z" },
    ]);

    const signedIn = request();
    const response = await GET(signedIn);

    expect(await response.json()).toEqual({
      providers: ["credential", "microsoft"],
    });
    // The account lookup runs as the person making the request.
    expect(listUserAccountsMock).toHaveBeenCalledWith({
      headers: signedIn.headers,
    });
  });

  it("refuses a request without a session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const response = await GET(request());

    expect(response.status).toBe(401);
    expect(listUserAccountsMock).not.toHaveBeenCalled();
  });
});
