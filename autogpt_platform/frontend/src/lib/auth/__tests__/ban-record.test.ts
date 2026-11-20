import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIGN_UP_BLOCKED_CODE } from "../access-blocks";

const mintServiceTokenMock = vi.fn();
vi.mock("../service-token", () => ({
  mintServiceToken: (...args: unknown[]) => mintServiceTokenMock(...args),
}));

import { assertNotBannedAddress, BAN_RECORD_SCOPE } from "../ban-record";

let fetchMock: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  mintServiceTokenMock.mockReset();
  mintServiceTokenMock.mockResolvedValue("service-token");
  fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValue(Response.json({ banned: false }));
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("assertNotBannedAddress", () => {
  it("asks the backend with a service token under its own scope", async () => {
    await assertNotBannedAddress("sam@example.com");

    expect(mintServiceTokenMock).toHaveBeenCalledWith(BAN_RECORD_SCOPE);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toMatch(/\/api\/auth\/email\/ban-record$/);
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("Authorization")).toBe(
      "Bearer service-token",
    );
    expect(JSON.parse(String(init.body))).toEqual({ email: "sam@example.com" });
  });

  it("refuses an address with a ban record, with the generic block sentence", async () => {
    fetchMock.mockResolvedValue(Response.json({ banned: true }));

    await expect(
      assertNotBannedAddress("sam@example.com"),
    ).rejects.toMatchObject({
      statusCode: 403,
      body: {
        code: SIGN_UP_BLOCKED_CODE,
        message: expect.not.stringContaining("ban"),
      },
    });
  });

  it.each([
    [
      "an error status",
      () => fetchMock.mockResolvedValue(new Response("", { status: 503 })),
    ],
    [
      "a network error",
      () => fetchMock.mockRejectedValue(new TypeError("fetch failed")),
    ],
    [
      "a token that can't be minted",
      () => mintServiceTokenMock.mockRejectedValue(new Error("no key")),
    ],
  ])("lets the sign-up in on %s", async (_case, arrange) => {
    arrange();
    vi.spyOn(console, "warn").mockImplementation(() => {});

    await expect(
      assertNotBannedAddress("sam@example.com"),
    ).resolves.toBeUndefined();
  });
});
