import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mintServiceTokenMock = vi.fn();
vi.mock("../service-token", () => ({
  mintServiceToken: (...args: unknown[]) => mintServiceTokenMock(...args),
}));

import { CLIENT_IP_SCOPE, getClientIPToken } from "../client-ip-token";

function forwardedFor(value: string) {
  return new Headers({ "x-forwarded-for": value });
}

beforeEach(() => {
  vi.stubEnv("VERCEL", "1");
  vi.useFakeTimers();
  mintServiceTokenMock.mockReset();
  mintServiceTokenMock.mockImplementation(
    async (_scope: string, claims: { ip: string }) =>
      `signed-${claims.ip}-${Date.now()}`,
  );
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
});

describe("getClientIPToken", () => {
  it("signs the visitor's address, the first forwarded one, under the client-ip scope", async () => {
    vi.setSystemTime(1_000_000);

    const token = await getClientIPToken(
      forwardedFor(" 203.0.113.7 , 10.0.0.1"),
    );

    expect(token).toBe("signed-203.0.113.7-1000000");
    expect(mintServiceTokenMock).toHaveBeenCalledWith(CLIENT_IP_SCOPE, {
      ip: "203.0.113.7",
    });
  });

  it("reuses a visitor's token for 30 seconds, then mints a fresh one", async () => {
    vi.setSystemTime(2_000_000);
    const first = await getClientIPToken(forwardedFor("2001:db8::1"));

    vi.setSystemTime(2_000_000 + 29_000);
    expect(await getClientIPToken(forwardedFor("2001:db8::1"))).toBe(first);

    vi.setSystemTime(2_000_000 + 30_000);
    expect(await getClientIPToken(forwardedFor("2001:db8::1"))).not.toBe(first);
    expect(mintServiceTokenMock).toHaveBeenCalledTimes(2);
  });

  it("signs nothing without a usable address", async () => {
    expect(await getClientIPToken(new Headers())).toBeNull();
    for (const junk of ["", "unknown", "203.0.113.7; DROP", "a".repeat(46)]) {
      expect(await getClientIPToken(forwardedFor(junk))).toBeNull();
    }
    expect(mintServiceTokenMock).not.toHaveBeenCalled();
  });

  it("signs an address only where the proxy in front vouches for it", async () => {
    vi.stubEnv("VERCEL", "");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "");
    vi.stubEnv("TRUST_X_FORWARDED_FOR", "");

    expect(await getClientIPToken(forwardedFor("198.51.100.8"))).toBeNull();
    expect(mintServiceTokenMock).not.toHaveBeenCalled();

    vi.stubEnv("TRUST_X_FORWARDED_FOR", "true");
    expect(await getClientIPToken(forwardedFor("198.51.100.8"))).not.toBeNull();
  });

  it("answers null when signing fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mintServiceTokenMock.mockRejectedValueOnce(new Error("no jwks"));

    expect(await getClientIPToken(forwardedFor("198.51.100.4"))).toBeNull();
  });
});
