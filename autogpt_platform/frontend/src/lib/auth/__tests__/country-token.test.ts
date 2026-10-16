import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mintServiceTokenMock = vi.fn();
vi.mock("../service-token", () => ({
  mintServiceToken: (...args: unknown[]) => mintServiceTokenMock(...args),
}));

import { CLIENT_COUNTRY_SCOPE, getCountryToken } from "../country-token";

beforeEach(() => {
  vi.useFakeTimers();
  mintServiceTokenMock.mockReset();
  mintServiceTokenMock.mockImplementation(
    async (_scope: string, claims: { country: string }) =>
      `signed-${claims.country}-${Date.now()}`,
  );
});

afterEach(() => {
  vi.useRealTimers();
});

describe("getCountryToken", () => {
  it("signs the normalised country under the client-country scope", async () => {
    vi.setSystemTime(1_000_000);

    const token = await getCountryToken(" de ");

    expect(token).toBe("signed-DE-1000000");
    expect(mintServiceTokenMock).toHaveBeenCalledWith(CLIENT_COUNTRY_SCOPE, {
      country: "DE",
    });
  });

  it("reuses a token for 30 seconds, then mints a fresh one", async () => {
    vi.setSystemTime(2_000_000);
    const first = await getCountryToken("FR");

    vi.setSystemTime(2_000_000 + 29_000);
    expect(await getCountryToken("FR")).toBe(first);

    vi.setSystemTime(2_000_000 + 30_000);
    expect(await getCountryToken("FR")).not.toBe(first);
    expect(mintServiceTokenMock).toHaveBeenCalledTimes(2);
  });

  it("signs nothing that is not a two-letter code", async () => {
    for (const junk of ["IN, US", "", "IND", "1N"]) {
      expect(await getCountryToken(junk)).toBeNull();
    }
    expect(mintServiceTokenMock).not.toHaveBeenCalled();
  });
});
