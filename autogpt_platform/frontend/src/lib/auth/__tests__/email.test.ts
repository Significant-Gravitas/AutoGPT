import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: () => "https://backend.example.com",
    isCloud: () => true,
  },
}));

const mintServiceTokenMock = vi.fn();
vi.mock("../service-token", () => ({
  mintServiceToken: (...args: unknown[]) => mintServiceTokenMock(...args),
}));

import { Resolver } from "node:dns/promises";
import { sendAuthEmail } from "../email";
import { clearMailServerCache } from "../email-deliverability";

const resetUrl = "https://platform.example.com/reset-password?token=abc123";
const args = {
  to: "user@example.com",
  type: "reset_password" as const,
  url: resetUrl,
};

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  mintServiceTokenMock.mockReset();
  mintServiceTokenMock.mockResolvedValue("svc-token");
  vi.stubGlobal("fetch", fetchMock);
  clearMailServerCache();
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("sendAuthEmail", () => {
  it("POSTs the link to the backend mailer with a scoped service token", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await sendAuthEmail(args);

    expect(mintServiceTokenMock).toHaveBeenCalledWith("auth-email:send");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://backend.example.com/api/auth/email/send");
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["Authorization"]).toBe(
      "Bearer svc-token",
    );
    expect(JSON.parse(init.body as string)).toEqual({
      type: "reset_password",
      to: "user@example.com",
      url: resetUrl,
    });
  });

  it("throws in production when the backend returns a non-2xx response", async () => {
    vi.stubEnv("NODE_ENV", "production");
    fetchMock.mockResolvedValue(new Response("mailer down", { status: 503 }));

    // Throwing (instead of silently returning) keeps the reset-password UI
    // from claiming "Email sent" when nothing was delivered.
    await expect(sendAuthEmail(args)).rejects.toThrow(
      "Backend auth-email send failed (503)",
    );
  });

  it("throws in production when minting the service token fails", async () => {
    vi.stubEnv("NODE_ENV", "production");
    mintServiceTokenMock.mockRejectedValue(new Error("no signing key"));

    await expect(sendAuthEmail(args)).rejects.toThrow("no signing key");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("logs the auth link instead of throwing when the send fails outside production", async () => {
    vi.stubEnv("NODE_ENV", "development");
    fetchMock.mockResolvedValue(new Response("mailer down", { status: 503 }));
    const infoSpy = vi
      .spyOn(console, "info")
      .mockImplementation(() => undefined);

    await sendAuthEmail(args);

    const logged = infoSpy.mock.calls.map((c) => String(c[0]));
    expect(logged.some((l) => l.includes(resetUrl))).toBe(true);
  });

  it("does not log the auth link in production", async () => {
    vi.stubEnv("NODE_ENV", "production");
    fetchMock.mockResolvedValue(new Response(null, { status: 500 }));
    const infoSpy = vi
      .spyOn(console, "info")
      .mockImplementation(() => undefined);

    await expect(sendAuthEmail(args)).rejects.toThrow();

    const logged = infoSpy.mock.calls.map((c) => String(c[0]));
    expect(logged.some((l) => l.includes(resetUrl))).toBe(false);
  });

  it.each([
    ["a domain that does not exist", "ENOTFOUND"],
    ["a domain with no MX, A or AAAA record", "ENODATA"],
  ])("skips the send to %s and reports success", async (_, code) => {
    vi.stubEnv("NODE_ENV", "production");
    const error = Object.assign(new Error(code), { code });
    vi.spyOn(Resolver.prototype, "resolveMx").mockRejectedValue(error);
    vi.spyOn(Resolver.prototype, "resolve4").mockRejectedValue(error);
    vi.spyOn(Resolver.prototype, "resolve6").mockRejectedValue(error);
    const warnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);

    await expect(sendAuthEmail(args)).resolves.toBeUndefined();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(JSON.stringify(warnSpy.mock.calls)).not.toContain(args.to);
  });

  it("still sends to a disposable domain", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await sendAuthEmail({ ...args, to: "user@mailinator.com" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("still sends when the DNS lookup errors", async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));
    vi.spyOn(Resolver.prototype, "resolveMx").mockRejectedValue(
      Object.assign(new Error("ESERVFAIL"), { code: "ESERVFAIL" }),
    );

    await sendAuthEmail(args);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
