import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/environment", () => ({
  environment: {
    getAGPTServerBaseUrl: () => "https://backend.example.com",
  },
}));

import { sendAuthEmail } from "../email";

const resetUrl = "https://platform.example.com/reset-password?token=abc123";
const args = {
  to: "user@example.com",
  type: "reset_password" as const,
  url: resetUrl,
};

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("sendAuthEmail", () => {
  it("POSTs the link to the backend mailer with the shared token", async () => {
    vi.stubEnv("AUTH_EMAIL_TOKEN", "s3cret");
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await sendAuthEmail(args);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://backend.example.com/api/auth-email/send");
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["X-Auth-Email-Token"]).toBe(
      "s3cret",
    );
    expect(JSON.parse(init.body as string)).toEqual({
      type: "reset_password",
      to: "user@example.com",
      url: resetUrl,
    });
  });

  it("logs the auth link in development when the token is missing", async () => {
    vi.stubEnv("AUTH_EMAIL_TOKEN", "");
    vi.stubEnv("NODE_ENV", "development");
    const infoSpy = vi
      .spyOn(console, "info")
      .mockImplementation(() => undefined);

    await sendAuthEmail(args);

    expect(fetchMock).not.toHaveBeenCalled();
    const logged = infoSpy.mock.calls.map((c) => String(c[0]));
    expect(logged.some((l) => l.includes(resetUrl))).toBe(true);
  });

  it("throws in production without leaking the link when the token is missing", async () => {
    vi.stubEnv("AUTH_EMAIL_TOKEN", "");
    vi.stubEnv("NODE_ENV", "production");
    const infoSpy = vi
      .spyOn(console, "info")
      .mockImplementation(() => undefined);

    // Throwing (instead of silently returning) keeps the reset-password UI
    // from claiming "Email sent" when nothing was delivered.
    await expect(sendAuthEmail(args)).rejects.toThrow(
      "AUTH_EMAIL_TOKEN is not set",
    );

    expect(fetchMock).not.toHaveBeenCalled();
    const logged = infoSpy.mock.calls.map((c) => String(c[0]));
    expect(logged.some((l) => l.includes(resetUrl))).toBe(false);
  });

  it("throws when the backend returns a non-2xx response", async () => {
    vi.stubEnv("AUTH_EMAIL_TOKEN", "s3cret");
    fetchMock.mockResolvedValue(new Response("mailer down", { status: 503 }));

    await expect(sendAuthEmail(args)).rejects.toThrow(
      "Backend auth-email send failed (503)",
    );
  });
});
