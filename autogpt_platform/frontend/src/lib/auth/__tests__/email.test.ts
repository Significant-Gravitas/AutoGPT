import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const createTransportMock = vi.fn();

vi.mock("nodemailer", () => ({
  default: {
    createTransport: (...args: unknown[]) => createTransportMock(...args),
  },
}));

import { sendAuthEmail } from "../email";

const resetUrl = "https://platform.example.com/reset-password?token=abc123";
const email = {
  to: "user@example.com",
  subject: "Reset your AutoGPT Platform password",
  text: `Click the link to reset your password: ${resetUrl}`,
};

beforeEach(() => {
  createTransportMock.mockReset();
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("sendAuthEmail", () => {
  it("delivers mail through SMTP when SMTP_HOST is configured", async () => {
    vi.stubEnv("SMTP_HOST", "smtp.example.com");
    vi.stubEnv("SMTP_PORT", "2525");
    vi.stubEnv("SMTP_SECURE", "true");
    vi.stubEnv("SMTP_USER", "mailer");
    vi.stubEnv("SMTP_PASS", "hunter2");
    vi.stubEnv("SMTP_FROM", "AutoGPT Platform <no-reply@example.com>");
    const sendMailMock = vi.fn().mockResolvedValue(undefined);
    createTransportMock.mockReturnValue({ sendMail: sendMailMock });

    await sendAuthEmail(email);

    expect(createTransportMock).toHaveBeenCalledWith({
      host: "smtp.example.com",
      port: 2525,
      secure: true,
      auth: { user: "mailer", pass: "hunter2" },
    });
    expect(sendMailMock).toHaveBeenCalledWith({
      from: "AutoGPT Platform <no-reply@example.com>",
      to: email.to,
      subject: email.subject,
      text: email.text,
    });
  });

  it("falls back to port 587 without auth when only SMTP_HOST is set", async () => {
    vi.stubEnv("SMTP_HOST", "smtp.example.com");
    vi.stubEnv("SMTP_PORT", "");
    vi.stubEnv("SMTP_SECURE", "");
    vi.stubEnv("SMTP_USER", "");
    vi.stubEnv("SMTP_FROM", "");
    const sendMailMock = vi.fn().mockResolvedValue(undefined);
    createTransportMock.mockReturnValue({ sendMail: sendMailMock });

    await sendAuthEmail(email);

    expect(createTransportMock).toHaveBeenCalledWith({
      host: "smtp.example.com",
      port: 587,
      secure: false,
      auth: undefined,
    });
    expect(sendMailMock).toHaveBeenCalledWith(
      expect.objectContaining({
        from: "AutoGPT Platform <no-reply@localhost>",
      }),
    );
  });

  it("logs the auth link to the console in development when SMTP is missing", async () => {
    vi.stubEnv("SMTP_HOST", "");
    vi.stubEnv("NODE_ENV", "development");
    const infoSpy = vi
      .spyOn(console, "info")
      .mockImplementation(() => undefined);

    await sendAuthEmail(email);

    expect(createTransportMock).not.toHaveBeenCalled();
    const loggedLines = infoSpy.mock.calls.map((call) => String(call[0]));
    expect(loggedLines.some((line) => line.includes(email.text))).toBe(true);
  });

  it("fails the auth flow without leaking the link in production", async () => {
    vi.stubEnv("SMTP_HOST", "");
    vi.stubEnv("NODE_ENV", "production");
    const infoSpy = vi
      .spyOn(console, "info")
      .mockImplementation(() => undefined);

    // Throwing (instead of silently returning) keeps the reset-password UI
    // from claiming "Email sent" when nothing was delivered.
    await expect(sendAuthEmail(email)).rejects.toThrow(
      "SMTP is not configured",
    );

    expect(createTransportMock).not.toHaveBeenCalled();
    const loggedLines = infoSpy.mock.calls.map((call) => String(call[0]));
    expect(loggedLines.some((line) => line.includes(resetUrl))).toBe(false);
  });
});
