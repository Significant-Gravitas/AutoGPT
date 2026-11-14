import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", () => ({ captureRequestError: vi.fn() }));
vi.mock("../../sentry.server.config", () => ({}));
vi.mock("../../sentry.edge.config", () => ({}));

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  vi.resetModules();
});

async function register() {
  const instrumentation = await import("../instrumentation");
  return instrumentation.register();
}

describe("register", () => {
  it("stops the server starting on a TRUSTED_PROXIES entry it cannot parse", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("TRUSTED_PROXIES", "10.0.0.1, 10.0.0.0/99");

    await expect(register()).rejects.toThrow("cannot parse: 10.0.0.0/99");
  });

  it("starts on a list of addresses and CIDR ranges", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("TRUSTED_PROXIES", "10.0.0.1, 192.168.0.0/16");

    await expect(register()).resolves.toBeUndefined();
  });

  it("warns that TRUST_X_FORWARDED_FOR is ignored when no proxies are listed", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("TRUSTED_PROXIES", "");
    vi.stubEnv("TRUST_X_FORWARDED_FOR", "true");
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    await register();

    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining(
        "TRUST_X_FORWARDED_FOR is deprecated and ignored",
      ),
    );
  });

  it("starts with no proxies listed", async () => {
    vi.stubEnv("NEXT_RUNTIME", "nodejs");
    vi.stubEnv("TRUSTED_PROXIES", "");

    await expect(register()).resolves.toBeUndefined();
  });
});
