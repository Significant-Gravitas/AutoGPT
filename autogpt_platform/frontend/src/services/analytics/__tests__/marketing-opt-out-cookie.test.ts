import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  cookies: vi.fn(),
  cookieGet: vi.fn(),
  cookieDelete: vi.fn(),
  captureException: vi.fn(),
}));

vi.mock("next/headers", () => ({ cookies: mocks.cookies }));
vi.mock("@sentry/nextjs", () => ({
  captureException: mocks.captureException,
}));

import {
  MARKETING_OPT_OUT_COOKIE,
  setMarketingOptOutFlag,
} from "../marketing-opt-out-cookie";
import { takeMarketingOptOutFlag } from "../marketing-opt-out-server";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("setMarketingOptOutFlag", () => {
  function captureCookieWrites() {
    return vi
      .spyOn(document, "cookie", "set")
      .mockImplementation(() => undefined);
  }

  it("writes a short-lived Secure refusal outside local stacks", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    const write = captureCookieWrites();

    setMarketingOptOutFlag(true);

    expect(write).toHaveBeenCalledExactlyOnceWith(
      "agpt_marketing_opt_out=1; Path=/; Max-Age=600; SameSite=Lax; Secure",
    );
  });

  it("drops Secure on local stacks, which run on http", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    const write = captureCookieWrites();

    setMarketingOptOutFlag(true);

    expect(write).toHaveBeenCalledExactlyOnceWith(
      "agpt_marketing_opt_out=1; Path=/; Max-Age=600; SameSite=Lax",
    );
  });

  it("clears the flag instead of writing a non-refusal", () => {
    const write = captureCookieWrites();

    setMarketingOptOutFlag(false);

    expect(write).toHaveBeenCalledExactlyOnceWith(
      "agpt_marketing_opt_out=; Path=/; Max-Age=0",
    );
  });

  it("removes a refusal left by an earlier attempt that was undone", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");

    setMarketingOptOutFlag(true);
    expect(document.cookie).toContain(`${MARKETING_OPT_OUT_COOKIE}=1`);

    setMarketingOptOutFlag(false);
    expect(document.cookie).not.toContain(`${MARKETING_OPT_OUT_COOKIE}=1`);
  });
});

describe("takeMarketingOptOutFlag", () => {
  beforeEach(() => {
    mocks.cookies.mockReset().mockResolvedValue({
      get: mocks.cookieGet,
      delete: mocks.cookieDelete,
    });
    mocks.cookieGet.mockReset();
    mocks.cookieDelete.mockReset();
    mocks.captureException.mockReset();
  });

  it("returns true for a refusal and deletes the cookie", async () => {
    mocks.cookieGet.mockReturnValue({
      name: MARKETING_OPT_OUT_COOKIE,
      value: "1",
    });

    await expect(takeMarketingOptOutFlag()).resolves.toBe(true);

    expect(mocks.cookieGet).toHaveBeenCalledWith(MARKETING_OPT_OUT_COOKIE);
    expect(mocks.cookieDelete).toHaveBeenCalledExactlyOnceWith(
      MARKETING_OPT_OUT_COOKIE,
    );
  });

  it("treats any other value as not refused but still deletes it", async () => {
    mocks.cookieGet.mockReturnValue({
      name: MARKETING_OPT_OUT_COOKIE,
      value: "true",
    });

    await expect(takeMarketingOptOutFlag()).resolves.toBe(false);

    expect(mocks.cookieDelete).toHaveBeenCalledExactlyOnceWith(
      MARKETING_OPT_OUT_COOKIE,
    );
  });

  it("returns false and leaves the store alone when there is no cookie", async () => {
    mocks.cookieGet.mockReturnValue(undefined);

    await expect(takeMarketingOptOutFlag()).resolves.toBe(false);

    expect(mocks.cookieDelete).not.toHaveBeenCalled();
  });

  it("returns false and reports the error when the cookie store fails", async () => {
    const error = new Error("cookies() called outside a request scope");
    mocks.cookies.mockRejectedValue(error);

    await expect(takeMarketingOptOutFlag()).resolves.toBe(false);

    expect(mocks.captureException).toHaveBeenCalledExactlyOnceWith(error, {
      tags: { signup_step: "read_marketing_opt_out" },
    });
  });
});
