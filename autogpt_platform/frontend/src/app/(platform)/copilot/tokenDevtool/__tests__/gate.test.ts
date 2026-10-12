import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const isDevelopmentBuild = vi.fn(() => false);
const isDev = vi.fn(() => false);

vi.mock("@/services/environment", () => ({
  environment: {
    isDevelopmentBuild: () => isDevelopmentBuild(),
    isDev: () => isDev(),
  },
}));

import { isTokenDevtoolEnabled } from "../gate";

beforeEach(() => {
  isDevelopmentBuild.mockReturnValue(false);
  isDev.mockReturnValue(false);
  vi.stubEnv("NEXT_PUBLIC_TOKEN_DEVTOOL", "true");
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("isTokenDevtoolEnabled", () => {
  it("is on for a development build", () => {
    isDevelopmentBuild.mockReturnValue(true);
    expect(isTokenDevtoolEnabled()).toBe(true);
  });

  it("is on for the cloud dev deployment", () => {
    isDev.mockReturnValue(true);
    expect(isTokenDevtoolEnabled()).toBe(true);
  });

  it("is off for a production build even when the flag is unset", () => {
    vi.stubEnv("NEXT_PUBLIC_TOKEN_DEVTOOL", undefined);
    expect(isTokenDevtoolEnabled()).toBe(false);
  });

  // The self-hosted single-container image builds with NODE_ENV=production and
  // NEXT_PUBLIC_BEHAVE_AS=LOCAL, and copies .env.default (which sets the flag
  // to true) into .env. Gating on the build type is what keeps it out.
  it("stays off for a self-hosted production build that opted in", () => {
    vi.stubEnv("NEXT_PUBLIC_TOKEN_DEVTOOL", "true");
    expect(isTokenDevtoolEnabled()).toBe(false);
  });

  it.each(["false", "FALSE", " False ", "0", "off"])(
    "the %s kill switch wins over a development build",
    (value) => {
      isDevelopmentBuild.mockReturnValue(true);
      vi.stubEnv("NEXT_PUBLIC_TOKEN_DEVTOOL", value);
      expect(isTokenDevtoolEnabled()).toBe(false);
    },
  );
});
