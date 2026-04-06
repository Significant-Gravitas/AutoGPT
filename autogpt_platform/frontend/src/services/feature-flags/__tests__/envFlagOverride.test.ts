import { describe, it, expect, beforeEach, vi } from "vitest";
import { envFlagOverride, Flag } from "../use-get-flag";

vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => ({}),
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => false },
}));

const ENV_KEY = "NEXT_PUBLIC_FORCE_FLAG_CHAT_MODE_OPTION";

describe("envFlagOverride", () => {
  beforeEach(() => {
    delete process.env[ENV_KEY];
  });

  it('returns true when env var is "true"', () => {
    process.env[ENV_KEY] = "true";
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBe(true);
  });

  it('returns false when env var is "false"', () => {
    process.env[ENV_KEY] = "false";
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBe(false);
  });

  it('returns true when env var is "1"', () => {
    process.env[ENV_KEY] = "1";
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBe(true);
  });

  it('returns true when env var is "yes"', () => {
    process.env[ENV_KEY] = "yes";
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBe(true);
  });

  it('returns true when env var is "on"', () => {
    process.env[ENV_KEY] = "on";
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBe(true);
  });

  it("returns undefined when env var is not set", () => {
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBeUndefined();
  });

  it("returns undefined for an empty string", () => {
    process.env[ENV_KEY] = "";
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBeUndefined();
  });

  it("returns undefined for an unrecognised string", () => {
    process.env[ENV_KEY] = "banana";
    expect(envFlagOverride(Flag.CHAT_MODE_OPTION)).toBeUndefined();
  });
});
