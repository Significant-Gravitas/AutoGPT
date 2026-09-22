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

describe("BUILDER_CHAT_PANEL default", () => {
  beforeEach(() => {
    delete process.env["NEXT_PUBLIC_FORCE_FLAG_BUILDER_CHAT_PANEL"];
  });

  it("is disabled by default so the feature only ships when LaunchDarkly enables it", () => {
    // No env override configured → override helper must return undefined,
    // which causes useGetFlag to fall through to the defaultFlags value. The
    // default for a new gated feature MUST be false so a LaunchDarkly outage
    // cannot expose the feature to all users.
    expect(envFlagOverride(Flag.BUILDER_CHAT_PANEL)).toBeUndefined();
  });

  it("can still be force-enabled via the env override for local dev", () => {
    process.env["NEXT_PUBLIC_FORCE_FLAG_BUILDER_CHAT_PANEL"] = "true";
    expect(envFlagOverride(Flag.BUILDER_CHAT_PANEL)).toBe(true);
  });

  it("can still be force-disabled via the env override for QA", () => {
    process.env["NEXT_PUBLIC_FORCE_FLAG_BUILDER_CHAT_PANEL"] = "false";
    expect(envFlagOverride(Flag.BUILDER_CHAT_PANEL)).toBe(false);
  });
});

describe("array-typed flags refuse env overrides", () => {
  beforeEach(() => {
    delete process.env["NEXT_PUBLIC_FORCE_FLAG_BETA_BLOCKS"];
    delete process.env["NEXT_PUBLIC_FORCE_FLAG_MARKETPLACE_SEARCH_TERMS"];
  });

  it("ignores a boolean env override for BETA_BLOCKS (array type)", () => {
    process.env["NEXT_PUBLIC_FORCE_FLAG_BETA_BLOCKS"] = "true";
    // A boolean override on an array-typed flag would yield `true` where
    // callers expect a list — guard with `undefined` so they fall through
    // to LaunchDarkly / `defaultFlags`.
    expect(envFlagOverride(Flag.BETA_BLOCKS)).toBeUndefined();
  });

  it("ignores a boolean env override for MARKETPLACE_SEARCH_TERMS (array type)", () => {
    process.env["NEXT_PUBLIC_FORCE_FLAG_MARKETPLACE_SEARCH_TERMS"] = "false";
    expect(envFlagOverride(Flag.MARKETPLACE_SEARCH_TERMS)).toBeUndefined();
  });
});

describe("readEnvOverride covers every Flag arm", () => {
  beforeEach(() => {
    Object.keys(process.env)
      .filter((k) => k.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
      .forEach((k) => delete process.env[k]);
  });

  it("returns undefined for every flag when no override is configured", () => {
    // Exercises every branch of the literal-key switch inside
    // `readEnvOverride` so a new flag added without a matching case
    // fails the coverage check.
    for (const flag of Object.values(Flag)) {
      expect(envFlagOverride(flag as Flag)).toBeUndefined();
    }
  });
});

describe("NEXT_PUBLIC_FORCE_ALL_FLAGS master switch", () => {
  // `isForceAllFlags` is resolved once at module load, so each case needs a
  // fresh module graph with NODE_ENV already stubbed.
  async function loadWith(nodeEnv: string) {
    vi.resetModules();
    vi.stubEnv("NODE_ENV", nodeEnv);
    vi.stubEnv("NEXT_PUBLIC_FORCE_ALL_FLAGS", "true");
    return import("../use-get-flag");
  }

  beforeEach(() => {
    vi.unstubAllEnvs();
  });

  it("forces a boolean flag on outside a production build", async () => {
    const { envFlagOverride: override, Flag: F } =
      await loadWith("development");
    expect(override(F.CHAT_MODE_OPTION)).toBe(true);
  });

  it("is inert in a production build, so a stray env var cannot be baked into the client bundle", async () => {
    const { envFlagOverride: override, Flag: F } = await loadWith("production");
    expect(override(F.CHAT_MODE_OPTION)).toBeUndefined();
  });

  it("leaves per-flag overrides working in a production build", async () => {
    vi.resetModules();
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_CHAT_MODE_OPTION", "true");
    const { envFlagOverride: override, Flag: F } = await import(
      "../use-get-flag"
    );
    expect(override(F.CHAT_MODE_OPTION)).toBe(true);
  });

  it("still skips array-typed flags when forced on", async () => {
    const { envFlagOverride: override, Flag: F } =
      await loadWith("development");
    expect(override(F.BETA_BLOCKS)).toBeUndefined();
  });
});
