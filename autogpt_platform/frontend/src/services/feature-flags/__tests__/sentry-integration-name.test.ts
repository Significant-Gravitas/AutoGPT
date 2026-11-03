import { featureFlagsIntegration } from "@sentry/nextjs";
import { describe, expect, test } from "vitest";

// The hooks look the integration up by this name; the other flag tests mock
// @sentry/nextjs, so only this one reads it off the real SDK.
describe("Sentry flag recording", () => {
  test("the SDK's feature-flag integration carries the name the hooks look up", () => {
    expect(featureFlagsIntegration().name).toBe("FeatureFlags");
  });
});
