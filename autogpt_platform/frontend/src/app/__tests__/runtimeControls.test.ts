import { describe, expect, it } from "vitest";

import {
  isHostedPlatformOrigin,
  resolveRuntimeControls,
} from "../runtimeControls";

describe("runtime controls", () => {
  it("enables hosted integrations only on the canonical production host", () => {
    expect(
      resolveRuntimeControls({
        publicOrigin: "https://platform.agpt.co",
        isDev: false,
        env: {},
      }),
    ).toMatchObject({
      telemetryEnabled: true,
      feedbackEnabled: true,
      gaMeasurementId: "G-FH2XK2W4GN",
    });

    expect(
      resolveRuntimeControls({
        publicOrigin: "https://agents.example.com",
        isDev: false,
        env: {},
      }),
    ).toMatchObject({
      telemetryEnabled: false,
      feedbackEnabled: false,
      gaMeasurementId: undefined,
    });
  });

  it("keeps self-hosted telemetry and feedback explicitly opt-in", () => {
    expect(
      resolveRuntimeControls({
        publicOrigin: "https://agents.example.com",
        isDev: false,
        env: {
          AUTOGPT_TELEMETRY_ENABLED: "yes",
          AUTOGPT_FEEDBACK_ENABLED: "1",
          AUTOGPT_GA_MEASUREMENT_ID: "G-OPERATOR",
        },
      }),
    ).toMatchObject({
      telemetryEnabled: true,
      feedbackEnabled: true,
      gaMeasurementId: "G-OPERATOR",
    });
  });

  it("uses the project GA property after a self-hosted operator opts in", () => {
    expect(
      resolveRuntimeControls({
        publicOrigin: "https://agents.example.com",
        isDev: false,
        env: { AUTOGPT_TELEMETRY_ENABLED: "true" },
      }).gaMeasurementId,
    ).toBe("G-FH2XK2W4GN");
  });

  it("does not use a baked hosted GA override for self-hosted deployments", () => {
    expect(
      resolveRuntimeControls({
        publicOrigin: "https://agents.example.com",
        isDev: false,
        env: {
          AUTOGPT_TELEMETRY_ENABLED: "true",
          NEXT_PUBLIC_GA_MEASUREMENT_ID: "G-HOSTED-BUILD",
        },
      }).gaMeasurementId,
    ).toBe("G-FH2XK2W4GN");
  });

  it("accepts common truthy values for developer tools", () => {
    expect(
      resolveRuntimeControls({
        publicOrigin: "http://localhost:3000",
        isDev: false,
        env: {
          AUTOGPT_DEVELOPER_UI_ENABLED: "on",
        },
      }).developerUiEnabled,
    ).toBe(true);
  });

  it("trusts only the configured canonical hosted origin", () => {
    expect(isHostedPlatformOrigin("https://platform.agpt.co/")).toBe(true);
    expect(
      isHostedPlatformOrigin("https://platform.agpt.co.evil.example"),
    ).toBe(false);
    expect(isHostedPlatformOrigin("https://user@platform.agpt.co")).toBe(false);
    expect(isHostedPlatformOrigin("https://platform.agpt.co/path")).toBe(false);
    expect(isHostedPlatformOrigin("not a URL")).toBe(false);
  });
});
