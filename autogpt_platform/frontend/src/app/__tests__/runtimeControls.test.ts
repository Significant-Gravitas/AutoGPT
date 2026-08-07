import { describe, expect, it } from "vitest";

import {
  isHostedPlatformHost,
  resolveRuntimeControls,
} from "../runtimeControls";

describe("runtime controls", () => {
  it("enables hosted integrations only on the canonical production host", () => {
    expect(
      resolveRuntimeControls({
        host: "platform.agpt.co:443",
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
        host: "agents.example.com",
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
        host: "agents.example.com",
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

  it("accepts common truthy values for developer tools", () => {
    expect(
      resolveRuntimeControls({
        host: "localhost:3000",
        isDev: false,
        env: {
          AUTOGPT_DEVELOPER_UI_ENABLED: "on",
          NEXT_PUBLIC_REACT_QUERY_DEVTOOL: "yes",
        },
      }).reactQueryDevtoolsEnabled,
    ).toBe(true);
  });

  it("normalizes proxy host lists without matching suffix attacks", () => {
    expect(isHostedPlatformHost("platform.agpt.co, proxy.internal")).toBe(true);
    expect(isHostedPlatformHost("platform.agpt.co.evil.example")).toBe(false);
  });
});
