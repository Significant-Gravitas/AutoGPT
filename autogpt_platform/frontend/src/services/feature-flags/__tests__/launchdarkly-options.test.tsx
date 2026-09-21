import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const ldProps = vi.fn();

vi.mock("launchdarkly-react-client-sdk", () => ({
  LDProvider: (props: Record<string, unknown>) => {
    ldProps(props);
    return <>{props.children as React.ReactNode}</>;
  },
}));

vi.mock("@sentry/nextjs", () => ({
  buildLaunchDarklyFlagUsedHandler: () => ({}),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false }),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    areFeatureFlagsEnabled: () => true,
    getLaunchDarklyClientId: () => "client-id",
  },
}));

vi.mock("../../analytics/anonymous-id", () => ({
  getAnonymousID: () => "anon",
}));

import { LaunchDarklyProvider } from "../feature-flag-provider";

function optionsOf() {
  render(
    <LaunchDarklyProvider>
      <span />
    </LaunchDarklyProvider>,
  );
  return ldProps.mock.calls.at(-1)?.[0] as Record<string, any>;
}

describe("LaunchDarkly client options", () => {
  it("sends analytics events to a first-party path a tracker blocker does not match", () => {
    expect(optionsOf().options.eventsUrl).toBe("/api/ld-events");
  });

  // The SDK warns on every page load above 5, and reads the value as seconds:
  // the old 5000 meant 83 minutes.
  it("bounds initialisation in seconds, at the SDK's recommended ceiling", () => {
    expect(optionsOf().timeout).toBeLessThanOrEqual(5);
  });

  it("keeps the Sentry flag inspector", () => {
    expect(optionsOf().options.inspectors).toHaveLength(1);
  });
});
