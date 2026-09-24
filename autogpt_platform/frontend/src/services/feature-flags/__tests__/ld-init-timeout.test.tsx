import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { LD_INIT_TIMEOUT_SECONDS } from "../constants";
import { LaunchDarklyProvider } from "../feature-flag-provider";
import { Flag, useGetFlag } from "../use-get-flag";

// launchdarkly-js-sdk-common warns above this many SECONDS ("We recommend a
// timeout of 5 seconds or less") and instrumentation-client.ts turns
// console.warn into a Sentry event, so a millisecond value in the `timeout`
// prop — which the React SDK hands straight to `waitForInitialization` —
// floods Sentry on every page load and never really times out.
const LD_HIGH_TIMEOUT_THRESHOLD_SECONDS = 5;

const ld = vi.hoisted(() => ({
  timeouts: [] as unknown[],
  options: [] as unknown[],
}));

interface Props {
  timeout?: number;
  options?: unknown;
  children: ReactNode;
}

vi.mock("launchdarkly-react-client-sdk", () => {
  // Declared inside the factory: vi.mock is hoisted above module scope.
  function MockLDProvider({ timeout, options, children }: Props) {
    ld.timeouts.push(timeout);
    ld.options.push(options);
    return <>{children}</>;
  }
  return {
    LDProvider: MockLDProvider,
    // LaunchDarkly has not answered for any key yet: init lag, or an outage
    // that will end in a timeout.
    useFlags: () => ({}),
    useLDClient: () => undefined,
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false }),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    areFeatureFlagsEnabled: () => true,
    getLaunchDarklyClientId: () => "test-client-id",
  },
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

function FlagProbe() {
  const enabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  return <span>payment: {String(enabled)}</span>;
}

describe("LaunchDarkly initialisation timeout", () => {
  it("gives the SDK a timeout in seconds, below the threshold it warns about", () => {
    render(
      <LaunchDarklyProvider>
        <FlagProbe />
      </LaunchDarklyProvider>,
    );

    expect(ld.timeouts).toEqual([LD_INIT_TIMEOUT_SECONDS]);
    expect(LD_INIT_TIMEOUT_SECONDS).toBeLessThanOrEqual(
      LD_HIGH_TIMEOUT_THRESHOLD_SECONDS,
    );
  });

  it("renders children on the default flag values while LaunchDarkly has not answered", () => {
    render(
      <LaunchDarklyProvider>
        <FlagProbe />
      </LaunchDarklyProvider>,
    );

    expect(screen.getByText("payment: false")).toBeDefined();
  });

  it("wires no LaunchDarkly-only Sentry inspector; the flag seam records for every vendor", () => {
    render(
      <LaunchDarklyProvider>
        <FlagProbe />
      </LaunchDarklyProvider>,
    );

    expect(ld.options.at(-1)).toBeUndefined();
  });
});
