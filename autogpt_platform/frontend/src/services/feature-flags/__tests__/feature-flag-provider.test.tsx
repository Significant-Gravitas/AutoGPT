import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LaunchDarklyProvider } from "../feature-flag-provider";
import { Flag, useFlagStatus } from "../use-get-flag";

const ld = vi.hoisted(() => ({
  props: [] as Array<{ context?: unknown; deferInitialization?: boolean }>,
}));

const auth = vi.hoisted(() => ({
  state: { user: null as { id: string } | null, isUserLoading: true },
}));

interface Props {
  context?: unknown;
  deferInitialization?: boolean;
  children: ReactNode;
}

vi.mock("launchdarkly-react-client-sdk", () => {
  function MockLDProvider({ context, deferInitialization, children }: Props) {
    ld.props.push({ context, deferInitialization });
    return <>{children}</>;
  }
  return {
    LDProvider: MockLDProvider,
    // Nothing answered yet, as before the SDK initialises.
    useFlags: () => ({}),
    useLDClient: () => undefined,
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => auth.state,
}));

vi.mock("@/services/environment", () => ({
  environment: {
    areFeatureFlagsEnabled: () => true,
    getLaunchDarklyClientId: () => "test-client-id",
    getCookiebotCBID: () => undefined,
  },
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

function HireProbe() {
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  return (
    <span>
      hire: {String(enabled)} ready: {String(ready)}
    </span>
  );
}

describe("LaunchDarklyProvider while the session loads", () => {
  beforeEach(() => {
    ld.props.length = 0;
    auth.state = { user: null, isUserLoading: true };
  });

  it("renders the page instead of a spinner, so server HTML carries it", () => {
    render(
      <LaunchDarklyProvider>
        <h1>Maria</h1>
      </LaunchDarklyProvider>,
    );

    expect(screen.getByRole("heading", { name: "Maria" })).toBeDefined();
  });

  it("defers LaunchDarkly until the context is known", () => {
    render(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );

    expect(ld.props.at(-1)).toEqual({
      context: undefined,
      deferInitialization: true,
    });
  });

  it("keeps gated UI in the not-answered state rather than off", () => {
    render(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );

    expect(screen.getByText("hire: false ready: false")).toBeDefined();
  });

  it("hands the resolved session to LaunchDarkly without remounting the page", () => {
    const { rerender } = render(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );
    const probe = screen.getByText(/^hire:/);

    auth.state = { user: { id: "user-1" }, isUserLoading: false };
    rerender(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );

    expect(ld.props.at(-1)?.context).toMatchObject({
      user: { key: "user-1", anonymous: false },
    });
    // Same DOM node: the provider stayed mounted across the session resolving.
    expect(screen.getByText(/^hire:/)).toBe(probe);
  });
});
