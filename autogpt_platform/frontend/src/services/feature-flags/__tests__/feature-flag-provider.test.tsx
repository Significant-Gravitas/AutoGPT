import { act, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LaunchDarklyProvider } from "../feature-flag-provider";
import { Flag, useFlagStatus } from "../use-get-flag";

const ld = vi.hoisted(() => ({
  props: [] as Array<{ context?: unknown; deferInitialization?: boolean }>,
  // What the SDK has answered so far; empty until it initialises.
  flags: {} as Record<string, unknown>,
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
    useFlags: () => ld.flags,
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
    ld.flags = {};
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

// The route gates (/raise, /team, /home, skills) call notFound() once a flag
// reads ready-but-off. With LaunchDarkly deferred until the session resolves,
// the fallback timeout must not count down while the vendor cannot answer.
describe("flag resolution timeout while LaunchDarkly is deferred", () => {
  function renderProbe() {
    return render(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );
  }

  beforeEach(() => {
    vi.useFakeTimers();
    ld.props.length = 0;
    ld.flags = {};
    auth.state = { user: null, isUserLoading: true };
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not fall back to the default while the session is still loading", () => {
    renderProbe();

    act(() => {
      vi.advanceTimersByTime(10_000);
    });

    expect(screen.getByText("hire: false ready: false")).toBeDefined();
  });

  it("gives LaunchDarkly its full timeout once the session resolves", () => {
    const { rerender } = renderProbe();
    act(() => {
      vi.advanceTimersByTime(6_000);
    });

    auth.state = { user: { id: "user-1" }, isUserLoading: false };
    rerender(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );
    act(() => {
      vi.advanceTimersByTime(4_000);
    });
    expect(screen.getByText("hire: false ready: false")).toBeDefined();

    ld.flags = { "hire-experts": true };
    rerender(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );

    expect(screen.getByText("hire: true ready: true")).toBeDefined();
  });

  it("still times out once initialisation has started and nothing answers", () => {
    const { rerender } = renderProbe();

    auth.state = { user: null, isUserLoading: false };
    rerender(
      <LaunchDarklyProvider>
        <HireProbe />
      </LaunchDarklyProvider>,
    );
    act(() => {
      vi.advanceTimersByTime(4_999);
    });
    expect(screen.getByText("hire: false ready: false")).toBeDefined();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.getByText("hire: false ready: true")).toBeDefined();
  });
});
