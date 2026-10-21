import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FeatureFlagProvider } from "../feature-flag-provider";
import { Flag, useFlagStatus } from "../use-get-flag";

const source = vi.hoisted(() => ({
  // What the flag service has answered so far; empty until it does.
  flags: {} as Record<string, unknown>,
}));

const auth = vi.hoisted(() => ({
  state: { user: null as { id: string } | null, isUserLoading: true },
}));

vi.mock("../flag-source", () => ({
  useFlagSource: (key: string) => ({
    value: source.flags[key],
    resolved: key in source.flags,
  }),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => auth.state,
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

function renderProbe() {
  return render(
    <FeatureFlagProvider>
      <HireProbe />
    </FeatureFlagProvider>,
  );
}

function rerenderProbe(rerender: ReturnType<typeof render>["rerender"]) {
  rerender(
    <FeatureFlagProvider>
      <HireProbe />
    </FeatureFlagProvider>,
  );
}

describe("FeatureFlagProvider while the session loads", () => {
  beforeEach(() => {
    source.flags = {};
    auth.state = { user: null, isUserLoading: true };
  });

  it("renders the page instead of a spinner, so server HTML carries it", () => {
    render(
      <FeatureFlagProvider>
        <h1>Maria</h1>
      </FeatureFlagProvider>,
    );

    expect(screen.getByRole("heading", { name: "Maria" })).toBeDefined();
  });

  it("keeps gated UI in the not-answered state rather than off", () => {
    renderProbe();

    expect(screen.getByText("hire: false ready: false")).toBeDefined();
  });

  it("keeps the page mounted while the session resolves", () => {
    const { rerender } = renderProbe();
    const probe = screen.getByText(/^hire:/);

    auth.state = { user: { id: "user-1" }, isUserLoading: false };
    rerenderProbe(rerender);

    expect(screen.getByText(/^hire:/)).toBe(probe);
  });
});

// The route gates (/raise, /team, /home, skills) call notFound() once a flag
// reads ready-but-off. Flags are read for the signed-in user, so the fallback
// timeout must not count down until the session is known.
describe("flag resolution timeout while the session loads", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    source.flags = {};
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

  it("gives the flag service its full timeout once the session resolves", () => {
    const { rerender } = renderProbe();
    act(() => {
      vi.advanceTimersByTime(6_000);
    });

    auth.state = { user: { id: "user-1" }, isUserLoading: false };
    rerenderProbe(rerender);
    act(() => {
      vi.advanceTimersByTime(4_000);
    });
    expect(screen.getByText("hire: false ready: false")).toBeDefined();

    source.flags = { "hire-experts": true };
    rerenderProbe(rerender);

    expect(screen.getByText("hire: true ready: true")).toBeDefined();
  });

  it("still times out once the session is known and nothing answers", () => {
    const { rerender } = renderProbe();

    auth.state = { user: null, isUserLoading: false };
    rerenderProbe(rerender);
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
