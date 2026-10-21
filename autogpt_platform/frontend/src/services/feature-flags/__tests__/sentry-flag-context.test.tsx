import { render, renderHook } from "@testing-library/react";
import { Component, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Flag, useFlagStatus, useGetFlag } from "../use-get-flag";

const source = vi.hoisted(() => ({
  flags: {} as Record<string, unknown>,
}));

vi.mock("../flag-source", () => ({
  useFlagSource: (key: string) => ({
    value: source.flags[key],
    resolved: key in source.flags,
  }),
}));

const sentry = vi.hoisted(() => ({
  addFeatureFlag: vi.fn(),
}));

vi.mock("@sentry/nextjs", () => ({
  getClient: () => ({
    getIntegrationByName: (name: string) =>
      name === "FeatureFlags"
        ? { addFeatureFlag: sentry.addFeatureFlag }
        : undefined,
  }),
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

const HIRE_EXPERTS = "hire-experts";

describe("Sentry's flag context", () => {
  it("records the served boolean", () => {
    source.flags = { [HIRE_EXPERTS]: true };

    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));

    expect(sentry.addFeatureFlag).toHaveBeenCalledExactlyOnceWith(
      HIRE_EXPERTS,
      true,
    );
  });

  it("records a flag read by a component that throws in the same render", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    source.flags = { [HIRE_EXPERTS]: true };
    let recordedAtCatch: unknown[][] = [];
    function Reader(): ReactNode {
      useGetFlag(Flag.HIRE_EXPERTS);
      throw new Error("render failed");
    }
    class Boundary extends Component<{ children: ReactNode }> {
      state = { failed: false };
      static getDerivedStateFromError() {
        return { failed: true };
      }
      componentDidCatch() {
        recordedAtCatch = [...sentry.addFeatureFlag.mock.calls];
      }
      render() {
        return this.state.failed ? null : this.props.children;
      }
    }

    render(
      <Boundary>
        <Reader />
      </Boundary>,
    );

    expect(recordedAtCatch).toContainEqual([HIRE_EXPERTS, true]);
  });

  it("records the default it serves while the flag has no answer", () => {
    source.flags = {};

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(false);
    expect(sentry.addFeatureFlag).toHaveBeenCalledExactlyOnceWith(
      HIRE_EXPERTS,
      false,
    );
  });

  it("records nothing for an env-forced flag", () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS = "true";

    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS));
    renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(sentry.addFeatureFlag).not.toHaveBeenCalled();
  });

  it("records nothing for a JSON-valued flag", () => {
    source.flags = { "copilot-bot-platforms": { slack: false } };

    renderHook(() => useGetFlag(Flag.COPILOT_BOT_PLATFORMS));

    expect(sentry.addFeatureFlag).not.toHaveBeenCalled();
  });

  it("still serves the flag when recording throws", () => {
    sentry.addFeatureFlag.mockImplementation(() => {
      throw new Error("sentry down");
    });
    source.flags = { [HIRE_EXPERTS]: true };

    const debug = vi.spyOn(console, "debug").mockImplementation(() => {});

    const { result } = renderHook(() => useGetFlag(Flag.HIRE_EXPERTS));

    expect(result.current).toBe(true);
    expect(debug).toHaveBeenCalledOnce();
  });
});

beforeEach(() => {
  source.flags = {};
  sentry.addFeatureFlag.mockReset();
  Object.keys(process.env)
    .filter((key) => key.startsWith("NEXT_PUBLIC_FORCE_FLAG_"))
    .forEach((key) => delete process.env[key]);
});

afterEach(() => {
  vi.restoreAllMocks();
});
