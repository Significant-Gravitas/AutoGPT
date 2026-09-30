import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Flag } from "../use-get-flag";
import { withFeatureFlag } from "../with-feature-flag";

// Reading through the seam is the point of this test: in posthog mode no
// LDProvider is mounted, so a component reading the LaunchDarkly SDK directly
// waits on an empty flag set forever.
const source = vi.hoisted(() => ({
  results: {} as Record<string, { value: unknown; resolved: boolean }>,
}));
const router = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("../flag-source", () => ({
  useFlagSource: (key: string) =>
    source.results[key] ?? { value: undefined, resolved: false },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => router,
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => true },
}));

const FORCE_ALL = "NEXT_PUBLIC_FORCE_ALL_FLAGS";
const PER_FLAG = "NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY";

function Gated() {
  return <div>gated content</div>;
}

const Wrapped = withFeatureFlag(Gated, Flag.GRAPHITI_MEMORY);

describe("withFeatureFlag reads through the flag seam", () => {
  it("renders the page once the vendor answers true", () => {
    source.results = { "graphiti-memory": { value: true, resolved: true } };

    render(<Wrapped />);

    expect(screen.getByText("gated content")).toBeDefined();
    expect(router.push).not.toHaveBeenCalled();
  });

  it("sends the user to 404 once the vendor answers false", () => {
    source.results = { "graphiti-memory": { value: false, resolved: true } };

    render(<Wrapped />);

    expect(screen.queryByText("gated content")).toBeNull();
    expect(router.push).toHaveBeenCalledWith("/404");
  });

  it("keeps waiting, without redirecting, past the resolution timeout", () => {
    // A user who has the flag must not land on /404 because the vendor is
    // slow or ad-blocked; only an answer may move them.
    vi.useFakeTimers();
    try {
      source.results = {};

      render(<Wrapped />);
      act(() => {
        vi.advanceTimersByTime(10_000);
      });

      expect(screen.queryByText("gated content")).toBeNull();
      expect(router.push).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("waits, without redirecting, while the answer is outstanding", () => {
    source.results = {};

    render(<Wrapped />);

    expect(screen.queryByText("gated content")).toBeNull();
    expect(router.push).not.toHaveBeenCalled();
  });
});

// `isForceAllFlags` is read once at module load, so each case sets the env
// first, drops the module cache, and imports the HOC fresh.
async function renderGated() {
  vi.resetModules();
  const { withFeatureFlag: freshHOC } = await import("../with-feature-flag");
  const { Flag: FreshFlag } = await import("../use-get-flag");
  const Page = freshHOC(Gated, FreshFlag.GRAPHITI_MEMORY);
  return render(<Page />);
}

describe("withFeatureFlag env override", () => {
  beforeEach(() => {
    delete process.env[FORCE_ALL];
    delete process.env[PER_FLAG];
    router.push.mockClear();
  });

  afterEach(() => {
    delete process.env[FORCE_ALL];
    delete process.env[PER_FLAG];
  });

  it("renders the wrapped component on first paint when force-all is on", async () => {
    process.env[FORCE_ALL] = "true";

    const { container } = await renderGated();

    expect(screen.getByText("gated content")).toBeTruthy();
    expect(container.querySelector(".animate-spin")).toBeNull();
    expect(router.push).not.toHaveBeenCalled();
  });

  it("redirects to /404 when a per-flag false overrides force-all", async () => {
    process.env[FORCE_ALL] = "true";
    process.env[PER_FLAG] = "false";

    await renderGated();

    await waitFor(() => expect(router.push).toHaveBeenCalledWith("/404"));
    expect(screen.queryByText("gated content")).toBeNull();
  });

  it("keeps the spinner while the vendor has not answered and nothing is forced", async () => {
    const { container } = await renderGated();

    expect(container.querySelector(".animate-spin")).not.toBeNull();
    expect(screen.queryByText("gated content")).toBeNull();
    expect(router.push).not.toHaveBeenCalled();
  });
});

beforeEach(() => {
  source.results = {};
  router.push.mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});
