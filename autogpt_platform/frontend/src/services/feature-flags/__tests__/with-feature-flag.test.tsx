import { render, screen } from "@testing-library/react";
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

function Guarded() {
  return <div>gated content</div>;
}

const Wrapped = withFeatureFlag(Guarded, Flag.GRAPHITI_MEMORY);

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

  it("waits, without redirecting, while the answer is outstanding", () => {
    source.results = {};

    render(<Wrapped />);

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
