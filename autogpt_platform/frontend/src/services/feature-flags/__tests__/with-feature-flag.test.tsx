import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const router = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("launchdarkly-react-client-sdk", () => ({
  useFlags: () => ({}),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => router,
}));

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

vi.mock("@/services/environment", () => ({
  environment: { areFeatureFlagsEnabled: () => false },
}));

const FORCE_ALL = "NEXT_PUBLIC_FORCE_ALL_FLAGS";
const PER_FLAG = "NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY";

function Gated() {
  return <div>gated content</div>;
}

// `isForceAllFlags` is read once at module load, so each case sets the env
// first, drops the module cache, and imports the HOC fresh.
async function renderGated() {
  vi.resetModules();
  const { withFeatureFlag } = await import("../with-feature-flag");
  const { Flag } = await import("../use-get-flag");
  const Page = withFeatureFlag(Gated, Flag.GRAPHITI_MEMORY);
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

  it("keeps the spinner while LaunchDarkly has not answered and nothing is forced", async () => {
    const { container } = await renderGated();

    expect(container.querySelector(".animate-spin")).not.toBeNull();
    expect(screen.queryByText("gated content")).toBeNull();
    expect(router.push).not.toHaveBeenCalled();
  });
});
