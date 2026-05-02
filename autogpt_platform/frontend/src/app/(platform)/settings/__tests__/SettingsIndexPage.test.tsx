import { afterEach, describe, expect, it, vi } from "vitest";

import { cleanup, render, waitFor } from "@/tests/integrations/test-utils";

const { mockReplace } = vi.hoisted(() => ({ mockReplace: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockReplace,
    push: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/settings",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import SettingsIndexPage from "../page";

describe("Settings index page", () => {
  afterEach(() => {
    mockReplace.mockClear();
    cleanup();
  });

  it("renders a polite content skeleton during the client-side redirect", () => {
    const { container } = render(<SettingsIndexPage />);

    const region = container.querySelector('[aria-busy="true"]');
    expect(region).not.toBeNull();
    expect(region?.getAttribute("aria-live")).toBe("polite");
    // Skeleton placeholders for header + sub + 3 sections
    expect(
      container.querySelectorAll('[aria-busy="true"] > div').length,
    ).toBeGreaterThanOrEqual(5);
  });

  it("redirects to /settings/profile via router.replace", async () => {
    render(<SettingsIndexPage />);
    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith("/settings/profile");
    });
  });
});
