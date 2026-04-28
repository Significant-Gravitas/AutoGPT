import type { AnchorHTMLAttributes, ReactNode } from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  children: ReactNode;
  href: string;
};

const { usePathnameMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(() => "/settings/billing"),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: usePathnameMock,
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: function MockLink({ children, href, ...props }: MockLinkProps) {
    return (
      <a href={href} {...props}>
        {children}
      </a>
    );
  },
  useLinkStatus: () => ({ pending: false }),
}));

import { SettingsMobileNav } from "../components/SettingsMobileNav/SettingsMobileNav";

describe("SettingsMobileNav", () => {
  beforeEach(() => {
    usePathnameMock.mockReturnValue("/settings/billing");
  });

  it("trigger shows the current page label", () => {
    render(<SettingsMobileNav />);

    const trigger = screen.getByRole("button", {
      name: /settings navigation/i,
    });
    expect(trigger.textContent).toContain("Billing");
  });

  it("opens popover listing all 7 sections on click", async () => {
    render(<SettingsMobileNav />);

    fireEvent.click(
      screen.getByRole("button", { name: /settings navigation/i }),
    );

    const labels = [
      "Profile",
      "Creator Dashboard",
      "Billing",
      "Integrations",
      "Preferences",
      "AutoGPT API Keys",
      "OAuth Apps",
    ];
    for (const label of labels) {
      expect(
        await screen.findByRole("link", { name: new RegExp(label, "i") }),
      ).toBeDefined();
    }
  });

  it("selecting an item closes the popover", async () => {
    render(<SettingsMobileNav />);

    fireEvent.click(
      screen.getByRole("button", { name: /settings navigation/i }),
    );

    const profileLink = await screen.findByRole("link", { name: /profile/i });
    fireEvent.click(profileLink);

    await waitFor(() => {
      expect(screen.queryByRole("link", { name: /oauth apps/i })).toBeNull();
    });
  });
});
