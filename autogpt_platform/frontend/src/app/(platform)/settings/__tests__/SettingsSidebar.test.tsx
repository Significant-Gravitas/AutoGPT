import type { AnchorHTMLAttributes, ReactNode } from "react";
import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  children: ReactNode;
  href: string;
};

const { usePathnameMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(() => "/settings/profile"),
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

import { SettingsSidebar } from "../components/SettingsSidebar/SettingsSidebar";

const expectedItems = [
  { label: "Profile", href: "/settings/profile" },
  { label: "Creator Dashboard", href: "/settings/creator-dashboard" },
  { label: "Billing", href: "/settings/billing" },
  { label: "Integrations", href: "/settings/integrations" },
  { label: "Preferences", href: "/settings/preferences" },
  { label: "AutoGPT API Keys", href: "/settings/api-keys" },
  { label: "OAuth Apps", href: "/settings/oauth-apps" },
];

describe("SettingsSidebar", () => {
  beforeEach(() => {
    usePathnameMock.mockReturnValue("/settings/profile");
  });

  it("renders SETTINGS header and all 7 nav items with correct hrefs", () => {
    render(<SettingsSidebar />);

    expect(screen.getByText("SETTINGS")).toBeDefined();

    for (const { label, href } of expectedItems) {
      const link = screen.getByRole("link", { name: new RegExp(label, "i") });
      expect(link.getAttribute("href")).toBe(href);
    }
  });

  it("marks the nav item matching the current pathname as active", () => {
    usePathnameMock.mockReturnValue("/settings/billing");
    render(<SettingsSidebar />);

    const billing = screen.getByRole("link", { name: /billing/i });
    expect(billing.getAttribute("aria-current")).toBe("page");

    const profile = screen.getByRole("link", { name: /profile/i });
    expect(profile.getAttribute("aria-current")).toBeNull();
  });

  it("treats nested paths under a nav item as active", () => {
    usePathnameMock.mockReturnValue("/settings/integrations/google");
    render(<SettingsSidebar />);

    const integrations = screen.getByRole("link", { name: /integrations/i });
    expect(integrations.getAttribute("aria-current")).toBe("page");
  });
});
