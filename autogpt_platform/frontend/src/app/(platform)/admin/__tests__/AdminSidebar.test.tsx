import type { AnchorHTMLAttributes, ReactNode } from "react";
import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  children: ReactNode;
  href: string;
};

const { usePathnameMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(() => "/admin/marketplace"),
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

import { AdminSidebar } from "../components/AdminSidebar/AdminSidebar";
import { adminNavItems } from "../components/AdminSidebar/helpers";

describe("AdminSidebar", () => {
  beforeEach(() => {
    usePathnameMock.mockReturnValue("/admin/marketplace");
  });

  it("renders a Back link to /copilot and every admin nav item with its href", () => {
    render(<AdminSidebar />);

    const back = screen.getByRole("link", { name: /back/i });
    expect(back.getAttribute("href")).toBe("/copilot");

    for (const { label, href } of adminNavItems) {
      const link = screen.getByRole("link", { name: new RegExp(label, "i") });
      expect(link.getAttribute("href")).toBe(href);
    }
  });

  it("marks the nav item matching the current pathname as active", () => {
    usePathnameMock.mockReturnValue("/admin/spending");
    render(<AdminSidebar />);

    expect(
      screen
        .getByRole("link", { name: /user spending/i })
        .getAttribute("aria-current"),
    ).toBe("page");
    expect(
      screen
        .getByRole("link", { name: /marketplace management/i })
        .getAttribute("aria-current"),
    ).toBeNull();
  });

  it("treats nested paths under a nav item as active", () => {
    usePathnameMock.mockReturnValue("/admin/memory/entity/123");
    render(<AdminSidebar />);

    expect(
      screen
        .getByRole("link", { name: /memory inspector/i })
        .getAttribute("aria-current"),
    ).toBe("page");
  });

  it("does not mark a nav item active on a partial href match", () => {
    usePathnameMock.mockReturnValue("/admin/bots-legacy");
    render(<AdminSidebar />);

    expect(
      screen
        .getByRole("link", { name: /bot analytics/i })
        .getAttribute("aria-current"),
    ).toBeNull();
  });
});
