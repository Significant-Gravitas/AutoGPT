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
  usePathnameMock: vi.fn(() => "/admin/spending"),
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

import { AdminMobileNav } from "../components/AdminMobileNav/AdminMobileNav";
import { adminNavItems } from "../components/AdminSidebar/helpers";

describe("AdminMobileNav", () => {
  beforeEach(() => {
    usePathnameMock.mockReturnValue("/admin/spending");
  });

  it("trigger shows the current page label", () => {
    render(<AdminMobileNav />);

    expect(
      screen.getByRole("button", { name: /admin navigation/i }).textContent,
    ).toContain("User Spending");
  });

  it("falls back to the first nav item when the pathname matches nothing", () => {
    usePathnameMock.mockReturnValue("/admin");
    render(<AdminMobileNav />);

    expect(
      screen.getByRole("button", { name: /admin navigation/i }).textContent,
    ).toContain(adminNavItems[0].label);
  });

  it("opens a popover listing every admin section", async () => {
    render(<AdminMobileNav />);

    fireEvent.click(screen.getByRole("button", { name: /admin navigation/i }));

    for (const { label } of adminNavItems) {
      expect(
        await screen.findByRole("link", { name: new RegExp(label, "i") }),
      ).toBeDefined();
    }
  });

  it("selecting an item closes the popover", async () => {
    render(<AdminMobileNav />);

    fireEvent.click(screen.getByRole("button", { name: /admin navigation/i }));

    const memoryLink = await screen.findByRole("link", {
      name: /memory inspector/i,
    });
    fireEvent.click(memoryLink);

    await waitFor(() => {
      expect(
        screen.queryByRole("link", { name: /memory inspector/i }),
      ).toBeNull();
    });
  });
});
