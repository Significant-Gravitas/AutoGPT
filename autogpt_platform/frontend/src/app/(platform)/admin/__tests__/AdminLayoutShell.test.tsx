import type { AnchorHTMLAttributes, ReactNode } from "react";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";

type MockLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  children: ReactNode;
  href: string;
};

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/admin/marketplace",
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

import AdminLayout from "../layout";

describe("AdminLayout shell", () => {
  it("renders the admin shell with its own Back link", () => {
    render(
      <AdminLayout>
        <p>admin page body</p>
      </AdminLayout>,
    );

    expect(screen.getByText("admin page body")).toBeDefined();
    // The shell owns navigation (no app sidebar), so it must expose a way back.
    expect(
      screen.getAllByRole("link", { name: /back to home/i }).length,
    ).toBeGreaterThan(0);
  });

  it("keeps every admin destination reachable", () => {
    render(
      <AdminLayout>
        <p>admin page body</p>
      </AdminLayout>,
    );

    expect(
      screen.getAllByRole("link", { name: /marketplace management/i }).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getAllByRole("link", { name: /admin user management/i }).length,
    ).toBeGreaterThan(0);
  });
});
