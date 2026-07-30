import type { AnchorHTMLAttributes, ReactNode } from "react";
import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  children: ReactNode;
  href: string;
};

const { usePathnameMock, usePlatformChromeMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(() => "/admin/marketplace"),
  usePlatformChromeMock: vi.fn(() => ({ isNewLayoutActive: false })),
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

vi.mock("@/app/(platform)/PlatformChrome/usePlatformChrome", () => ({
  usePlatformChrome: usePlatformChromeMock,
}));

import AdminLayout from "../layout";

describe("AdminLayout shell switching", () => {
  beforeEach(() => {
    usePathnameMock.mockReturnValue("/admin/marketplace");
    usePlatformChromeMock.mockReturnValue({ isNewLayoutActive: false });
  });

  it("renders the new admin shell with its own Back link when the new layout is active", () => {
    usePlatformChromeMock.mockReturnValue({ isNewLayoutActive: true });

    render(
      <AdminLayout>
        <p>admin page body</p>
      </AdminLayout>,
    );

    expect(screen.getByText("admin page body")).toBeDefined();
    // The new shell owns navigation (no top Navbar), so it must expose a way back.
    expect(
      screen.getAllByRole("link", { name: /back to dashboard/i }).length,
    ).toBeGreaterThan(0);
  });

  it("renders the classic shell without the new admin Back link when the flag is off", () => {
    render(
      <AdminLayout>
        <p>admin page body</p>
      </AdminLayout>,
    );

    expect(screen.getByText("admin page body")).toBeDefined();
    expect(
      screen.queryByRole("link", { name: /back to dashboard/i }),
    ).toBeNull();
  });

  it("keeps every admin destination reachable in the new shell", () => {
    usePlatformChromeMock.mockReturnValue({ isNewLayoutActive: true });

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
