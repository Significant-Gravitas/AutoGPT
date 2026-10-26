import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";

import { SidebarUserActions } from "../SidebarUserActions";

const mockUseAuth = vi.fn();
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => mockUseAuth(),
}));

const sidebar = vi.hoisted(() => ({ state: "expanded" }));
vi.mock("@/components/molecules/Sidebar/Sidebar", async (importOriginal) => {
  const actual = await importOriginal<object>();
  return { ...actual, useSidebar: () => sidebar };
});

vi.mock("@/components/layout/Navbar/components/Wallet/Wallet", () => ({
  Wallet: () => <div data-testid="wallet" />,
}));
vi.mock(
  "@/components/layout/Navbar/components/AccountMenu/AccountMenu",
  () => ({
    AccountMenu: ({ userName }: { userName?: string }) => (
      <div data-testid="account-menu">{userName}</div>
    ),
  }),
);

beforeEach(() => {
  server.use(
    http.get("*/api/store/profile", () =>
      HttpResponse.json({
        username: "alice",
        name: "Alice",
        avatar_url: "",
      }),
    ),
  );
});

afterEach(() => {
  sidebar.state = "expanded";
  vi.clearAllMocks();
  server.resetHandlers();
});

describe("SidebarUserActions", () => {
  it("renders nothing when the viewer is logged out", () => {
    mockUseAuth.mockReturnValue({
      user: null,
      isLoggedIn: false,
      isUserLoading: false,
    });
    const { container } = render(<SidebarUserActions />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the user actions row for a logged-in user", async () => {
    mockUseAuth.mockReturnValue({
      user: { id: "u1", email: "alice@example.com", role: "user" },
      isLoggedIn: true,
      isUserLoading: false,
    });
    render(<SidebarUserActions />);

    expect(screen.getByTestId("account-menu")).toBeDefined();
    expect(await screen.findByTestId("wallet")).toBeDefined();
    expect(screen.queryByTestId("agent-activity")).toBeNull();
    expect(screen.queryByTestId("usage-indicator")).toBeNull();
  });

  it("keeps the wallet in the collapsed rail, where new accounts start", async () => {
    // A new account's sidebar starts collapsed to icons. The walkthrough and
    // the cost tip point at the wallet, and people need to find it, so the
    // rail shows it as an icon instead of hiding it.
    sidebar.state = "collapsed";
    mockUseAuth.mockReturnValue({
      user: { id: "u1", email: "alice@example.com", role: "user" },
      isLoggedIn: true,
      isUserLoading: false,
    });
    render(<SidebarUserActions />);

    const wallet = await screen.findByTestId("wallet");
    const holder = wallet.closest('[data-tour-target="wallet"]');
    expect(holder).not.toBeNull();
    expect(holder?.className ?? "").not.toContain("collapsible=icon]:hidden");
  });
});
