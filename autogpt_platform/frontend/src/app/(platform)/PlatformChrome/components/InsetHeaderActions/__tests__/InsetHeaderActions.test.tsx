import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";

import { InsetHeaderActions } from "../InsetHeaderActions";

const mockUseSupabase = vi.fn();
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => mockUseSupabase(),
}));

vi.mock(
  "@/components/layout/Navbar/components/AgentActivityDropdown/AgentActivityDropdown",
  () => ({ AgentActivityDropdown: () => <div data-testid="agent-activity" /> }),
);
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
vi.mock(
  "@/app/(platform)/PlatformChrome/components/UsageIndicator/UsageIndicator",
  () => ({ UsageIndicator: () => <div data-testid="usage-indicator" /> }),
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
  vi.clearAllMocks();
  server.resetHandlers();
});

describe("InsetHeaderActions", () => {
  it("renders nothing when the viewer is logged out", () => {
    mockUseSupabase.mockReturnValue({
      user: null,
      isLoggedIn: false,
      isUserLoading: false,
    });
    const { container } = render(<InsetHeaderActions />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the header actions for a logged-in user", async () => {
    mockUseSupabase.mockReturnValue({
      user: { id: "u1", email: "alice@example.com", role: "user" },
      isLoggedIn: true,
      isUserLoading: false,
    });
    render(<InsetHeaderActions />);

    expect(screen.getByTestId("agent-activity")).toBeDefined();
    expect(screen.getByTestId("usage-indicator")).toBeDefined();
    expect(screen.getByTestId("account-menu")).toBeDefined();
    expect(await screen.findByTestId("wallet")).toBeDefined();
  });
});
