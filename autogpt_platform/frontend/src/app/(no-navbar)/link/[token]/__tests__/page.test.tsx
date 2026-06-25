import { beforeEach, describe, expect, test, vi } from "vitest";

import { LinkType } from "@/app/api/__generated__/models/linkType";
import {
  getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler200,
  getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler401,
  getPostPlatformLinkingConfirmAServerLinkTokenUserMustBeAuthenticatedMockHandler200,
  getPostPlatformLinkingConfirmAServerLinkTokenUserMustBeAuthenticatedMockHandler401,
  getPostPlatformLinkingConfirmAUserLinkTokenUserMustBeAuthenticatedMockHandler200,
} from "@/app/api/__generated__/endpoints/platform-linking/platform-linking.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import PlatformLinkPage from "../page";

const mockUseParams = vi.hoisted(() => vi.fn());
const mockUseSearchParams = vi.hoisted(() => vi.fn());
const mockUseSupabase = vi.hoisted(() => vi.fn());
const mockLogOut = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useParams: mockUseParams,
  usePathname: () => "/link/token-123",
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  useSearchParams: mockUseSearchParams,
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

function authenticate() {
  mockUseSupabase.mockReturnValue({
    user: {
      id: "user-1",
      email: "owner@example.com",
      app_metadata: {},
      user_metadata: {},
      aud: "authenticated",
      created_at: "2026-01-01T00:00:00.000Z",
    },
    isLoggedIn: true,
    isUserLoading: false,
    logOut: mockLogOut,
    supabase: {},
  });
}

function setRoute(token = "token-123", platform = "discord") {
  mockUseParams.mockReturnValue({ token });
  mockUseSearchParams.mockReturnValue(new URLSearchParams({ platform }));
}

beforeEach(() => {
  vi.clearAllMocks();
  authenticate();
  setRoute();
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { href: "http://localhost/link/token-123" },
  });
});

describe("PlatformLinkPage", () => {
  test("shows a malformed-token error without fetching link info", () => {
    const infoHandler = vi.fn();
    setRoute("bad.token");
    server.use(
      getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler200(() => {
        infoHandler();
        return {
          platform: "DISCORD",
          link_type: LinkType.SERVER,
          server_name: "Guild",
        };
      }),
    );

    render(<PlatformLinkPage />);

    expect(screen.getByText(/setup link is malformed/i)).toBeDefined();
    expect(infoHandler).not.toHaveBeenCalled();
  });

  test("asks unauthenticated users to sign in without fetching link info", () => {
    const infoHandler = vi.fn();
    mockUseSupabase.mockReturnValue({
      user: null,
      isLoggedIn: false,
      isUserLoading: false,
      logOut: mockLogOut,
      supabase: {},
    });
    server.use(
      getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler200(() => {
        infoHandler();
        return {
          platform: "DISCORD",
          link_type: LinkType.SERVER,
          server_name: "Guild",
        };
      }),
    );

    render(<PlatformLinkPage />);

    expect(
      screen.getByRole("heading", { name: /sign in to continue/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: /^sign in$/i }).getAttribute("href"),
    ).toBe("/login?next=%2Flink%2Ftoken-123");
    expect(infoHandler).not.toHaveBeenCalled();
  });

  test("loads server link details and confirms the server link", async () => {
    server.use(
      getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler200({
        platform: "DISCORD",
        link_type: LinkType.SERVER,
        server_name: "Builders Guild",
      }),
      getPostPlatformLinkingConfirmAServerLinkTokenUserMustBeAuthenticatedMockHandler200(
        {
          success: true,
          link_type: LinkType.SERVER,
          platform: "DISCORD",
          platform_server_id: "server-1",
          server_name: "Builders Guild",
        },
      ),
    );

    render(<PlatformLinkPage />);

    expect(
      await screen.findByRole("heading", {
        name: /set up autopilot for builders guild/i,
      }),
    ).toBeDefined();
    expect(screen.getByText(/signed in as owner@example.com/i)).toBeDefined();

    fireEvent.click(
      screen.getByRole("button", { name: /connect discord to autogpt/i }),
    );

    expect(
      await screen.findByRole("heading", { name: /autopilot is ready/i }),
    ).toBeDefined();
    expect(screen.getByText(/builders guild/i)).toBeDefined();
  });

  test("loads user link details and confirms the user link endpoint", async () => {
    let serverConfirmCalls = 0;
    let userConfirmCalls = 0;
    server.use(
      getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler200({
        platform: "TELEGRAM",
        link_type: LinkType.USER,
        server_name: null,
      }),
      getPostPlatformLinkingConfirmAServerLinkTokenUserMustBeAuthenticatedMockHandler200(
        () => {
          serverConfirmCalls += 1;
          return {
            success: true,
            link_type: LinkType.SERVER,
            platform: "TELEGRAM",
            platform_server_id: "server-1",
            server_name: null,
          };
        },
      ),
      getPostPlatformLinkingConfirmAUserLinkTokenUserMustBeAuthenticatedMockHandler200(
        () => {
          userConfirmCalls += 1;
          return {
            success: true,
            link_type: LinkType.USER,
            platform: "TELEGRAM",
            platform_user_id: "platform-user-1",
          };
        },
      ),
    );

    render(<PlatformLinkPage />);

    expect(
      await screen.findByRole("heading", { name: /link your telegram dms/i }),
    ).toBeDefined();

    fireEvent.click(
      screen.getByRole("button", { name: /connect my telegram dms/i }),
    );

    expect(
      await screen.findByRole("heading", { name: /autopilot is ready/i }),
    ).toBeDefined();
    expect(userConfirmCalls).toBe(1);
    expect(serverConfirmCalls).toBe(0);
  });

  test("shows an expired-link message when info loading fails", async () => {
    server.use(
      getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler401(),
    );

    render(<PlatformLinkPage />);

    expect(
      await screen.findByText(/couldn't load setup details/i),
    ).toBeDefined();
  });

  test("shows backend detail when confirmation fails", async () => {
    server.use(
      getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler200({
        platform: "DISCORD",
        link_type: LinkType.SERVER,
        server_name: "Builders Guild",
      }),
      getPostPlatformLinkingConfirmAServerLinkTokenUserMustBeAuthenticatedMockHandler401(
        {
          detail: "This setup link was already used.",
        },
      ),
    );

    render(<PlatformLinkPage />);

    fireEvent.click(
      await screen.findByRole("button", {
        name: /connect discord to autogpt/i,
      }),
    );

    expect(
      await screen.findByText(/this setup link was already used/i),
    ).toBeDefined();
  });

  test("signs out and redirects back through login when switching account", async () => {
    mockLogOut.mockResolvedValue(undefined);
    server.use(
      getGetPlatformLinkingGetDisplayInfoForALinkTokenMockHandler200({
        platform: "DISCORD",
        link_type: LinkType.SERVER,
        server_name: "Builders Guild",
      }),
    );

    render(<PlatformLinkPage />);

    fireEvent.click(await screen.findByRole("button", { name: /not you/i }));

    await waitFor(() => {
      expect(mockLogOut).toHaveBeenCalledTimes(1);
      expect(window.location.href).toBe("/login?next=%2Flink%2Ftoken-123");
    });
  });
});
