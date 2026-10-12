import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetOauthGetOauthAppInfoMockHandler200,
  getGetOauthGetOauthAppInfoMockHandler400,
} from "@/app/api/__generated__/endpoints/oauth/oauth.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import AuthorizePage from "../page";

const mockUseSearchParams = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  usePathname: () => "/auth/authorize",
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

const PAGE = "http://localhost/auth/authorize";
const REGISTERED = "https://app.example.com/callback";

function openWith(redirectURI: string) {
  mockUseSearchParams.mockReturnValue(
    new URLSearchParams({
      client_id: "client-1",
      redirect_uri: redirectURI,
      scope: "IDENTITY",
      state: "state-1",
      code_challenge: "challenge",
    }),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { href: PAGE },
  });
});

describe("AuthorizePage", () => {
  test("offers no way back to a redirect_uri the app did not register", async () => {
    openWith("https://evil.example/collect");
    server.use(getGetOauthGetOauthAppInfoMockHandler400());

    render(<AuthorizePage />);

    expect(await screen.findByText("Invalid Request")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /return to application|deny/i }),
    ).toBeNull();
    expect(window.location.href).toBe(PAGE);
  });

  test("denying returns the user to the redirect_uri the backend accepted", async () => {
    let requested: URL | undefined;
    openWith(REGISTERED);
    server.use(
      getGetOauthGetOauthAppInfoMockHandler200(({ request }) => {
        requested = new URL(request.url);
        return { name: "Example App", scopes: ["IDENTITY"] };
      }),
    );

    render(<AuthorizePage />);
    fireEvent.click(await screen.findByRole("button", { name: "Deny" }));

    expect(requested?.searchParams.get("redirect_uri")).toBe(REGISTERED);
    const sentTo = new URL(window.location.href);
    expect(`${sentTo.origin}${sentTo.pathname}`).toBe(REGISTERED);
    expect(sentTo.searchParams.get("error")).toBe("access_denied");
    expect(sentTo.searchParams.get("state")).toBe("state-1");
  });
});
