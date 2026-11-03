import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetOauthGetOauthAppInfoMockHandler200,
  getGetOauthGetOauthAppInfoMockHandler400,
  getPostOauthDenyAuthorizationMockHandler200,
} from "@/app/api/__generated__/endpoints/oauth/oauth.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
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
const realLocation = Object.getOwnPropertyDescriptor(window, "location")!;

function openWith(redirectURI: string, clientID = "client-1") {
  mockUseSearchParams.mockReturnValue(
    new URLSearchParams({
      client_id: clientID,
      redirect_uri: redirectURI,
      scope: "IDENTITY",
      state: "state-1",
      code_challenge: "challenge",
    }),
  );
}

function serveAppInfo() {
  server.use(
    getGetOauthGetOauthAppInfoMockHandler200({
      name: "Example App",
      scopes: ["IDENTITY"],
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

afterEach(() => {
  Object.defineProperty(window, "location", realLocation);
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

  test("denying sends the user only where the backend's denial points", async () => {
    let denial: unknown;
    const backendRedirect = `${REGISTERED}?error=access_denied&state=state-1`;
    openWith(REGISTERED);
    serveAppInfo();
    server.use(
      getPostOauthDenyAuthorizationMockHandler200(async ({ request }) => {
        denial = await request.json();
        return { redirect_url: backendRedirect };
      }),
    );

    render(<AuthorizePage />);
    fireEvent.click(await screen.findByRole("button", { name: "Deny" }));

    await waitFor(() => expect(window.location.href).toBe(backendRedirect));
    expect(denial).toEqual({
      client_id: "client-1",
      redirect_uri: REGISTERED,
      state: "state-1",
    });
  });

  test("stays on the page when the backend refuses the denial", async () => {
    openWith(REGISTERED);
    serveAppInfo();
    server.use(
      http.post("/api/proxy/api/oauth/authorize/deny", () =>
        HttpResponse.json({ detail: "Invalid redirect_uri" }, { status: 400 }),
      ),
    );

    render(<AuthorizePage />);
    fireEvent.click(await screen.findByRole("button", { name: "Deny" }));

    expect(await screen.findByText("Invalid redirect_uri")).toBeDefined();
    expect(window.location.href).toBe(PAGE);
  });

  test("shows why returning to the app failed from the invalid-scopes screen", async () => {
    mockUseSearchParams.mockReturnValue(
      new URLSearchParams({
        client_id: "client-1",
        redirect_uri: REGISTERED,
        scope: "EXECUTE_GRAPH",
        state: "state-1",
        code_challenge: "challenge",
      }),
    );
    serveAppInfo();
    server.use(
      http.post("/api/proxy/api/oauth/authorize/deny", () =>
        HttpResponse.json({ detail: "Invalid redirect_uri" }, { status: 400 }),
      ),
    );

    render(<AuthorizePage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Return to Application" }),
    );

    expect(await screen.findByText("Invalid redirect_uri")).toBeDefined();
    expect(window.location.href).toBe(PAGE);
  });

  test("encodes the client id in the app-info request", async () => {
    let requested: URL | undefined;
    openWith(REGISTERED, "client#1");
    server.use(
      getGetOauthGetOauthAppInfoMockHandler200(({ request }) => {
        requested = new URL(request.url);
        return { name: "Example App", scopes: ["IDENTITY"] };
      }),
    );

    render(<AuthorizePage />);
    await screen.findByRole("button", { name: "Deny" });

    expect(requested?.pathname).toBe("/api/proxy/api/oauth/app/client%231");
    expect(requested?.searchParams.get("redirect_uri")).toBe(REGISTERED);
  });
});
