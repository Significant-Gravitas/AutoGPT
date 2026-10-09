import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetOauthGetOauthAppInfoMockHandler200,
  getGetOauthGetOauthAppInfoMockHandler400,
} from "@/app/api/__generated__/endpoints/oauth/oauth.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import IntegrationSetupWizardPage from "../page";

const mockUseSearchParams = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  usePathname: () => "/auth/integrations/setup-wizard",
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

const PAGE = "http://localhost/auth/integrations/setup-wizard";
const REGISTERED = "https://app.example.com/callback";

function openWith(params: Record<string, string>) {
  mockUseSearchParams.mockReturnValue(
    new URLSearchParams({
      client_id: "client-1",
      providers: btoa("{}"),
      redirect_uri: REGISTERED,
      state: "state-1",
      ...params,
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

describe("IntegrationSetupWizardPage", () => {
  test("needs a client_id to check the redirect_uri against", () => {
    mockUseSearchParams.mockReturnValue(
      new URLSearchParams({
        providers: btoa("[]"),
        redirect_uri: REGISTERED,
      }),
    );

    render(<IntegrationSetupWizardPage />);

    expect(
      screen.getByText("Missing required parameters: client_id"),
    ).toBeDefined();
  });

  test("offers no way back to a redirect_uri the app did not register", async () => {
    openWith({ redirect_uri: "https://evil.example/collect" });
    server.use(getGetOauthGetOauthAppInfoMockHandler400());

    render(<IntegrationSetupWizardPage />);

    expect(await screen.findByText("Invalid Request")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Cancel" })).toBeNull();
    expect(window.location.href).toBe(PAGE);
  });

  test("cancelling returns the user to the redirect_uri the backend accepted", async () => {
    let requested: URL | undefined;
    openWith({});
    server.use(
      getGetOauthGetOauthAppInfoMockHandler200(({ request }) => {
        requested = new URL(request.url);
        return { name: "Example App", scopes: ["IDENTITY"] };
      }),
    );

    render(<IntegrationSetupWizardPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(requested?.searchParams.get("redirect_uri")).toBe(REGISTERED);
    const sentTo = new URL(window.location.href);
    expect(`${sentTo.origin}${sentTo.pathname}`).toBe(REGISTERED);
    expect(sentTo.searchParams.get("error")).toBe("user_cancelled");
    expect(sentTo.searchParams.get("state")).toBe("state-1");
  });
});
