import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

const mocks = vi.hoisted(() => ({
  postAuthorize: vi.fn(),
  postDeny: vi.fn(),
  searchParams: new URLSearchParams(),
  useGetAppInfo: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => mocks.searchParams,
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/app/api/__generated__/endpoints/oauth/oauth", () => ({
  postOauthAuthorize: mocks.postAuthorize,
  postOauthDenyAuthorization: mocks.postDeny,
  useGetOauthGetOauthAppInfo: mocks.useGetAppInfo,
}));

import AuthorizePage from "../page";

const originalLocation = window.location;

function setSearchParams(redirectURI: string) {
  mocks.searchParams = new URLSearchParams({
    client_id: "client-1",
    redirect_uri: redirectURI,
    scope: "USE_TOOLS",
    state: "state-1",
    code_challenge: "a".repeat(43),
    code_challenge_method: "S256",
    response_type: "code",
  });
}

describe("OAuth authorization denial", () => {
  beforeEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: { href: "" },
    });
    mocks.useGetAppInfo.mockReturnValue({
      data: {
        status: 200,
        data: {
          name: "Local Executor",
          description: null,
          logo_url: null,
          scopes: ["USE_TOOLS"],
        },
      },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: originalLocation,
    });
  });

  it("navigates only to the backend-validated denial redirect", async () => {
    setSearchParams("javascript:alert(document.domain)");
    mocks.postDeny.mockResolvedValue({
      status: 200,
      data: {
        redirect_url:
          "https://registered.example/callback?error=access_denied&state=state-1",
      },
    });

    render(<AuthorizePage />);
    fireEvent.click(screen.getByRole("button", { name: "Deny" }));

    await waitFor(() => {
      expect(mocks.postDeny).toHaveBeenCalledWith({
        client_id: "client-1",
        redirect_uri: "javascript:alert(document.domain)",
        state: "state-1",
      });
      expect(window.location.href).toBe(
        "https://registered.example/callback?error=access_denied&state=state-1",
      );
    });
  });

  it("does not navigate when the backend rejects the redirect URI", async () => {
    setSearchParams("javascript:alert(document.domain)");
    mocks.postDeny.mockRejectedValue(new Error("Invalid redirect_uri"));

    render(<AuthorizePage />);
    fireEvent.click(screen.getByRole("button", { name: "Deny" }));

    expect(await screen.findByText("Invalid redirect_uri")).toBeDefined();
    expect(window.location.href).toBe("");
  });
});
