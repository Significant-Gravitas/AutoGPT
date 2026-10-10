import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const GOOD = "https://example.com/callback";
const EVIL = "https://evil.example/phish";

const searchParams = new Map<string, string>([
  ["client_id", "client-1"],
  ["redirect_uri", EVIL],
  ["scope", "EXECUTE_GRAPH"],
  ["state", "csrf-state"],
  ["code_challenge", "challenge"],
  ["code_challenge_method", "S256"],
  ["response_type", "code"],
]);

vi.mock("next/navigation", () => ({
  useSearchParams: () => ({
    get: (key: string) => searchParams.get(key) ?? null,
  }),
}));

const postOauthAuthorizeMock = vi.fn();
const useGetOauthGetOauthAppInfoMock = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/oauth/oauth", () => ({
  postOauthAuthorize: (...args: unknown[]) => postOauthAuthorizeMock(...args),
  useGetOauthGetOauthAppInfo: (...args: unknown[]) =>
    useGetOauthGetOauthAppInfoMock(...args),
}));

import AuthorizePage from "../page";

describe("AuthorizePage open-redirect guards (#15048)", () => {
  let hrefAssignments: string[];

  beforeEach(() => {
    hrefAssignments = [];
    postOauthAuthorizeMock.mockReset();
    useGetOauthGetOauthAppInfoMock.mockReset();

    searchParams.set("redirect_uri", EVIL);
    searchParams.set("response_type", "code");

    useGetOauthGetOauthAppInfoMock.mockReturnValue({
      data: {
        status: 200,
        data: {
          name: "Test App",
          description: "desc",
          logo_url: null,
          scopes: ["EXECUTE_GRAPH"],
          redirect_uris: [GOOD],
        },
      },
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });

    // Capture writes to window.location.href without navigating jsdom
    const original = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        ...original,
        get href() {
          return original.href;
        },
        set href(value: string) {
          hrefAssignments.push(value);
        },
      },
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("refuses an unregistered redirect_uri before rendering consent", async () => {
    render(<AuthorizePage />);

    expect(await screen.findByText("Invalid Request")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Authorize" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Deny" })).toBeNull();
    expect(hrefAssignments).toEqual([]);
  });

  it("does not offer Return to an unregistered redirect_uri on invalid scopes", async () => {
    searchParams.set("scope", "WRITE_GRAPH");

    try {
      render(<AuthorizePage />);

      expect(await screen.findByText("Invalid Request")).toBeTruthy();
      expect(
        screen.queryByRole("button", { name: "Return to Application" }),
      ).toBeNull();
      expect(hrefAssignments).toEqual([]);
    } finally {
      searchParams.set("scope", "EXECUTE_GRAPH");
    }
  });

  it("Approve does not follow an evil backend redirect_url", async () => {
    postOauthAuthorizeMock.mockResolvedValue({
      status: 200,
      data: {
        redirect_url: `${EVIL}?error=unsupported_response_type&state=csrf-state`,
      },
    });

    // Use registered URI in query so Approve reaches the backend call,
    // then backend returns evil redirect_url which must still be rejected.
    searchParams.set("redirect_uri", GOOD);

    render(<AuthorizePage />);

    fireEvent.click(await screen.findByRole("button", { name: "Authorize" }));

    await waitFor(() => {
      expect(postOauthAuthorizeMock).toHaveBeenCalled();
    });

    expect(hrefAssignments).toEqual([]);
    expect(
      await screen.findByText(/unsafe redirect URL rejected/i),
    ).toBeTruthy();
  });

  it("Deny navigates when redirect_uri is registered", async () => {
    searchParams.set("redirect_uri", GOOD);

    render(<AuthorizePage />);

    fireEvent.click(await screen.findByRole("button", { name: "Deny" }));

    expect(hrefAssignments).toHaveLength(1);
    expect(hrefAssignments[0]).toContain(GOOD);
    expect(hrefAssignments[0]).toContain("error=access_denied");
    expect(hrefAssignments[0]).not.toContain(EVIL);
  });
});
