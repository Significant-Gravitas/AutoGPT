import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const GOOD = "https://client.example/callback";
const EVIL = "https://evil.example/phish";
// The query string builds `javascript:<code>//?error=...`, and the trailing
// query becomes a JS comment, so this runs in the platform's origin.
const JS_URI = "javascript:document.title='pwned:'+location.origin//";

const searchParams = new Map<string, string>();

vi.mock("next/navigation", () => ({
  useSearchParams: () => ({
    get: (key: string) => searchParams.get(key) ?? null,
  }),
}));

const useGetOauthGetOauthAppInfoMock = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/oauth/oauth", () => ({
  useGetOauthGetOauthAppInfo: (...args: unknown[]) =>
    useGetOauthGetOauthAppInfoMock(...args),
}));

// The credentials picker pulls in the whole credential-schema machinery, which
// the redirect guards under test do not touch. Stub it, but keep it able to
// mark a provider as connected so the Continue button can be reached.
vi.mock("@/components/contextual/CredentialsInput/CredentialsInput", () => ({
  CredentialsInput: ({
    onSelectCredentials,
  }: {
    onSelectCredentials: (cred: unknown) => void;
  }) => (
    <button type="button" onClick={() => onSelectCredentials({ id: "cred-1" })}>
      connect
    </button>
  ),
}));

import SetupWizardPage from "../page";

function setParams(overrides: Record<string, string | null>) {
  searchParams.clear();
  searchParams.set("providers", btoa(JSON.stringify([{ provider: "github" }])));
  for (const [key, value] of Object.entries(overrides)) {
    if (value === null) {
      searchParams.delete(key);
    } else {
      searchParams.set(key, value);
    }
  }
}

describe("IntegrationSetupWizardPage redirect guards (#15303)", () => {
  let hrefAssignments: string[];

  beforeEach(() => {
    hrefAssignments = [];
    useGetOauthGetOauthAppInfoMock.mockReset();

    // Capture writes to window.location.href without navigating the test DOM.
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

  it("Cancel does not navigate to a javascript: redirect_uri", async () => {
    setParams({
      client_id: "client-1",
      redirect_uri: JS_URI,
      state: "csrf-state",
    });
    useGetOauthGetOauthAppInfoMock.mockReturnValue({
      data: { name: "Test App", redirect_uris: [GOOD] },
      isLoading: false,
      error: null,
    });

    render(<SetupWizardPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(hrefAssignments).toEqual([]);
    expect(await screen.findByText(/Invalid redirect_uri/)).toBeTruthy();
  });

  it("Cancel does not navigate to a redirect_uri the app did not register", async () => {
    setParams({
      client_id: "client-1",
      redirect_uri: EVIL,
      state: "csrf-state",
    });
    useGetOauthGetOauthAppInfoMock.mockReturnValue({
      data: { name: "Test App", redirect_uris: [GOOD] },
      isLoading: false,
      error: null,
    });

    render(<SetupWizardPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(hrefAssignments).toEqual([]);
    expect(await screen.findByText(/Invalid redirect_uri/)).toBeTruthy();
  });

  it("Continue does not navigate to a redirect_uri the app did not register", async () => {
    setParams({
      client_id: "client-1",
      redirect_uri: EVIL,
      state: "csrf-state",
    });
    useGetOauthGetOauthAppInfoMock.mockReturnValue({
      data: { name: "Test App", redirect_uris: [GOOD] },
      isLoading: false,
      error: null,
    });

    render(<SetupWizardPage />);

    fireEvent.click(await screen.findByRole("button", { name: "connect" }));
    fireEvent.click(await screen.findByRole("button", { name: "Continue" }));

    expect(hrefAssignments).toEqual([]);
    expect(await screen.findByText(/Invalid redirect_uri/)).toBeTruthy();
  });

  it("Cancel navigates to a registered redirect_uri with the error params", async () => {
    setParams({
      client_id: "client-1",
      redirect_uri: GOOD,
      state: "csrf-state",
    });
    useGetOauthGetOauthAppInfoMock.mockReturnValue({
      data: { name: "Test App", redirect_uris: [GOOD] },
      isLoading: false,
      error: null,
    });

    render(<SetupWizardPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(hrefAssignments).toHaveLength(1);
    expect(hrefAssignments[0]).toContain(GOOD);
    expect(hrefAssignments[0]).toContain("error=user_cancelled");
    expect(hrefAssignments[0]).toContain("state=csrf-state");
  });

  it("Continue navigates to a registered redirect_uri with success", async () => {
    setParams({
      client_id: "client-1",
      redirect_uri: GOOD,
      state: "csrf-state",
    });
    useGetOauthGetOauthAppInfoMock.mockReturnValue({
      data: { name: "Test App", redirect_uris: [GOOD] },
      isLoading: false,
      error: null,
    });

    render(<SetupWizardPage />);

    fireEvent.click(await screen.findByRole("button", { name: "connect" }));
    fireEvent.click(await screen.findByRole("button", { name: "Continue" }));

    expect(hrefAssignments).toHaveLength(1);
    expect(hrefAssignments[0]).toContain(GOOD);
    expect(hrefAssignments[0]).toContain("success=true");
  });

  it("keeps a query string the registered redirect_uri already carried", async () => {
    const withQuery = "https://client.example/callback?tenant=1";
    setParams({
      client_id: "client-1",
      redirect_uri: withQuery,
      state: "csrf-state",
    });
    useGetOauthGetOauthAppInfoMock.mockReturnValue({
      data: { name: "Test App", redirect_uris: [withQuery] },
      isLoading: false,
      error: null,
    });

    render(<SetupWizardPage />);

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(hrefAssignments).toHaveLength(1);
    expect(hrefAssignments[0]).toBe(
      "https://client.example/callback?tenant=1&error=user_cancelled&error_description=User+cancelled+the+integration+setup&state=csrf-state",
    );
  });
});
