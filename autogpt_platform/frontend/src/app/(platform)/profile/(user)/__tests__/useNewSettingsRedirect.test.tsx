import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { usePathnameMock, replaceMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(() => "/profile"),
  replaceMock: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: replaceMock,
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: usePathnameMock,
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { useNewSettingsRedirect } from "../useNewSettingsRedirect";

function setLocation(url: string) {
  window.history.replaceState({}, "", url);
}

describe("useNewSettingsRedirect", () => {
  beforeEach(() => {
    replaceMock.mockClear();
    setLocation("/");
  });

  const mappings: [string, string][] = [
    ["/profile", "/settings/profile"],
    ["/profile/dashboard", "/settings/creator-dashboard"],
    ["/profile/credits", "/settings/billing"],
    ["/profile/integrations", "/settings/integrations"],
    ["/profile/api-keys", "/settings/api-keys"],
  ];

  it.each(mappings)("redirects %s to %s", (from, to) => {
    usePathnameMock.mockReturnValue(from);

    const { result } = renderHook(() => useNewSettingsRedirect());

    expect(result.current.isRedirecting).toBe(true);
    expect(replaceMock).toHaveBeenCalledWith(to);
  });

  it("falls back to /settings/profile for an unmapped legacy path", () => {
    usePathnameMock.mockReturnValue("/profile/something-else");

    const { result } = renderHook(() => useNewSettingsRedirect());

    expect(result.current.isRedirecting).toBe(true);
    expect(replaceMock).toHaveBeenCalledWith("/settings/profile");
  });

  it("keeps the hash so anchored deep links survive the hop", () => {
    usePathnameMock.mockReturnValue("/profile/dashboard");
    setLocation("/profile/dashboard#submissions");

    renderHook(() => useNewSettingsRedirect());

    expect(replaceMock).toHaveBeenCalledWith(
      "/settings/creator-dashboard#submissions",
    );
  });

  it("keeps the query string so Stripe return params survive the hop", () => {
    usePathnameMock.mockReturnValue("/profile/credits");
    setLocation("/profile/credits?subscription=success");

    renderHook(() => useNewSettingsRedirect());

    expect(replaceMock).toHaveBeenCalledWith(
      "/settings/billing?subscription=success",
    );
  });

  const keptOnLegacy = ["/profile/oauth-apps", "/profile/settings"];

  it.each(keptOnLegacy)(
    "keeps %s on the legacy page while the new one lacks its features",
    (pathname) => {
      usePathnameMock.mockReturnValue(pathname);

      const { result } = renderHook(() => useNewSettingsRedirect());

      expect(result.current.isRedirecting).toBe(false);
      expect(replaceMock).not.toHaveBeenCalled();
    },
  );

  it("keeps the #notifications anchor reachable on the legacy settings page", () => {
    usePathnameMock.mockReturnValue("/profile/settings");
    setLocation("/profile/settings#notifications");

    const { result } = renderHook(() => useNewSettingsRedirect());

    expect(result.current.isRedirecting).toBe(false);
    expect(replaceMock).not.toHaveBeenCalled();
  });

  it("does not redirect paths outside /profile", () => {
    usePathnameMock.mockReturnValue("/marketplace");

    const { result } = renderHook(() => useNewSettingsRedirect());

    expect(result.current.isRedirecting).toBe(false);
    expect(replaceMock).not.toHaveBeenCalled();
  });
});
