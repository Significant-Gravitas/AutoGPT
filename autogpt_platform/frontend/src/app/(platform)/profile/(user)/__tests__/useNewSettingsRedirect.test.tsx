import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { usePathnameMock, usePlatformChromeMock, replaceMock } = vi.hoisted(
  () => ({
    usePathnameMock: vi.fn(() => "/profile"),
    usePlatformChromeMock: vi.fn(() => ({ isNewLayoutActive: true })),
    replaceMock: vi.fn(),
  }),
);

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

vi.mock("@/app/(platform)/PlatformChrome/usePlatformChrome", () => ({
  usePlatformChrome: usePlatformChromeMock,
}));

import { useNewSettingsRedirect } from "../useNewSettingsRedirect";

describe("useNewSettingsRedirect", () => {
  beforeEach(() => {
    replaceMock.mockClear();
    usePlatformChromeMock.mockReturnValue({ isNewLayoutActive: true });
  });

  const mappings: [string, string][] = [
    ["/profile", "/settings/profile"],
    ["/profile/dashboard", "/settings/creator-dashboard"],
    ["/profile/credits", "/settings/billing"],
    ["/profile/integrations", "/settings/integrations"],
    ["/profile/settings", "/settings/account"],
    ["/profile/api-keys", "/settings/api-keys"],
    ["/profile/oauth-apps", "/settings/oauth-apps"],
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

  it("does not redirect while the classic layout is active", () => {
    usePlatformChromeMock.mockReturnValue({ isNewLayoutActive: false });
    usePathnameMock.mockReturnValue("/profile");

    const { result } = renderHook(() => useNewSettingsRedirect());

    expect(result.current.isRedirecting).toBe(false);
    expect(replaceMock).not.toHaveBeenCalled();
  });
});
