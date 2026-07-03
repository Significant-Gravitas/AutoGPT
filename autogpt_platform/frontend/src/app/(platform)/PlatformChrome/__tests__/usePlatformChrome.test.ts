import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { usePlatformChrome } from "../usePlatformChrome";

const pathnameMock = vi.fn<() => string>(() => "/marketplace");
vi.mock("next/navigation", () => ({
  usePathname: () => pathnameMock(),
}));

const flagMock = vi.fn<(flag: string) => boolean>(() => true);
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flagMock(flag),
  };
});

describe("usePlatformChrome", () => {
  beforeEach(() => {
    pathnameMock.mockReturnValue("/marketplace");
    flagMock.mockReturnValue(true);
  });

  it("enables the new layout after mount when the flag is on and route is allowed", async () => {
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(true);
    });
  });

  it("keeps the classic layout when the flag is off", async () => {
    flagMock.mockReturnValue(false);
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(false);
    });
  });

  it("excludes the /settings route from the new layout", async () => {
    pathnameMock.mockReturnValue("/settings");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      // give the mount effect a chance to run; it should still be false.
      expect(result.current.showNewLayout).toBe(false);
    });
  });

  it("excludes nested /settings/* routes from the new layout", async () => {
    pathnameMock.mockReturnValue("/settings/billing");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(false);
    });
  });

  it("passes the flag enum to useGetFlag", async () => {
    renderHook(() => usePlatformChrome());
    await waitFor(() => {
      expect(flagMock).toHaveBeenCalledWith("autogpt-new-layout");
    });
  });
});
