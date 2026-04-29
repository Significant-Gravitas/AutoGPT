import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useReportClientUrl } from "../useReportClientUrl";

const mockPathname = vi.fn(() => "/copilot");
const mockSearchParams = vi.fn(() => new URLSearchParams("sessionId=A"));

vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname(),
  useSearchParams: () => mockSearchParams(),
}));

const mockPostMessage = vi.fn();
const swListeners: Record<string, ((e?: unknown) => void) | undefined> = {};

beforeEach(() => {
  vi.clearAllMocks();
  mockPathname.mockReturnValue("/copilot");
  mockSearchParams.mockReturnValue(new URLSearchParams("sessionId=A"));
  Object.defineProperty(navigator, "serviceWorker", {
    value: {
      controller: { postMessage: mockPostMessage },
      addEventListener: (name: string, fn: () => void) => {
        swListeners[name] = fn;
      },
      removeEventListener: (name: string) => {
        swListeners[name] = undefined;
      },
    },
    configurable: true,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useReportClientUrl", () => {
  it("posts the current URL on mount", () => {
    renderHook(() => useReportClientUrl());
    expect(mockPostMessage).toHaveBeenCalledWith({
      type: "CLIENT_URL",
      url: "/copilot?sessionId=A",
    });
  });

  it("includes the pathname only when there are no query params", () => {
    mockSearchParams.mockReturnValue(new URLSearchParams());
    renderHook(() => useReportClientUrl());
    expect(mockPostMessage).toHaveBeenCalledWith({
      type: "CLIENT_URL",
      url: "/copilot",
    });
  });

  it("re-posts when pathname changes", () => {
    const { rerender } = renderHook(() => useReportClientUrl());
    expect(mockPostMessage).toHaveBeenLastCalledWith({
      type: "CLIENT_URL",
      url: "/copilot?sessionId=A",
    });

    mockPathname.mockReturnValue("/library");
    mockSearchParams.mockReturnValue(new URLSearchParams());
    rerender();

    expect(mockPostMessage).toHaveBeenLastCalledWith({
      type: "CLIENT_URL",
      url: "/library",
    });
  });

  it("re-posts when the SW controller changes", async () => {
    renderHook(() => useReportClientUrl());
    mockPostMessage.mockClear();

    swListeners.controllerchange?.();

    await waitFor(() => {
      expect(mockPostMessage).toHaveBeenCalledWith({
        type: "CLIENT_URL",
        url: "/copilot?sessionId=A",
      });
    });
  });

  it("does nothing when serviceWorker API is unavailable", () => {
    Object.defineProperty(navigator, "serviceWorker", {
      value: undefined,
      configurable: true,
    });
    renderHook(() => useReportClientUrl());
    expect(mockPostMessage).not.toHaveBeenCalled();
  });
});
