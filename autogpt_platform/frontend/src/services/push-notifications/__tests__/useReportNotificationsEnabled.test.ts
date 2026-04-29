import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useReportNotificationsEnabled } from "../useReportNotificationsEnabled";

const mockGetState = vi.fn(() => true);
vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: (
    selector: (s: { isNotificationsEnabled: boolean }) => unknown,
  ) => selector({ isNotificationsEnabled: mockGetState() }),
}));

const mockPostMessage = vi.fn();
const swListeners: Record<string, ((e?: unknown) => void) | undefined> = {};

beforeEach(() => {
  vi.clearAllMocks();
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

describe("useReportNotificationsEnabled", () => {
  it("posts the toggle value on mount when enabled", () => {
    mockGetState.mockReturnValue(true);
    renderHook(() => useReportNotificationsEnabled());
    expect(mockPostMessage).toHaveBeenCalledWith({
      type: "NOTIFICATIONS_ENABLED",
      value: true,
    });
  });

  it("posts the toggle value on mount when disabled", () => {
    mockGetState.mockReturnValue(false);
    renderHook(() => useReportNotificationsEnabled());
    expect(mockPostMessage).toHaveBeenCalledWith({
      type: "NOTIFICATIONS_ENABLED",
      value: false,
    });
  });

  it("re-posts when the toggle flips", () => {
    mockGetState.mockReturnValue(true);
    const { rerender } = renderHook(() => useReportNotificationsEnabled());
    expect(mockPostMessage).toHaveBeenLastCalledWith({
      type: "NOTIFICATIONS_ENABLED",
      value: true,
    });

    mockGetState.mockReturnValue(false);
    rerender();

    expect(mockPostMessage).toHaveBeenLastCalledWith({
      type: "NOTIFICATIONS_ENABLED",
      value: false,
    });
  });

  it("re-posts on SW controllerchange", () => {
    mockGetState.mockReturnValue(false);
    renderHook(() => useReportNotificationsEnabled());
    mockPostMessage.mockClear();

    swListeners.controllerchange?.();

    expect(mockPostMessage).toHaveBeenCalledWith({
      type: "NOTIFICATIONS_ENABLED",
      value: false,
    });
  });

  it("does nothing when serviceWorker API is unavailable", () => {
    Object.defineProperty(navigator, "serviceWorker", {
      value: undefined,
      configurable: true,
    });
    renderHook(() => useReportNotificationsEnabled());
    expect(mockPostMessage).not.toHaveBeenCalled();
  });
});
