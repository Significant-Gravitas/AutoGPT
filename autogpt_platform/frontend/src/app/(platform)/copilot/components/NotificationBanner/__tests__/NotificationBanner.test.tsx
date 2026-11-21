import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getPersistedNotificationBannerDismissed,
  useCopilotUIStore,
} from "../../../store";
import { NotificationBanner } from "../NotificationBanner";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

const mockToast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: mockToast,
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
    isClientSide: vi.fn(() => true),
    getAGPTServerApiUrl: vi.fn(() => "http://localhost:8006/api"),
  },
}));

const BANNER_TEXT = /get a notification when otto or your experts finish/i;
const DISMISSED_KEY = "copilot-notification-banner-dismissed";
const ENABLED_KEY = "copilot-notifications-enabled";

// A real browser updates `Notification.permission` before the request resolves.
function stubNotification(
  permission: NotificationPermission,
  answer: NotificationPermission = permission,
) {
  const value = {
    permission,
    requestPermission: vi.fn(async () => {
      value.permission = answer;
      return answer;
    }),
  };
  Object.defineProperty(globalThis, "Notification", {
    value,
    configurable: true,
    writable: true,
  });
  return value.requestPermission;
}

function removeNotification() {
  Object.defineProperty(globalThis, "Notification", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

function turnOnButton() {
  return screen.getByRole("button", { name: /turn on notifications/i });
}

describe("NotificationBanner", () => {
  beforeEach(() => {
    mockToast.mockClear();
    window.localStorage.clear();
    useCopilotUIStore.setState({
      isNotificationsEnabled: false,
      isNotificationBannerDismissed: false,
      isSoundEnabled: true,
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("offers to turn notifications on when permission is default", () => {
    stubNotification("default");

    render(<NotificationBanner />);

    expect(screen.getByText(BANNER_TEXT)).toBeDefined();
    expect(turnOnButton()).toBeDefined();
  });

  it("turns notifications on in one click once the browser allows it", async () => {
    const requestPermission = stubNotification("default", "granted");
    render(<NotificationBanner />);

    await userEvent.click(turnOnButton());

    expect(requestPermission).toHaveBeenCalledTimes(1);
    await waitFor(() =>
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true),
    );
    expect(window.localStorage.getItem(ENABLED_KEY)).toBe("true");
    expect(screen.queryByText(BANNER_TEXT)).toBeNull();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Notifications are on" }),
    );
  });

  it("hides and explains when the browser blocks notifications", async () => {
    stubNotification("default", "denied");
    render(<NotificationBanner />);

    await userEvent.click(turnOnButton());

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Notifications blocked" }),
      ),
    );
    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
    expect(screen.queryByText(BANNER_TEXT)).toBeNull();
  });

  it("stays up when the permission prompt is dismissed", async () => {
    stubNotification("default", "default");
    render(<NotificationBanner />);

    await userEvent.click(turnOnButton());

    await waitFor(() =>
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Notifications not turned on" }),
      ),
    );
    expect(screen.getByText(BANNER_TEXT)).toBeDefined();
  });

  it("offers one click to turn back on when the browser already allows notifications", async () => {
    stubNotification("granted");
    render(<NotificationBanner />);

    await userEvent.click(turnOnButton());

    await waitFor(() =>
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true),
    );
    expect(screen.queryByText(BANNER_TEXT)).toBeNull();
  });

  it("does not render when notifications are unsupported", () => {
    removeNotification();

    const { container } = render(<NotificationBanner />);

    expect(container.innerHTML).toBe("");
  });

  it("does not render when already dismissed", () => {
    stubNotification("default");
    useCopilotUIStore.setState({ isNotificationBannerDismissed: true });

    const { container } = render(<NotificationBanner />);

    expect(container.innerHTML).toBe("");
  });

  it("does not render when notifications are already enabled", () => {
    stubNotification("granted");
    useCopilotUIStore.setState({ isNotificationsEnabled: true });

    const { container } = render(<NotificationBanner />);

    expect(container.innerHTML).toBe("");
  });

  it("does not render when permission is denied", () => {
    stubNotification("denied");

    const { container } = render(<NotificationBanner />);

    expect(container.innerHTML).toBe("");
  });

  it("hides for good once dismissed", async () => {
    stubNotification("default");
    const { rerender } = render(<NotificationBanner />);

    await userEvent.click(screen.getByLabelText("Dismiss"));

    expect(window.localStorage.getItem(DISMISSED_KEY)).toBe("true");
    expect(screen.queryByText(BANNER_TEXT)).toBeNull();
    rerender(<NotificationBanner />);
    expect(screen.queryByText(BANNER_TEXT)).toBeNull();
  });

  it("hides once notifications are switched on elsewhere", () => {
    stubNotification("granted");
    render(<NotificationBanner />);
    expect(screen.getByText(BANNER_TEXT)).toBeDefined();

    // What the cross-tab sync does when another tab switches them on.
    act(() => {
      useCopilotUIStore.setState({ isNotificationsEnabled: true });
    });

    expect(screen.queryByText(BANNER_TEXT)).toBeNull();
  });
});

describe("getPersistedNotificationBannerDismissed", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("is false for a browser that never chose", () => {
    expect(getPersistedNotificationBannerDismissed()).toBe(false);
  });

  it("is true once the banner was dismissed", () => {
    window.localStorage.setItem(DISMISSED_KEY, "true");
    expect(getPersistedNotificationBannerDismissed()).toBe(true);
  });

  it("treats notifications switched off before as dismissed", () => {
    window.localStorage.setItem(ENABLED_KEY, "false");
    expect(getPersistedNotificationBannerDismissed()).toBe(true);
  });

  it("asks again once copilot data is cleared, which removes the stored switch", () => {
    window.localStorage.setItem(ENABLED_KEY, "false");
    window.localStorage.setItem(DISMISSED_KEY, "true");
    useCopilotUIStore.getState().clearCopilotLocalData();

    expect(getPersistedNotificationBannerDismissed()).toBe(false);
    expect(useCopilotUIStore.getState().isNotificationBannerDismissed).toBe(
      false,
    );
  });
});
