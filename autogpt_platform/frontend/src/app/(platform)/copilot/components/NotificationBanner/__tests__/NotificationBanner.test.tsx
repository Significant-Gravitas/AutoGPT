import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { NotificationBanner } from "../NotificationBanner";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
    isClientSide: vi.fn(() => true),
    getAGPTServerApiUrl: vi.fn(() => "http://localhost:8006/api"),
  },
}));

function stubNotification(permission: NotificationPermission) {
  const requestPermission = vi.fn();
  Object.defineProperty(globalThis, "Notification", {
    value: { permission, requestPermission },
    configurable: true,
    writable: true,
  });
  return requestPermission;
}

function resetStore() {
  useCopilotUIStore.setState({
    isNotificationsEnabled: false,
    isSoundEnabled: true,
  });
}

describe("NotificationBanner", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetStore();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders when permission is default and not dismissed", () => {
    stubNotification("default");

    render(<NotificationBanner />);

    expect(screen.getByText(/notifications are off/i)).toBeDefined();
    expect(screen.getByRole("link", { name: /open settings/i })).toBeDefined();
  });

  it("does not render when already dismissed", () => {
    stubNotification("default");
    window.localStorage.setItem(
      "copilot-notification-banner-dismissed",
      "true",
    );

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

  it("hands off to account settings instead of prompting for permission", () => {
    const requestPermission = stubNotification("default");

    render(<NotificationBanner />);

    const action = screen.getByRole("link", { name: /open settings/i });
    expect(action.getAttribute("href")).toBe("/settings/account");
    expect(screen.queryByRole("button", { name: /^enable$/i })).toBeNull();
    expect(requestPermission).not.toHaveBeenCalled();
  });

  it("dismisses banner and sets localStorage on dismiss click", () => {
    stubNotification("default");

    render(<NotificationBanner />);

    fireEvent.click(screen.getByLabelText("Dismiss"));

    expect(
      window.localStorage.getItem("copilot-notification-banner-dismissed"),
    ).toBe("true");
    expect(screen.queryByText(/notifications are off/i)).toBeNull();
  });

  it("stays mounted when the settings link is clicked", async () => {
    stubNotification("default");

    render(<NotificationBanner />);

    fireEvent.click(screen.getByRole("link", { name: /open settings/i }));

    // Dismissing via state here would unmount the banner — and this link with
    // it — before Next got to navigate, so the click must only persist.
    await waitFor(() =>
      expect(
        window.localStorage.getItem("copilot-notification-banner-dismissed"),
      ).toBe("true"),
    );
    expect(screen.getByRole("link", { name: /open settings/i })).toBeDefined();
  });

  it("stays dismissable when settings is opened in a new tab", () => {
    stubNotification("default");

    render(<NotificationBanner />);

    const link = screen.getByRole("link", { name: /open settings/i });
    fireEvent.click(link, { metaKey: true });
    fireEvent.click(link, { ctrlKey: true });
    fireEvent.click(link, { shiftKey: true });

    expect(
      window.localStorage.getItem("copilot-notification-banner-dismissed"),
    ).toBeNull();
    expect(screen.getByText(/notifications are off/i)).toBeDefined();
  });

  it("hides once permission is granted from another tab", () => {
    stubNotification("default");
    render(<NotificationBanner />);
    expect(screen.getByText(/notifications are off/i)).toBeDefined();

    stubNotification("granted");
    act(() => {
      window.dispatchEvent(new Event("focus"));
    });

    expect(screen.queryByText(/notifications are off/i)).toBeNull();
  });
});
