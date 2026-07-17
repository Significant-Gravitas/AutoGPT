import {
  cleanup,
  render,
  screen,
  fireEvent,
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
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });

    render(<NotificationBanner />);

    expect(screen.getByText(/enable browser notifications/i)).toBeDefined();
    expect(screen.getByRole("button", { name: /^enable$/i })).toBeDefined();
  });

  it("does not render when already dismissed", () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });
    window.localStorage.setItem(
      "copilot-notification-banner-dismissed",
      "true",
    );

    const { container } = render(<NotificationBanner />);

    expect(container.innerHTML).toBe("");
  });

  it("does not render when notifications are already enabled", () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "granted", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });
    useCopilotUIStore.setState({ isNotificationsEnabled: true });

    const { container } = render(<NotificationBanner />);

    expect(container.innerHTML).toBe("");
  });

  it("does not render when permission is denied", () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "denied", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });

    const { container } = render(<NotificationBanner />);

    expect(container.innerHTML).toBe("");
  });

  it("requests permission and enables notifications on Enable click", async () => {
    const requestPermission = vi.fn().mockResolvedValue("granted");
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission },
      configurable: true,
      writable: true,
    });

    render(<NotificationBanner />);

    fireEvent.click(screen.getByRole("button", { name: /^enable$/i }));

    await waitFor(() => {
      expect(requestPermission).toHaveBeenCalled();
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
    });
  });

  it("dismisses banner and sets localStorage on dismiss click", () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });

    render(<NotificationBanner />);

    fireEvent.click(screen.getByLabelText("Dismiss"));

    expect(
      window.localStorage.getItem("copilot-notification-banner-dismissed"),
    ).toBe("true");
  });

  it("does not enable notifications when permission is denied", async () => {
    const requestPermission = vi.fn().mockResolvedValue("denied");
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission },
      configurable: true,
      writable: true,
    });

    render(<NotificationBanner />);

    fireEvent.click(screen.getByRole("button", { name: /^enable$/i }));

    await waitFor(() => {
      expect(requestPermission).toHaveBeenCalled();
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
    });
  });
});
