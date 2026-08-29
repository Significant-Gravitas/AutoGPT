import {
  cleanup,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { NotificationDialog } from "../NotificationDialog";

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
    showNotificationDialog: false,
  });
}

describe("NotificationDialog", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetStore();
  });

  afterEach(() => {
    cleanup();
  });

  it("shows dialog when permission is default and not dismissed", async () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });

    render(<NotificationDialog />);

    expect(await screen.findByText(/AutoPilot can notify you/i)).toBeDefined();
    expect(screen.getByText("Enable notifications")).toBeDefined();
    expect(screen.getByText("Not now")).toBeDefined();
  });

  it("does not show when already dismissed", () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });
    window.localStorage.setItem(
      "copilot-notification-dialog-dismissed",
      "true",
    );

    render(<NotificationDialog />);

    expect(screen.queryByText(/AutoPilot can notify you/i)).toBeNull();
  });

  it("does not show when permission is already granted", () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "granted", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });

    render(<NotificationDialog />);

    expect(screen.queryByText(/AutoPilot can notify you/i)).toBeNull();
  });

  it("requests permission on Enable click and enables notifications", async () => {
    const requestPermission = vi.fn().mockResolvedValue("granted");
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission },
      configurable: true,
      writable: true,
    });

    render(<NotificationDialog />);

    fireEvent.click(await screen.findByText("Enable notifications"));

    await waitFor(() => {
      expect(requestPermission).toHaveBeenCalled();
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
    });
  });

  it("dismisses dialog on Not now click", async () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });

    render(<NotificationDialog />);

    fireEvent.click(await screen.findByText("Not now"));

    await waitFor(() => {
      expect(
        window.localStorage.getItem("copilot-notification-dialog-dismissed"),
      ).toBe("true");
    });
  });

  it("shows dialog when showNotificationDialog is set in store", async () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "denied", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });
    // permission is denied and would not auto-show, but store flag overrides
    useCopilotUIStore.setState({ showNotificationDialog: true });

    render(<NotificationDialog />);

    expect(await screen.findByText(/AutoPilot can notify you/i)).toBeDefined();
  });
});
