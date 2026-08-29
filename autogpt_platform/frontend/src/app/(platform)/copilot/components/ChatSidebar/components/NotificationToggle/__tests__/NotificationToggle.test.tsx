import {
  cleanup,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../../../store";
import { NotificationToggle } from "../NotificationToggle";

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

function clickTrigger() {
  const triggers = screen.getAllByLabelText("Notification settings");
  fireEvent.click(triggers[0]);
}

describe("NotificationToggle", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetStore();
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission: vi.fn() },
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("renders notification settings button", () => {
    render(<NotificationToggle />);

    expect(
      screen.getAllByLabelText("Notification settings").length,
    ).toBeGreaterThan(0);
  });

  it("shows popover with toggles when clicked", async () => {
    render(<NotificationToggle />);

    clickTrigger();

    expect(await screen.findByText("Notifications")).toBeDefined();
    expect(screen.getByText("Sound")).toBeDefined();
  });

  it("enables notifications when toggled on with granted permission", async () => {
    const requestPermission = vi.fn().mockResolvedValue("granted");
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "default", requestPermission },
      configurable: true,
      writable: true,
    });

    render(<NotificationToggle />);

    clickTrigger();

    const notifSwitch = await screen.findByRole("switch", {
      name: /notifications/i,
    });
    fireEvent.click(notifSwitch);

    await waitFor(() => {
      expect(requestPermission).toHaveBeenCalled();
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
    });
  });

  it("disables notifications when toggled off", async () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: true });

    render(<NotificationToggle />);

    clickTrigger();

    const notifSwitch = await screen.findByRole("switch", {
      name: /notifications/i,
    });
    fireEvent.click(notifSwitch);

    await waitFor(() => {
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
    });
  });

  it("disables sound switch when notifications are off", async () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: false });

    render(<NotificationToggle />);

    clickTrigger();

    const soundSwitch = await screen.findByRole("switch", {
      name: /sound/i,
    });
    expect(
      soundSwitch.hasAttribute("disabled") ||
        soundSwitch.getAttribute("data-disabled") !== null,
    ).toBe(true);
  });

  it("toggles sound when clicked", async () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: true });

    render(<NotificationToggle />);

    clickTrigger();

    const soundSwitch = await screen.findByRole("switch", {
      name: /sound/i,
    });
    fireEvent.click(soundSwitch);

    await waitFor(() => {
      expect(useCopilotUIStore.getState().isSoundEnabled).toBe(false);
    });
  });
});
