import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { NotificationSettingsControls } from "../NotificationSettingsControls";

vi.mock("@sentry/nextjs", () => ({ captureException: vi.fn() }));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
    isClientSide: vi.fn(() => true),
    getAGPTServerApiUrl: vi.fn(() => "http://localhost:8006/api"),
  },
}));

// Radix reflects a disabled switch as `data-disabled`, not always `disabled`.
function isDisabled(element: HTMLElement) {
  return (
    element.hasAttribute("disabled") ||
    element.getAttribute("data-disabled") !== null
  );
}

function setPermission(
  permission: string,
  requestPermission = vi.fn().mockResolvedValue(permission),
) {
  Object.defineProperty(globalThis, "Notification", {
    value: { permission, requestPermission },
    configurable: true,
    writable: true,
  });
  return requestPermission;
}

describe("NotificationSettingsControls", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useCopilotUIStore.setState({
      isNotificationsEnabled: false,
      isSoundEnabled: true,
    });
    setPermission("default");
  });

  afterEach(() => cleanup());

  it("turns notifications back off — the switch is not one-way", async () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    render(<NotificationSettingsControls />);

    await userEvent.click(
      screen.getByRole("switch", { name: "Notifications" }),
    );

    await waitFor(() =>
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false),
    );
  });

  it("asks the browser for permission when switching on", async () => {
    const requestPermission = setPermission("default");
    requestPermission.mockResolvedValue("granted");

    render(<NotificationSettingsControls />);
    await userEvent.click(
      screen.getByRole("switch", { name: "Notifications" }),
    );

    await waitFor(() => {
      expect(requestPermission).toHaveBeenCalled();
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
    });
  });

  it("explains the browser-level block instead of offering a dead switch", async () => {
    setPermission("denied");
    render(<NotificationSettingsControls />);

    await waitFor(() =>
      expect(
        screen.getByText(/browser is blocking notifications/i),
      ).toBeTruthy(),
    );
    expect(
      isDisabled(screen.getByRole("switch", { name: "Notifications" })),
    ).toBe(true);
  });

  it("keeps sound gated behind notifications being on", async () => {
    render(<NotificationSettingsControls />);

    expect(isDisabled(screen.getByRole("switch", { name: "Sound" }))).toBe(
      true,
    );
  });

  it("reports gracefully when the browser has no Notification API", async () => {
    Object.defineProperty(globalThis, "Notification", {
      value: undefined,
      configurable: true,
      writable: true,
    });

    render(<NotificationSettingsControls />);

    await waitFor(() =>
      expect(screen.getByText(/doesn't support notifications/i)).toBeTruthy(),
    );
  });
});
