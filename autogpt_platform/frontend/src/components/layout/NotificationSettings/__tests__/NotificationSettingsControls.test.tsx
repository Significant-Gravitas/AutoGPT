import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { NotificationSettingsControls } from "../NotificationSettingsControls";
import { useNotificationSettings } from "../useNotificationSettings";

vi.mock("@sentry/nextjs", () => ({ captureException: vi.fn() }));

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

// Radix reflects a disabled switch as `data-disabled`, not always `disabled`.
function isDisabled(element: HTMLElement) {
  return (
    element.hasAttribute("disabled") ||
    element.getAttribute("data-disabled") !== null
  );
}

function describedByText(element: HTMLElement) {
  return (element.getAttribute("aria-describedby") ?? "")
    .split(" ")
    .map((id) => document.getElementById(id)?.textContent ?? "")
    .join(" ");
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
    mockToast.mockClear();
    window.localStorage.clear();
    useCopilotUIStore.setState({
      isNotificationsEnabled: false,
      isSoundEnabled: true,
    });
    setPermission("default");
  });

  afterEach(() => cleanup());

  it("turns notifications back off — the switch is not one-way", async () => {
    setPermission("granted");
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
    // A real browser updates `Notification.permission` before resolving.
    const requestPermission = setPermission("default");
    requestPermission.mockImplementation(async () => {
      setPermission("granted");
      return "granted";
    });

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

  it("does not call a dismissed prompt a block", async () => {
    setPermission("default");
    render(<NotificationSettingsControls />);

    await userEvent.click(
      screen.getByRole("switch", { name: "Notifications" }),
    );

    await waitFor(() => expect(mockToast).toHaveBeenCalledTimes(1));
    expect(mockToast.mock.calls[0][0].title).not.toMatch(/blocked/i);
    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
    expect(screen.queryByText(/browser is blocking notifications/i)).toBeNull();
  });

  it("switches the flag off when permission is no longer granted", async () => {
    setPermission("denied");
    useCopilotUIStore.setState({ isNotificationsEnabled: true });

    render(<NotificationSettingsControls />);

    await waitFor(() =>
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false),
    );
  });

  it("picks up a permission change made in site settings when the tab regains focus", async () => {
    setPermission("granted");
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    render(<NotificationSettingsControls />);
    const notifications = screen.getByRole("switch", { name: "Notifications" });
    expect(notifications.getAttribute("aria-checked")).toBe("true");

    setPermission("denied");
    act(() => {
      window.dispatchEvent(new Event("focus"));
    });

    await waitFor(() =>
      expect(
        screen.getByText(/browser is blocking notifications/i),
      ).toBeTruthy(),
    );
    expect(notifications.getAttribute("aria-checked")).toBe("false");
    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
  });

  it("sees a blocked browser as blocked on the very first render", () => {
    setPermission("denied");
    const seen: boolean[] = [];
    function Probe() {
      seen.push(useNotificationSettings().isBlocked);
      return null;
    }

    render(<Probe />);

    // An effect-based read would first render as not blocked, and paint an
    // enabled switch before correcting itself.
    expect(seen[0]).toBe(true);
  });

  it("keeps sound gated behind notifications being on", async () => {
    render(<NotificationSettingsControls />);

    expect(isDisabled(screen.getByRole("switch", { name: "Sound" }))).toBe(
      true,
    );
  });

  it("toggles sound once notifications are on", async () => {
    setPermission("granted");
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    render(<NotificationSettingsControls />);

    await userEvent.click(screen.getByRole("switch", { name: "Sound" }));

    expect(useCopilotUIStore.getState().isSoundEnabled).toBe(false);
  });

  it("toggles from the visible label, not just the switch", async () => {
    setPermission("granted");
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    render(<NotificationSettingsControls />);

    await userEvent.click(screen.getByText("Notifications"));

    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
  });

  it("describes each switch, including why a blocked one can't move", async () => {
    setPermission("denied");
    render(<NotificationSettingsControls />);

    const notifications = screen.getByRole("switch", { name: "Notifications" });
    await waitFor(() =>
      expect(describedByText(notifications)).toMatch(
        /browser is blocking notifications/i,
      ),
    );
    expect(describedByText(notifications)).toMatch(/your experts finish/i);
    expect(
      describedByText(screen.getByRole("switch", { name: "Sound" })),
    ).toMatch(/play a chime/i);
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
