import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { Key, storage } from "@/services/storage/local-storage";
import { act, renderHook } from "@testing-library/react";
import { afterAll, afterEach, beforeEach, describe, expect, it } from "vitest";
import { useSyncNotificationsEnabled } from "../useSyncNotificationsEnabled";

const originalNotificationDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  "Notification",
);

function setPermission(permission: NotificationPermission) {
  Object.defineProperty(globalThis, "Notification", {
    value: { permission },
    configurable: true,
    writable: true,
  });
}

function storeValueFromAnotherTab(value: string | null) {
  if (value === null) storage.clean(Key.COPILOT_NOTIFICATIONS_ENABLED);
  else storage.set(Key.COPILOT_NOTIFICATIONS_ENABLED, value);
  act(() => {
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: Key.COPILOT_NOTIFICATIONS_ENABLED,
        newValue: value,
      }),
    );
  });
}

describe("useSyncNotificationsEnabled", () => {
  beforeEach(() => {
    setPermission("granted");
    storage.clean(Key.COPILOT_NOTIFICATIONS_ENABLED);
  });

  afterEach(() => {
    storage.clean(Key.COPILOT_NOTIFICATIONS_ENABLED);
    useCopilotUIStore.setState({ isNotificationsEnabled: false });
  });

  afterAll(() => {
    if (originalNotificationDescriptor) {
      Object.defineProperty(
        globalThis,
        "Notification",
        originalNotificationDescriptor,
      );
    } else {
      delete (globalThis as Record<string, unknown>).Notification;
    }
  });

  it("turns notifications off when another tab switches them off", () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    renderHook(() => useSyncNotificationsEnabled());

    storeValueFromAnotherTab("false");

    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
  });

  it("turns notifications on when another tab switches them on", () => {
    renderHook(() => useSyncNotificationsEnabled());

    storeValueFromAnotherTab("true");

    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
  });

  it("turns notifications off when another tab clears the setting on logout", () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    renderHook(() => useSyncNotificationsEnabled());

    storeValueFromAnotherTab(null);

    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
  });

  it("stays off when switched on elsewhere but this browser has no permission", () => {
    setPermission("denied");
    renderHook(() => useSyncNotificationsEnabled());

    storeValueFromAnotherTab("true");

    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(false);
  });

  it("ignores changes to other settings", () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    renderHook(() => useSyncNotificationsEnabled());

    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: Key.COPILOT_SOUND_ENABLED,
          newValue: "false",
        }),
      );
    });

    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
  });

  it("stops listening once unmounted", () => {
    useCopilotUIStore.setState({ isNotificationsEnabled: true });
    const { unmount } = renderHook(() => useSyncNotificationsEnabled());
    unmount();

    storeValueFromAnotherTab("false");

    expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
  });
});
