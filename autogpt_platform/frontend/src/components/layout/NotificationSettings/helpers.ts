export type NotificationPermissionState =
  | NotificationPermission
  | "unsupported";

export function readPermission(): NotificationPermissionState {
  if (typeof Notification === "undefined") return "unsupported";
  return Notification.permission;
}

// The server can't know, and "default" renders the neutral state.
export function readServerPermission(): NotificationPermissionState {
  return "default";
}

// Site settings change permission without firing anything we can listen to,
// so re-read whenever the tab comes back.
export function subscribeToPermission(onChange: () => void) {
  window.addEventListener("focus", onChange);
  document.addEventListener("visibilitychange", onChange);
  return () => {
    window.removeEventListener("focus", onChange);
    document.removeEventListener("visibilitychange", onChange);
  };
}
