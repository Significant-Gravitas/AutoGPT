import { afterEach, describe, expect, it, vi } from "vitest";
import {
  readPermission,
  readServerPermission,
  subscribeToPermission,
} from "../helpers";

describe("notification permission helpers", () => {
  afterEach(() => {
    delete (globalThis as { Notification?: unknown }).Notification;
  });

  it("reads the browser's permission", () => {
    Object.defineProperty(globalThis, "Notification", {
      value: { permission: "denied" },
      configurable: true,
      writable: true,
    });
    expect(readPermission()).toBe("denied");
  });

  it("reports a browser without the Notification API as unsupported", () => {
    expect(readPermission()).toBe("unsupported");
  });

  it("renders the neutral state on the server", () => {
    expect(readServerPermission()).toBe("default");
  });

  it("re-reads on focus and visibility change until unsubscribed", () => {
    const onChange = vi.fn();
    const unsubscribe = subscribeToPermission(onChange);

    window.dispatchEvent(new Event("focus"));
    document.dispatchEvent(new Event("visibilitychange"));
    expect(onChange).toHaveBeenCalledTimes(2);

    unsubscribe();
    window.dispatchEvent(new Event("focus"));
    document.dispatchEvent(new Event("visibilitychange"));
    expect(onChange).toHaveBeenCalledTimes(2);
  });
});
