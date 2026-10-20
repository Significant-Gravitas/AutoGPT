import { describe, expect, it, beforeEach, vi } from "vitest";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
  },
}));

import * as Sentry from "@sentry/nextjs";
import { Key, storage } from "../local-storage";
import { environment } from "@/services/environment";

describe("storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.mocked(Sentry.captureException).mockClear();
    vi.mocked(environment.isServerSide).mockReturnValue(false);
  });

  describe("set and get", () => {
    it("stores and retrieves a value", () => {
      storage.set(Key.COPILOT_MODE, "fast");
      expect(storage.get(Key.COPILOT_MODE)).toBe("fast");
    });

    it("returns null for unset keys", () => {
      expect(storage.get(Key.COPILOT_MODE)).toBeNull();
    });
  });

  describe("clean", () => {
    it("removes a stored value", () => {
      storage.set(Key.COPILOT_SOUND_ENABLED, "true");
      storage.clean(Key.COPILOT_SOUND_ENABLED);
      expect(storage.get(Key.COPILOT_SOUND_ENABLED)).toBeNull();
    });
  });

  describe("null localStorage", () => {
    it("keeps values in memory instead of throwing", () => {
      const original = Object.getOwnPropertyDescriptor(window, "localStorage");
      Object.defineProperty(window, "localStorage", {
        value: null,
        configurable: true,
      });
      try {
        expect(storage.set(Key.COPILOT_MODE, "fast")).toBeUndefined();
        expect(storage.get(Key.COPILOT_MODE)).toBe("fast");
        expect(storage.clean(Key.COPILOT_MODE)).toBeUndefined();
        expect(storage.get(Key.COPILOT_MODE)).toBeNull();
      } finally {
        if (original) {
          Object.defineProperty(window, "localStorage", original);
        }
      }
    });
  });

  describe("unexpected write failures", () => {
    it("reports them to Sentry when storage is available", () => {
      const error = new Error("QuotaExceededError");
      const setItem = vi
        .spyOn(window.localStorage, "setItem")
        .mockImplementation(() => {
          throw error;
        });
      const removeItem = vi
        .spyOn(window.localStorage, "removeItem")
        .mockImplementation(() => {
          throw error;
        });
      try {
        storage.set(Key.COPILOT_MODE, "fast");
        storage.clean(Key.COPILOT_MODE);
        expect(Sentry.captureException).toHaveBeenCalledTimes(2);
        expect(Sentry.captureException).toHaveBeenCalledWith(error);
      } finally {
        setItem.mockRestore();
        removeItem.mockRestore();
      }
    });
  });

  describe("server-side guard", () => {
    // Every route pulls a module in that reads storage while zustand builds
    // its store, so this path runs on every server render. It must be silent.
    it("reads null on the server without reporting to Sentry", () => {
      vi.mocked(environment.isServerSide).mockReturnValue(true);

      expect(storage.get(Key.COPILOT_MODE)).toBeNull();
      expect(Sentry.captureException).not.toHaveBeenCalled();
    });

    it("drops writes on the server without reporting to Sentry", () => {
      vi.mocked(environment.isServerSide).mockReturnValue(true);

      expect(storage.set(Key.COPILOT_MODE, "fast")).toBeUndefined();
      expect(storage.clean(Key.COPILOT_MODE)).toBeUndefined();
      expect(Sentry.captureException).not.toHaveBeenCalled();
    });
  });
});

describe("Key enum", () => {
  it("has expected keys", () => {
    expect(Key.COPILOT_MODE).toBe("copilot-mode");
    expect(Key.COPILOT_SOUND_ENABLED).toBe("copilot-sound-enabled");
    expect(Key.COPILOT_NOTIFICATIONS_ENABLED).toBe(
      "copilot-notifications-enabled",
    );
    expect(Key.CHAT_SESSION_ID).toBe("chat_session_id");
  });
});
