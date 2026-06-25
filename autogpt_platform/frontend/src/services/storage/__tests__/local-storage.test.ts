import { describe, expect, it, beforeEach, vi } from "vitest";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
  },
}));

import { Key, storage } from "../local-storage";
import { environment } from "@/services/environment";

describe("storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
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

  describe("server-side guard", () => {
    it("returns undefined for get when on server side", () => {
      vi.mocked(environment.isServerSide).mockReturnValue(true);
      expect(storage.get(Key.COPILOT_MODE)).toBeUndefined();
    });

    it("returns undefined for set when on server side", () => {
      vi.mocked(environment.isServerSide).mockReturnValue(true);
      expect(storage.set(Key.COPILOT_MODE, "fast")).toBeUndefined();
    });

    it("returns undefined for clean when on server side", () => {
      vi.mocked(environment.isServerSide).mockReturnValue(true);
      expect(storage.clean(Key.COPILOT_MODE)).toBeUndefined();
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
