import { describe, expect, it, beforeEach, vi } from "vitest";
import { useCopilotUIStore } from "../store";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
  },
}));

describe("useCopilotUIStore", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useCopilotUIStore.setState({
      initialPrompt: null,
      sessionToDelete: null,
      isDrawerOpen: false,
      completedSessionIDs: new Set<string>(),
      isNotificationsEnabled: false,
      isSoundEnabled: true,
      showNotificationDialog: false,
      copilotMode: "extended_thinking",
    });
  });

  describe("initialPrompt", () => {
    it("starts as null", () => {
      expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
    });

    it("sets and clears prompt", () => {
      useCopilotUIStore.getState().setInitialPrompt("Hello");
      expect(useCopilotUIStore.getState().initialPrompt).toBe("Hello");

      useCopilotUIStore.getState().setInitialPrompt(null);
      expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
    });
  });

  describe("sessionToDelete", () => {
    it("starts as null", () => {
      expect(useCopilotUIStore.getState().sessionToDelete).toBeNull();
    });

    it("sets and clears a delete target", () => {
      useCopilotUIStore
        .getState()
        .setSessionToDelete({ id: "abc", title: "Test" });
      expect(useCopilotUIStore.getState().sessionToDelete).toEqual({
        id: "abc",
        title: "Test",
      });

      useCopilotUIStore.getState().setSessionToDelete(null);
      expect(useCopilotUIStore.getState().sessionToDelete).toBeNull();
    });
  });

  describe("drawer", () => {
    it("starts closed", () => {
      expect(useCopilotUIStore.getState().isDrawerOpen).toBe(false);
    });

    it("opens and closes", () => {
      useCopilotUIStore.getState().setDrawerOpen(true);
      expect(useCopilotUIStore.getState().isDrawerOpen).toBe(true);

      useCopilotUIStore.getState().setDrawerOpen(false);
      expect(useCopilotUIStore.getState().isDrawerOpen).toBe(false);
    });
  });

  describe("completedSessionIDs", () => {
    it("starts empty", () => {
      expect(useCopilotUIStore.getState().completedSessionIDs.size).toBe(0);
    });

    it("adds a completed session", () => {
      useCopilotUIStore.getState().addCompletedSession("s1");
      expect(useCopilotUIStore.getState().completedSessionIDs.has("s1")).toBe(
        true,
      );
    });

    it("persists added sessions to localStorage", () => {
      useCopilotUIStore.getState().addCompletedSession("s1");
      useCopilotUIStore.getState().addCompletedSession("s2");
      const raw = window.localStorage.getItem("copilot-completed-sessions");
      expect(raw).not.toBeNull();
      const parsed = JSON.parse(raw!) as string[];
      expect(parsed).toContain("s1");
      expect(parsed).toContain("s2");
    });

    it("clears a single completed session", () => {
      useCopilotUIStore.getState().addCompletedSession("s1");
      useCopilotUIStore.getState().addCompletedSession("s2");
      useCopilotUIStore.getState().clearCompletedSession("s1");
      expect(useCopilotUIStore.getState().completedSessionIDs.has("s1")).toBe(
        false,
      );
      expect(useCopilotUIStore.getState().completedSessionIDs.has("s2")).toBe(
        true,
      );
    });

    it("updates localStorage when a session is cleared", () => {
      useCopilotUIStore.getState().addCompletedSession("s1");
      useCopilotUIStore.getState().addCompletedSession("s2");
      useCopilotUIStore.getState().clearCompletedSession("s1");
      const raw = window.localStorage.getItem("copilot-completed-sessions");
      const parsed = JSON.parse(raw!) as string[];
      expect(parsed).not.toContain("s1");
      expect(parsed).toContain("s2");
    });

    it("clears all completed sessions", () => {
      useCopilotUIStore.getState().addCompletedSession("s1");
      useCopilotUIStore.getState().addCompletedSession("s2");
      useCopilotUIStore.getState().clearAllCompletedSessions();
      expect(useCopilotUIStore.getState().completedSessionIDs.size).toBe(0);
    });

    it("removes localStorage key when all sessions are cleared", () => {
      useCopilotUIStore.getState().addCompletedSession("s1");
      useCopilotUIStore.getState().clearAllCompletedSessions();
      expect(
        window.localStorage.getItem("copilot-completed-sessions"),
      ).toBeNull();
    });
  });

  describe("sound toggle", () => {
    it("starts enabled", () => {
      expect(useCopilotUIStore.getState().isSoundEnabled).toBe(true);
    });

    it("toggles sound off and on", () => {
      useCopilotUIStore.getState().toggleSound();
      expect(useCopilotUIStore.getState().isSoundEnabled).toBe(false);

      useCopilotUIStore.getState().toggleSound();
      expect(useCopilotUIStore.getState().isSoundEnabled).toBe(true);
    });

    it("persists to localStorage", () => {
      useCopilotUIStore.getState().toggleSound();
      expect(window.localStorage.getItem("copilot-sound-enabled")).toBe(
        "false",
      );
    });
  });

  describe("copilotMode", () => {
    it("defaults to extended_thinking", () => {
      expect(useCopilotUIStore.getState().copilotMode).toBe(
        "extended_thinking",
      );
    });

    it("sets mode to fast", () => {
      useCopilotUIStore.getState().setCopilotMode("fast");
      expect(useCopilotUIStore.getState().copilotMode).toBe("fast");
      expect(window.localStorage.getItem("copilot-mode")).toBe("fast");
    });

    it("sets mode back to extended_thinking", () => {
      useCopilotUIStore.getState().setCopilotMode("fast");
      useCopilotUIStore.getState().setCopilotMode("extended_thinking");
      expect(useCopilotUIStore.getState().copilotMode).toBe(
        "extended_thinking",
      );
    });
  });

  describe("clearCopilotLocalData", () => {
    it("resets state and clears localStorage keys", () => {
      useCopilotUIStore.getState().setCopilotMode("fast");
      useCopilotUIStore.getState().setNotificationsEnabled(true);
      useCopilotUIStore.getState().toggleSound();
      useCopilotUIStore.getState().addCompletedSession("s1");

      useCopilotUIStore.getState().clearCopilotLocalData();

      const state = useCopilotUIStore.getState();
      expect(state.copilotMode).toBe("extended_thinking");
      expect(state.isNotificationsEnabled).toBe(false);
      expect(state.isSoundEnabled).toBe(true);
      expect(state.completedSessionIDs.size).toBe(0);
      expect(window.localStorage.getItem("copilot-mode")).toBeNull();
      expect(
        window.localStorage.getItem("copilot-notifications-enabled"),
      ).toBeNull();
      expect(window.localStorage.getItem("copilot-sound-enabled")).toBeNull();
      expect(
        window.localStorage.getItem("copilot-completed-sessions"),
      ).toBeNull();
    });
  });

  describe("notifications", () => {
    it("sets notification preference", () => {
      useCopilotUIStore.getState().setNotificationsEnabled(true);
      expect(useCopilotUIStore.getState().isNotificationsEnabled).toBe(true);
      expect(window.localStorage.getItem("copilot-notifications-enabled")).toBe(
        "true",
      );
    });

    it("shows and hides notification dialog", () => {
      useCopilotUIStore.getState().setShowNotificationDialog(true);
      expect(useCopilotUIStore.getState().showNotificationDialog).toBe(true);

      useCopilotUIStore.getState().setShowNotificationDialog(false);
      expect(useCopilotUIStore.getState().showNotificationDialog).toBe(false);
    });
  });
});
