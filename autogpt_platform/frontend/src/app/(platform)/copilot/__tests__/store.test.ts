import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ArtifactRef } from "../store";
import { useCopilotUIStore } from "../store";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
  },
}));

function makeArtifact(id: string, title = `file-${id}`): ArtifactRef {
  return {
    id,
    title,
    mimeType: "text/plain",
    sourceUrl: `/api/proxy/api/workspace/files/${id}/download`,
    origin: "agent",
  };
}

function resetStore() {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      isMinimized: false,
      isMaximized: false,
      width: 600,
      activeArtifact: null,
      history: [],
    },
  });
}

describe("artifactPanel store actions", () => {
  beforeEach(resetStore);

  it("openArtifact opens the panel and sets the active artifact", () => {
    const a = makeArtifact("a");
    useCopilotUIStore.getState().openArtifact(a);
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(true);
    expect(s.isMinimized).toBe(false);
    expect(s.activeArtifact?.id).toBe("a");
    expect(s.history).toEqual([]);
  });

  it("openArtifact pushes the previous artifact onto history", () => {
    const a = makeArtifact("a");
    const b = makeArtifact("b");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().openArtifact(b);
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact?.id).toBe("b");
    expect(s.history.map((h) => h.id)).toEqual(["a"]);
  });

  it("openArtifact does NOT push history when re-opening the same artifact", () => {
    const a = makeArtifact("a");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().openArtifact(a);
    expect(useCopilotUIStore.getState().artifactPanel.history).toEqual([]);
  });

  it("openArtifact pops the top of history when returning to it (A→B→A)", () => {
    const a = makeArtifact("a");
    const b = makeArtifact("b");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().openArtifact(b);
    useCopilotUIStore.getState().openArtifact(a); // ping-pong
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact?.id).toBe("a");
    // History was [a]; returning to a should pop, not push.
    expect(s.history).toEqual([]);
  });

  it("goBackArtifact pops the last entry and becomes active", () => {
    const a = makeArtifact("a");
    const b = makeArtifact("b");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().openArtifact(b);
    useCopilotUIStore.getState().goBackArtifact();
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact?.id).toBe("a");
    expect(s.history).toEqual([]);
  });

  it("goBackArtifact is a no-op when history is empty", () => {
    const a = makeArtifact("a");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().goBackArtifact();
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact?.id).toBe("a");
  });

  it("closeArtifactPanel keeps activeArtifact (for exit animation) and clears history", () => {
    const a = makeArtifact("a");
    const b = makeArtifact("b");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().openArtifact(b);
    useCopilotUIStore.getState().closeArtifactPanel();
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.isMinimized).toBe(false);
    expect(s.activeArtifact?.id).toBe("b");
    expect(s.history).toEqual([]);
  });

  it("openArtifact does not resurrect a previously closed artifact into history", () => {
    const a = makeArtifact("a");
    const b = makeArtifact("b");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().closeArtifactPanel();
    useCopilotUIStore.getState().openArtifact(b);

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(true);
    expect(s.activeArtifact?.id).toBe("b");
    expect(s.history).toEqual([]);
  });

  it("openArtifact opens even non-previewable artifacts", () => {
    const binary = {
      ...makeArtifact("bin", "artifact.bin"),
      mimeType: "application/octet-stream",
    };

    useCopilotUIStore.getState().openArtifact(binary);

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(true);
    expect(s.activeArtifact?.id).toBe("bin");
  });

  it("resetArtifactPanel clears active artifact and history", () => {
    const a = makeArtifact("a");
    const b = makeArtifact("b");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().openArtifact(b);
    useCopilotUIStore.getState().maximizeArtifactPanel();

    useCopilotUIStore.getState().resetArtifactPanel();

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.isMinimized).toBe(false);
    expect(s.isMaximized).toBe(false);
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
  });

  it("minimize/restore toggles isMinimized without touching activeArtifact", () => {
    const a = makeArtifact("a");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().minimizeArtifactPanel();
    expect(useCopilotUIStore.getState().artifactPanel.isMinimized).toBe(true);
    useCopilotUIStore.getState().restoreArtifactPanel();
    expect(useCopilotUIStore.getState().artifactPanel.isMinimized).toBe(false);
    expect(useCopilotUIStore.getState().artifactPanel.activeArtifact?.id).toBe(
      "a",
    );
  });

  it("maximize sets isMaximized and clears isMinimized", () => {
    const a = makeArtifact("a");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().minimizeArtifactPanel();
    useCopilotUIStore.getState().maximizeArtifactPanel();
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isMaximized).toBe(true);
    expect(s.isMinimized).toBe(false);
  });

  it("restoreArtifactPanel clears both isMinimized and isMaximized", () => {
    const a = makeArtifact("a");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().maximizeArtifactPanel();
    useCopilotUIStore.getState().restoreArtifactPanel();
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isMaximized).toBe(false);
    expect(s.isMinimized).toBe(false);
  });

  it("setArtifactPanelWidth updates width and clears isMaximized", () => {
    useCopilotUIStore.getState().maximizeArtifactPanel();
    useCopilotUIStore.getState().setArtifactPanelWidth(720);
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.width).toBe(720);
    expect(s.isMaximized).toBe(false);
  });

  it("history is capped at 25 entries (MAX_HISTORY)", () => {
    // Open 27 artifacts sequentially (A0..A26). History should never exceed 25.
    for (let i = 0; i < 27; i++) {
      useCopilotUIStore.getState().openArtifact(makeArtifact(`a${i}`));
    }
    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.activeArtifact?.id).toBe("a26");
    expect(s.history.length).toBe(25);
    // The oldest entry (a0) should have been dropped; a1 is the earliest surviving.
    expect(s.history[0].id).toBe("a1");
    expect(s.history[24].id).toBe("a25");
  });

  it("clearCopilotLocalData resets artifact panel to default", () => {
    const a = makeArtifact("a");
    const b = makeArtifact("b");
    useCopilotUIStore.getState().openArtifact(a);
    useCopilotUIStore.getState().openArtifact(b);
    useCopilotUIStore.getState().maximizeArtifactPanel();

    useCopilotUIStore.getState().clearCopilotLocalData();

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.isMinimized).toBe(false);
    expect(s.isMaximized).toBe(false);
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
    expect(s.width).toBe(600); // DEFAULT_PANEL_WIDTH
  });
});

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
      copilotChatMode: "extended_thinking",
      copilotLlmModel: "standard",
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

  describe("copilotChatMode", () => {
    it("defaults to extended_thinking", () => {
      expect(useCopilotUIStore.getState().copilotChatMode).toBe(
        "extended_thinking",
      );
    });

    it("sets mode to fast", () => {
      useCopilotUIStore.getState().setCopilotChatMode("fast");
      expect(useCopilotUIStore.getState().copilotChatMode).toBe("fast");
    });

    it("sets mode back to extended_thinking", () => {
      useCopilotUIStore.getState().setCopilotChatMode("fast");
      useCopilotUIStore.getState().setCopilotChatMode("extended_thinking");
      expect(useCopilotUIStore.getState().copilotChatMode).toBe(
        "extended_thinking",
      );
    });

    it("persists mode to localStorage", () => {
      useCopilotUIStore.getState().setCopilotChatMode("fast");
      expect(window.localStorage.getItem("copilot-mode")).toBe("fast");
    });
  });

  describe("copilotLlmModel", () => {
    it("defaults to standard", () => {
      expect(useCopilotUIStore.getState().copilotLlmModel).toBe("standard");
    });

    it("sets model to advanced", () => {
      useCopilotUIStore.getState().setCopilotLlmModel("advanced");
      expect(useCopilotUIStore.getState().copilotLlmModel).toBe("advanced");
    });

    it("persists model to localStorage", () => {
      useCopilotUIStore.getState().setCopilotLlmModel("advanced");
      expect(window.localStorage.getItem("copilot-model")).toBe("advanced");
    });
  });

  describe("clearCopilotLocalData", () => {
    it("resets state and clears localStorage keys", () => {
      useCopilotUIStore.getState().setCopilotChatMode("fast");
      useCopilotUIStore.getState().setCopilotLlmModel("advanced");
      useCopilotUIStore.getState().setNotificationsEnabled(true);
      useCopilotUIStore.getState().toggleSound();
      useCopilotUIStore.getState().addCompletedSession("s1");

      useCopilotUIStore.getState().clearCopilotLocalData();

      const state = useCopilotUIStore.getState();
      expect(state.copilotChatMode).toBe("extended_thinking");
      expect(state.copilotLlmModel).toBe("standard");
      expect(state.isNotificationsEnabled).toBe(false);
      expect(state.isSoundEnabled).toBe(true);
      expect(state.completedSessionIDs.size).toBe(0);
      expect(
        window.localStorage.getItem("copilot-notifications-enabled"),
      ).toBeNull();
      expect(window.localStorage.getItem("copilot-sound-enabled")).toBeNull();
      expect(window.localStorage.getItem("copilot-mode")).toBeNull();
      expect(window.localStorage.getItem("copilot-model")).toBeNull();
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

describe("useCopilotUIStore localStorage initialisation", () => {
  afterEach(() => {
    vi.resetModules();
    window.localStorage.clear();
  });

  it("reads fast chat mode from localStorage on store creation", async () => {
    window.localStorage.setItem("copilot-mode", "fast");
    vi.resetModules();
    const { useCopilotUIStore: fresh } = await import("../store");
    expect(fresh.getState().copilotChatMode).toBe("fast");
  });

  it("reads advanced model from localStorage on store creation", async () => {
    window.localStorage.setItem("copilot-model", "advanced");
    vi.resetModules();
    const { useCopilotUIStore: fresh } = await import("../store");
    expect(fresh.getState().copilotLlmModel).toBe("advanced");
  });
});
