import { beforeEach, describe, expect, it } from "vitest";
import type { ArtifactRef } from "./store";
import { useCopilotUIStore } from "./store";

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

  it("openArtifact ignores non-previewable artifacts", () => {
    const binary = {
      ...makeArtifact("bin", "artifact.bin"),
      mimeType: "application/octet-stream",
    };

    useCopilotUIStore.getState().openArtifact(binary);

    const s = useCopilotUIStore.getState().artifactPanel;
    expect(s.isOpen).toBe(false);
    expect(s.activeArtifact).toBeNull();
    expect(s.history).toEqual([]);
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
