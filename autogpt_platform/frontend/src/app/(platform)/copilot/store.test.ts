import { beforeEach, describe, expect, it } from "vitest";
import { Key } from "@/services/storage/local-storage";
import { useCopilotUIStore } from "./store";

describe("useCopilotUIStore artifact panel", () => {
  beforeEach(() => {
    window.localStorage.removeItem(Key.COPILOT_ARTIFACT_PANEL_WIDTH);
    useCopilotUIStore.setState({
      artifactPanel: {
        isOpen: true,
        isMinimized: false,
        isMaximized: true,
        width: 600,
        activeArtifact: null,
        history: [],
      },
    });
  });

  it("exits maximized mode when the artifact panel is resized", () => {
    useCopilotUIStore.getState().setArtifactPanelWidth(480);

    expect(useCopilotUIStore.getState().artifactPanel).toMatchObject({
      width: 480,
      isMaximized: false,
    });
    expect(window.localStorage.getItem(Key.COPILOT_ARTIFACT_PANEL_WIDTH)).toBe(
      "480",
    );
  });
});
