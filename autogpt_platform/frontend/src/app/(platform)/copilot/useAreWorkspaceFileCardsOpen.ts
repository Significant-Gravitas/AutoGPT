import { useCopilotUIStore } from "./store";

// The one condition under which the floating workspace-files card is on
// screen — read by the card itself and by the composer and message column
// that slide aside for it. The open flag also drives the docked artifacts
// panel and the artifact preview, so only a non-artifacts tab with no
// preview means the card is showing. The docked panel narrows the chat
// column on its own, so it must never also trigger the slide, or the two
// shifts stack and push the messages off screen under the app sidebar.
export function useAreWorkspaceFileCardsOpen() {
  return useCopilotUIStore(
    (s) =>
      s.artifactPanel.isOpen &&
      s.artifactPanel.activeArtifact == null &&
      s.artifactPanel.activeTab !== "artifacts",
  );
}
