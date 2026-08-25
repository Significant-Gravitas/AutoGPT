import { useCopilotUIStore } from "./store";

// Mirrors the files card's visibility (useWorkspaceFileCards.isOpen). The
// open flag also drives the docked artifacts panel and the artifact preview,
// so only a non-artifacts tab with no preview means the floating card is on
// screen. The docked panel narrows the chat column on its own, so it must
// never also trigger the slide, or the two shifts stack and push the
// messages off screen under the app sidebar.
export function useAreWorkspaceFileCardsOpen() {
  return useCopilotUIStore(
    (s) =>
      s.artifactPanel.isOpen &&
      s.artifactPanel.activeArtifact == null &&
      s.artifactPanel.activeTab !== "artifacts",
  );
}
