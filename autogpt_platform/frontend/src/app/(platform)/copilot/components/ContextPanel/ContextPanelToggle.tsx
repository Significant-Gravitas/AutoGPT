"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useCopilotUIStore } from "../../store";
import { useAreWorkspaceFileCardsOpen } from "../../useAreWorkspaceFileCardsOpen";
import { fileItemToArtifactRef } from "./components/FilesTab/helpers";
import {
  useSessionFiles,
  type SessionFile,
} from "./components/FilesTab/useSessionFiles";
import { LicenseDraftIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  sessionId?: string | null;
}

function getLastGeneratedFile(generated: SessionFile[]): SessionFile | null {
  if (generated.length === 0) return null;
  return generated.reduce((latest, file) =>
    new Date(file.item.created_at).getTime() >
    new Date(latest.item.created_at).getTime()
      ? file
      : latest,
  );
}

/** The chat's one top-right control: the artifacts toggle. It wears the name
 *  of the session's most recently generated file so the current working
 *  document stays visible, and clicking it opens that file directly in the
 *  artifact panel. Workspace files open from the thread chip instead. */
export function ContextPanelToggle({ sessionId = null }: Props) {
  const isOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const hasArtifact = useCopilotUIStore(
    (s) => s.artifactPanel.activeArtifact != null,
  );
  const activeTab = useCopilotUIStore((s) => s.artifactPanel.activeTab);
  const toggleContextPanelTab = useCopilotUIStore(
    (s) => s.toggleContextPanelTab,
  );
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);
  const lastArtifact = useCopilotUIStore((s) => s.artifactPanel.lastArtifact);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const { generated } = useSessionFiles(sessionId);
  const lastGenerated = getLastGeneratedFile(generated);
  const isFilesCardOpen = useAreWorkspaceFileCardsOpen();
  const isArtifactsOpen = isOpen && activeTab === "artifacts";
  // An open artifact preview and the artifacts tab are both faces of the
  // right sidebar, so the toggle reads active for either and is the one
  // control that closes them — the panel carries no close button.
  const isRightSidebarOpen = hasArtifact || isArtifactsOpen;

  // The open activity card already lists the same file, so the labeled
  // button floating above it is pure duplication — the card's rows are the
  // way in while it shows.
  if (isFilesCardOpen) return null;

  // With the panel open its own header already names the document, so the
  // button collapses to the bare icon; closed, the name is the reminder of
  // what's being worked on.
  const showFileName = lastGenerated != null && !isRightSidebarOpen;

  function handleSidebarToggle() {
    if (hasArtifact) {
      closeArtifactPanel();
      return;
    }
    if (isArtifactsOpen) {
      toggleContextPanelTab("artifacts");
      return;
    }
    // Straight to the document: the freshest generated file, then the
    // remembered preview, and only then the tabs view.
    const target =
      (lastGenerated ? fileItemToArtifactRef(lastGenerated.item) : null) ??
      lastArtifact;
    if (target) {
      openArtifact(target);
      return;
    }
    toggleContextPanelTab("artifacts");
  }

  return (
    <div className="flex shrink-0 items-center gap-1 p-2">
      <Button
        type="button"
        variant="ghost"
        size={showFileName ? "sm" : "icon"}
        onClick={handleSidebarToggle}
        aria-label={
          isRightSidebarOpen
            ? "Hide artifacts"
            : lastGenerated
              ? `Open ${lastGenerated.item.name}`
              : "Open artifacts"
        }
        aria-pressed={isRightSidebarOpen}
        className={cn(
          // Sized and stroked like the sidebar's nav icons so the chat's
          // top-right control reads as the same family.
          "shrink-0 rounded-md transition-[background-color,transform] duration-150 ease-out hover:bg-zinc-100 active:scale-[0.97] motion-reduce:transition-none",
          showFileName ? "h-8 gap-1.5 px-2" : "size-8",
          isRightSidebarOpen && "bg-zinc-100",
        )}
      >
        <Icon
          icon={LicenseDraftIcon}
          className="!size-4 text-sidebar-foreground/90"
        />
        {showFileName && (
          <span className="max-w-[9rem] truncate text-xs font-medium text-sidebar-foreground/90">
            {lastGenerated.item.name}
          </span>
        )}
      </Button>
    </div>
  );
}
