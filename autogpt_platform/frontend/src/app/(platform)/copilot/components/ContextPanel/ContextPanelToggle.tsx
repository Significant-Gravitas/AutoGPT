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
import { ComputerIcon, LicenseDraftIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useIsMobile } from "../../useIsMobile";
import { TeamToggle } from "./components/TeamTab/components/TeamToggle";
import { useTeamRoster } from "./components/TeamTab/useTeamRoster";

interface Props {
  sessionId?: string | null;
}

function getLastGeneratedFile(generated: SessionFile[]): SessionFile | null {
  let latest: SessionFile | null = null;
  let latestTime = Number.NEGATIVE_INFINITY;
  for (const file of generated) {
    const time = new Date(file.item.created_at).getTime();
    if (latest === null || time > latestTime) {
      latest = file;
      latestTime = time;
    }
  }
  return latest;
}

/** The chat's top-right controls: the Team toggle (only once someone has
 *  been delegated to), the Computer toggle and the artifacts toggle, one per
 *  face of the side panel. The artifacts toggle wears the
 *  name of the session's most recently generated file so the current working
 *  document stays visible, and clicking it opens that file directly in the
 *  artifact panel. Workspace files open from the thread chip instead. The
 *  Computer toggle is always there for a chat with a session: the panel's
 *  own "Turn on screen" button lives on that face, so it must be reachable
 *  before any desktop exists and without an artifact to switch from. */
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
  const isComputerOpen = useCopilotUIStore(
    (s) => s.artifactPanel.isComputerOpen,
  );
  const openComputer = useCopilotUIStore((s) => s.openComputer);
  const closeComputer = useCopilotUIStore((s) => s.closeComputer);
  const setArtifactPanelMode = useCopilotUIStore((s) => s.setArtifactPanelMode);
  const isMobile = useIsMobile();
  const { deliverables } = useSessionFiles(sessionId);
  const lastGenerated = getLastGeneratedFile(deliverables);
  const isFilesCardOpen = useAreWorkspaceFileCardsOpen();
  const isArtifactsOpen = isOpen && activeTab === "artifacts";
  // An open artifact preview and the artifacts tab are both the document
  // face of the right sidebar, so the toggle reads active for either and is
  // the one control that closes them — the panel carries no close button.
  // With the computer face on top, the artifact underneath is not showing.
  const isDocumentOpen = !isComputerOpen && (hasArtifact || isArtifactsOpen);
  const isRightSidebarOpen = isDocumentOpen || isComputerOpen;
  // The mobile sheet has no computer face to open.
  const showComputerToggle = !!sessionId && !isMobile;

  // The open activity card already lists the same file, so the labeled
  // artifacts button floating above it is pure duplication — the card's rows
  // are the way in while it shows. The Computer button is not about files
  // and stays, or "Turn on screen" is out of reach until the card closes.
  const showArtifactsToggle = !isFilesCardOpen;
  const hasTeam = useTeamRoster(sessionId).rows.length > 0;
  if (!showArtifactsToggle && !showComputerToggle && !hasTeam) return null;

  // With the panel open its own header already names the document, so the
  // button collapses to the bare icon; closed, the name is the reminder of
  // what's being worked on.
  const showFileName = lastGenerated != null && !isRightSidebarOpen;

  // Straight to the document: the freshest generated file, then the
  // remembered preview, and only then the tabs view.
  function openDocument() {
    const target =
      (lastGenerated ? fileItemToArtifactRef(lastGenerated.item) : null) ??
      lastArtifact;
    if (target) {
      openArtifact(target);
      return;
    }
    // Already on the library (the computer face was just turned off it):
    // toggling would close it.
    if (isArtifactsOpen) return;
    toggleContextPanelTab("artifacts");
  }

  function handleSidebarToggle() {
    if (isComputerOpen) {
      // Turn the panel to its document face: the preview that was under
      // the computer if there is one, else wherever a closed panel opens.
      setArtifactPanelMode("artifact");
      if (hasArtifact) return;
      openDocument();
      return;
    }
    if (hasArtifact) {
      closeArtifactPanel();
      return;
    }
    if (isArtifactsOpen) {
      toggleContextPanelTab("artifacts");
      return;
    }
    openDocument();
  }

  function handleComputerToggle() {
    // Back to whatever the computer was covering: the preview, the tab, or
    // nothing at all.
    if (isComputerOpen) closeComputer();
    else openComputer();
  }

  // Sized and stroked like the sidebar's nav icons so the chat's top-right
  // controls read as the same family.
  const toggleClass =
    "shrink-0 rounded-md transition-[background-color,transform] duration-150 ease-out hover:bg-zinc-100 active:scale-[0.97] motion-reduce:transition-none";

  return (
    <div className="flex shrink-0 items-center gap-1 p-2">
      <TeamToggle sessionId={sessionId} />
      {showComputerToggle && (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={handleComputerToggle}
          aria-label={isComputerOpen ? "Hide computer" : "Open computer"}
          aria-pressed={isComputerOpen}
          className={cn(toggleClass, "size-8", isComputerOpen && "bg-zinc-100")}
        >
          <Icon
            icon={ComputerIcon}
            className="!size-4 text-sidebar-foreground/90"
          />
        </Button>
      )}
      {showArtifactsToggle && (
        <Button
          type="button"
          variant="ghost"
          size={showFileName ? "sm" : "icon"}
          onClick={handleSidebarToggle}
          aria-label={
            isDocumentOpen
              ? "Hide artifacts"
              : lastGenerated
                ? `Open ${lastGenerated.item.name}`
                : "Open artifacts"
          }
          aria-pressed={isDocumentOpen}
          className={cn(
            toggleClass,
            showFileName ? "h-8 gap-1.5 px-2" : "size-8",
            isDocumentOpen && "bg-zinc-100",
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
      )}
    </div>
  );
}
