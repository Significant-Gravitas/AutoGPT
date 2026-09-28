"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useCopilotUIStore, type IntegrationsPanelExpert } from "../../store";
import { ComputerIcon, LicenseDraftIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useIsMobile } from "../../useIsMobile";
import { IntegrationsToggle } from "./components/IntegrationsToggle/IntegrationsToggle";

interface Props {
  sessionId?: string | null;
  /** The chat's live expert, whose integrations lead the controls. */
  expert?: IntegrationsPanelExpert | null;
}

// Sized and stroked like the sidebar's nav icons so the chat's top-right
// controls read as the same family.
const toggleClass =
  "shrink-0 rounded-md transition-[background-color,transform] duration-150 ease-out hover:bg-zinc-100 active:scale-[0.97] motion-reduce:transition-none";

/** The chat's top-right controls: the expert's integrations, the Computer
 *  toggle and the files toggle, each opening its face of the side panel. The
 *  Computer toggle is always there for a chat with a session: the panel's
 *  own "Turn on screen" button lives on that face, so it must be reachable
 *  before any desktop exists. */
export function ContextPanelToggle({ sessionId = null, expert = null }: Props) {
  const isFilesOpen = useCopilotUIStore(
    (s) =>
      s.artifactPanel.isOpen &&
      s.artifactPanel.activeTab === "files" &&
      s.artifactPanel.activeArtifact == null &&
      !s.artifactPanel.isComputerOpen,
  );
  const toggleContextPanelTab = useCopilotUIStore(
    (s) => s.toggleContextPanelTab,
  );
  const isComputerOpen = useCopilotUIStore(
    (s) => s.artifactPanel.isComputerOpen,
  );
  const openComputer = useCopilotUIStore((s) => s.openComputer);
  const closeComputer = useCopilotUIStore((s) => s.closeComputer);
  const isMobile = useIsMobile();
  // The mobile sheet has no computer face to open.
  const showComputerToggle = !!sessionId && !isMobile;

  function handleComputerToggle() {
    // Back to whatever the computer was covering: the preview, the tab, or
    // nothing at all.
    if (isComputerOpen) closeComputer();
    else openComputer();
  }

  return (
    <div className="flex shrink-0 items-center gap-1 p-2">
      {expert && <IntegrationsToggle expert={expert} className={toggleClass} />}
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
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={() => toggleContextPanelTab("files")}
        aria-label={isFilesOpen ? "Hide files" : "Open files"}
        aria-pressed={isFilesOpen}
        className={cn(toggleClass, "size-8", isFilesOpen && "bg-zinc-100")}
      >
        <Icon
          icon={LicenseDraftIcon}
          className="!size-4 text-sidebar-foreground/90"
        />
      </Button>
    </div>
  );
}
