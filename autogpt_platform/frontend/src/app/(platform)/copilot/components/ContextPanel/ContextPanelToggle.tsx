"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useCopilotUIStore, type ContextPanelExpert } from "../../store";
import { ComputerIcon, LicenseDraftIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useIsMobile } from "../../useIsMobile";
import { IntegrationsToggle } from "./components/IntegrationsToggle/IntegrationsToggle";
import { useSessionFiles } from "./components/FilesTab/useSessionFiles";
import { useEffect } from "react";

interface Props {
  sessionId?: string | null;
  /** The chat's live expert, whose integrations lead the controls. */
  expert?: ContextPanelExpert | null;
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
  const { deliverables, documentCount: liveDocumentCount } =
    useSessionFiles(sessionId);
  const documentCount = Math.max(deliverables.length, liveDocumentCount);
  const expertId = expert?.id ?? null;
  const expertName = expert?.name ?? null;
  const setContextPanelExpert = useCopilotUIStore(
    (s) => s.setContextPanelExpert,
  );
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
  const isPanelOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  const openComputer = useCopilotUIStore((s) => s.openComputer);
  const closeComputer = useCopilotUIStore((s) => s.closeComputer);
  const isMobile = useIsMobile();
  // The mobile sheet has no computer face to open.
  const showComputerToggle = !!sessionId && !isMobile;

  useEffect(() => {
    setContextPanelExpert(
      expertId && expertName ? { id: expertId, name: expertName } : null,
    );
  }, [expertId, expertName, setContextPanelExpert]);

  function handleComputerToggle() {
    // Back to whatever the computer was covering: the preview, the tab, or
    // nothing at all.
    if (isComputerOpen) closeComputer();
    else openComputer();
  }

  return (
    // With the side panel open the chat column narrows under these controls,
    // so they lift off the messages instead of blending into them.
    <div
      className={cn(
        "m-1 flex shrink-0 items-center gap-1 rounded-xl border border-transparent bg-white p-1 transition-shadow duration-150",
        isPanelOpen && "border-zinc-200/70 shadow-sm",
      )}
    >
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
        aria-label={`${isFilesOpen ? "Hide" : "Open"} files${
          documentCount > 0
            ? ` (${documentCount} ${documentCount === 1 ? "document" : "documents"})`
            : ""
        }`}
        aria-pressed={isFilesOpen}
        className={cn(
          toggleClass,
          "h-8",
          documentCount > 0 ? "w-auto gap-1 px-2" : "w-8",
          isFilesOpen && "bg-zinc-100",
        )}
      >
        <Icon
          icon={LicenseDraftIcon}
          className="!size-4 text-sidebar-foreground/90"
        />
        {documentCount > 0 && (
          <span className="text-xs font-medium tabular-nums text-sidebar-foreground/90">
            {documentCount}
          </span>
        )}
      </Button>
    </div>
  );
}
