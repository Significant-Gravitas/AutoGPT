"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useCopilotUIStore } from "../../store";
import { useAreWorkspaceFileCardsOpen } from "../../useAreWorkspaceFileCardsOpen";
import { WorkspaceFilesPopover } from "../WorkspaceFileCards/components/WorkspaceFilesPopover";
import { File02Icon, SidebarRightIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  sessionId?: string | null;
}

export function ContextPanelToggle({ sessionId }: Props) {
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
  // Sized and stroked like the sidebar's nav icons so the chat's top-right
  // controls read as the same family.
  const buttonClass =
    "size-8 shrink-0 rounded-md transition-[background-color,transform] duration-150 ease-out hover:!bg-zinc-200 active:scale-[0.97] motion-reduce:transition-none";
  const iconClass = "!size-4 text-sidebar-foreground/90";
  const isArtifactsOpen = isOpen && activeTab === "artifacts";
  // An open artifact preview and the artifacts tab are both faces of the
  // right sidebar, so the sidebar toggle reads active for either and is the
  // one control that closes them — the panel carries no close button.
  const isRightSidebarOpen = hasArtifact || isArtifactsOpen;
  // While the right sidebar owns that region the inline files card has
  // nowhere to sit, so the workspace-files trigger becomes a popover; it
  // returns to the pinned card as soon as the sidebar closes.
  const showFilesAsPopover = isRightSidebarOpen;
  const isFilesCardOpen = useAreWorkspaceFileCardsOpen();

  function handleSidebarToggle() {
    if (hasArtifact) {
      closeArtifactPanel();
      return;
    }
    // Reopening brings back the preview that was showing when the sidebar
    // closed (e.g. hello.md), not just the tabs view.
    if (!isRightSidebarOpen && lastArtifact) {
      openArtifact(lastArtifact);
      return;
    }
    toggleContextPanelTab("artifacts");
  }

  return (
    <div className="flex shrink-0 items-center gap-1 p-2">
      {showFilesAsPopover && sessionId ? (
        <WorkspaceFilesPopover
          sessionId={sessionId}
          wrapperClassName="p-0"
          triggerClassName={buttonClass}
          iconClassName={iconClass}
        />
      ) : (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={() => toggleContextPanelTab("files")}
          aria-label={
            isFilesCardOpen ? "Hide workspace files" : "Open workspace files"
          }
          aria-pressed={isFilesCardOpen}
          className={cn(buttonClass, isFilesCardOpen && "bg-zinc-200/70")}
        >
          <Icon icon={File02Icon} className={iconClass} />
        </Button>
      )}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={handleSidebarToggle}
        aria-label={isRightSidebarOpen ? "Hide artifacts" : "Open artifacts"}
        aria-pressed={isRightSidebarOpen}
        className={cn(buttonClass, isRightSidebarOpen && "bg-zinc-200/70")}
      >
        <Icon icon={SidebarRightIcon} className={iconClass} />
      </Button>
    </div>
  );
}
