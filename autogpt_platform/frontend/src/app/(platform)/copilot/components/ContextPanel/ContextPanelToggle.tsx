"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useCopilotUIStore } from "../../store";
import { LicenseDraftIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

/** The chat's one top-right control: the artifacts toggle. Workspace files
 *  open from the thread chip instead, so the old files trigger is gone. */
export function ContextPanelToggle() {
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
  const isArtifactsOpen = isOpen && activeTab === "artifacts";
  // An open artifact preview and the artifacts tab are both faces of the
  // right sidebar, so the toggle reads active for either and is the one
  // control that closes them — the panel carries no close button.
  const isRightSidebarOpen = hasArtifact || isArtifactsOpen;

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
      <Button
        type="button"
        variant="ghost"
        size="icon"
        onClick={handleSidebarToggle}
        aria-label={isRightSidebarOpen ? "Hide artifacts" : "Open artifacts"}
        aria-pressed={isRightSidebarOpen}
        className={cn(
          // Sized and stroked like the sidebar's nav icons so the chat's
          // top-right control reads as the same family.
          "size-8 shrink-0 rounded-md transition-[background-color,transform] duration-150 ease-out hover:!bg-zinc-200 active:scale-[0.97] motion-reduce:transition-none",
          isRightSidebarOpen && "bg-zinc-200/70",
        )}
      >
        <Icon
          icon={LicenseDraftIcon}
          className="!size-4 text-sidebar-foreground/90"
        />
      </Button>
    </div>
  );
}
