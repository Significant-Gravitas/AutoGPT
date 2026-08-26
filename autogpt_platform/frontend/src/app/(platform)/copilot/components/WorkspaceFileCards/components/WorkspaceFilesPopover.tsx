"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { File02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Fragment, useState } from "react";
import type { SessionFile } from "../../ContextPanel/components/FilesTab/useSessionFiles";
import { useSessionActivity } from "../useSessionActivity";
import { useWorkspaceFileCards } from "../useWorkspaceFileCards";
import { RunsList, SchedulesList } from "./SessionActivityContent";
import { WorkspaceFilesContent } from "./WorkspaceFilesContent";

// The stack hoists section titles above each card; one popover surface can't,
// so the labels stay inline here.
const SECTION_LABEL = "block pb-1.5 text-[15px] text-zinc-400";

interface Props {
  sessionId: string;
  wrapperClassName?: string;
  triggerClassName?: string;
  iconClassName?: string;
}

/**
 * While an artifact (or the artifacts side panel) owns the right side there's
 * nowhere for the inline files card to sit, so the workspace-files trigger
 * swaps to this popover: same icon, same card content, anchored to the button
 * instead of pinned to the chat column.
 */
export function WorkspaceFilesPopover({
  sessionId,
  wrapperClassName,
  triggerClassName,
  iconClassName,
}: Props) {
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  // The delete confirm renders as a modal dialog outside the popover tree —
  // without the guards below the popover reads that interaction as "outside"
  // and closes, unmounting the dialog mid-confirm.
  const [isConfirmingDelete, setIsConfirmingDelete] = useState(false);

  // The trigger can still close the popover mid-confirm; the content (and its
  // pending delete) unmounts with it, so the guard must not outlive them.
  function handleOpenChange(open: boolean) {
    setIsPopoverOpen(open);
    if (!open) setIsConfirmingDelete(false);
  }

  return (
    <div className={cn("flex shrink-0 items-start p-2", wrapperClassName)}>
      <Popover open={isPopoverOpen} onOpenChange={handleOpenChange}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Workspace files"
            // p-0: the atom's size="icon" is p-3, which on a size-8 trigger
            // leaves an 8px content box and squeezes the glyph. shrink-0: the
            // atom has no [&_svg]:shrink-0, so the icon would flex-shrink to
            // fit regardless of its own size class.
            className={cn(
              "p-0",
              triggerClassName ?? "size-9 rounded-xl",
              isPopoverOpen && "bg-zinc-200/70 text-zinc-900",
            )}
          >
            <Icon
              icon={File02Icon}
              className={cn("shrink-0", iconClassName ?? "!size-[1.15rem]")}
            />
          </Button>
        </PopoverTrigger>
        <PopoverContent
          align="end"
          className="w-80 rounded-3xl border-none bg-white/90 px-4 py-3 shadow-none backdrop-blur smooth-shadow-ring-sm"
          onInteractOutside={(e) => {
            if (isConfirmingDelete) e.preventDefault();
          }}
          onEscapeKeyDown={(e) => {
            if (isConfirmingDelete) e.preventDefault();
          }}
        >
          {/* Mounted only while the popover is open, so the file-list request
              and the transcript scans wait for the click — as the floating
              card's body does. */}
          <WorkspaceFilesPopoverContent
            sessionId={sessionId}
            onFileOpened={() => setIsPopoverOpen(false)}
            onConfirmingDeleteChange={setIsConfirmingDelete}
          />
        </PopoverContent>
      </Popover>
    </div>
  );
}

interface ContentProps {
  sessionId: string;
  onFileOpened: () => void;
  onConfirmingDeleteChange: (isConfirming: boolean) => void;
}

function WorkspaceFilesPopoverContent({
  sessionId,
  onFileOpened,
  onConfirmingDeleteChange,
}: ContentProps) {
  const {
    files,
    isLoading,
    isError,
    isDeleting,
    isZipping,
    pendingDelete,
    setPendingDelete,
    handleOpen,
    handleDownload,
    handleConfirmDelete,
    handleDownloadAll,
  } = useWorkspaceFileCards(sessionId);
  const { runs, schedules } = useSessionActivity(sessionId);

  function handleOpenFile(file: SessionFile) {
    handleOpen(file);
    onFileOpened();
  }

  function handleRequestDelete(file: SessionFile) {
    setPendingDelete(file);
    onConfirmingDeleteChange(true);
  }

  function handleCancelDelete() {
    setPendingDelete(null);
    onConfirmingDeleteChange(false);
  }

  async function handleConfirm() {
    await handleConfirmDelete();
    onConfirmingDeleteChange(false);
  }

  // Mirrors the floating stack: each section only earns its space once it has
  // something to show (files keep loading/error states so a slow fetch doesn't
  // read as "no files"). Keyed by section identity rather than array index so
  // a section dropping out doesn't reassign a neighboring section's state.
  const sections = [
    {
      key: "files",
      node: (isLoading || isError || files.length > 0) && (
        <WorkspaceFilesContent
          files={files}
          isLoading={isLoading}
          isError={isError}
          isDeleting={isDeleting}
          isZipping={isZipping}
          pendingDelete={pendingDelete}
          onOpen={handleOpenFile}
          onDownload={handleDownload}
          onRequestDelete={handleRequestDelete}
          onConfirmDelete={handleConfirm}
          onCancelDelete={handleCancelDelete}
          onDownloadAll={handleDownloadAll}
        />
      ),
    },
    {
      key: "runs",
      node: runs.length > 0 && (
        <>
          <span className={SECTION_LABEL}>Runs ({runs.length})</span>
          <RunsList runs={runs} />
        </>
      ),
    },
    {
      key: "schedules",
      node: schedules.length > 0 && (
        <>
          <span className={SECTION_LABEL}>Schedules ({schedules.length})</span>
          <SchedulesList schedules={schedules} />
        </>
      ),
    },
  ].filter((section) => section.node);

  if (sections.length === 0) {
    return (
      <p className="py-2 text-center text-sm text-zinc-400">
        Nothing here yet.
      </p>
    );
  }

  // One popover can't float three cards, so the stack's card breaks become
  // hairlines here.
  return sections.map((section, index) => (
    <Fragment key={section.key}>
      {index > 0 && <div className="my-3 border-t border-zinc-100" />}
      {section.node}
    </Fragment>
  ));
}
