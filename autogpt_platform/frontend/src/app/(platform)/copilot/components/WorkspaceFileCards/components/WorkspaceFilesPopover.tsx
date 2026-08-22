"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { CheckListIcon } from "@hugeicons/core-free-icons";
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
    setIsPopoverOpen(false);
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
          onRequestDelete={setPendingDelete}
          onConfirmDelete={handleConfirmDelete}
          onCancelDelete={() => setPendingDelete(null)}
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

  return (
    <div className={cn("flex shrink-0 items-start p-2", wrapperClassName)}>
      <Popover open={isPopoverOpen} onOpenChange={setIsPopoverOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Workspace files"
            className={cn(
              triggerClassName ?? "size-9 rounded-xl",
              isPopoverOpen && "bg-zinc-200/70 text-zinc-900",
            )}
          >
            <Icon
              icon={CheckListIcon}
              className={iconClassName ?? "!size-[1.15rem]"}
            />
          </Button>
        </PopoverTrigger>
        <PopoverContent
          align="end"
          className="w-80 rounded-3xl border-none bg-white/90 px-4 py-3 shadow-none backdrop-blur smooth-shadow-ring-sm"
          // The delete confirm renders as a modal dialog outside the popover
          // tree — without these guards the popover reads that interaction as
          // "outside" and closes, unmounting the dialog mid-confirm.
          onInteractOutside={(e) => {
            if (pendingDelete) e.preventDefault();
          }}
          onEscapeKeyDown={(e) => {
            if (pendingDelete) e.preventDefault();
          }}
        >
          {sections.length === 0 ? (
            <p className="py-2 text-center text-sm text-zinc-400">
              Nothing here yet.
            </p>
          ) : (
            // One popover can't float three cards, so the stack's card breaks
            // become hairlines here.
            sections.map((section, index) => (
              <Fragment key={section.key}>
                {index > 0 && <div className="my-3 border-t border-zinc-100" />}
                {section.node}
              </Fragment>
            ))
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}
