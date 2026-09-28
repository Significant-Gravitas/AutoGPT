"use client";

import { Download01Icon, File02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SessionActivityCard } from "./components/SessionActivityCard";
import { StackSection } from "./components/StackSection";
import { WorkspaceFilesContent } from "./components/WorkspaceFilesContent";
import { useSessionActivity } from "./useSessionActivity";
import { useWorkspaceFileCards } from "./useWorkspaceFileCards";

interface Props {
  sessionId: string;
}

/** The chat's files, then the runs and schedules it set in motion — the
 *  files tab of the docked side panel. */
export function WorkspaceFileCards({ sessionId }: Props) {
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

  // The files section only shows once there's something in it (or something
  // to report); a chat with nothing at all gets one empty state.
  const showFilesCard = isLoading || isError || files.length > 0;
  const { runs, schedules } = useSessionActivity(sessionId);
  const hasActivity = runs.length > 0 || schedules.length > 0;

  return (
    <div className="flex flex-col gap-3">
      {!showFilesCard && !hasActivity && (
        <div className="rounded-3xl bg-white/90 px-4 py-3 backdrop-blur smooth-shadow-ring-sm">
          <p className="py-2 text-center text-sm text-zinc-400">
            Nothing here yet.
          </p>
        </div>
      )}
      {showFilesCard && (
        <StackSection
          title="Files"
          icon={File02Icon}
          count={files.length || undefined}
          action={
            files.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleDownloadAll}
                loading={isZipping}
                aria-label="Download all"
                className="size-6 rounded-lg !p-0 text-zinc-400"
              >
                <Icon icon={Download01Icon} size={14} />
              </Button>
            )
          }
        >
          <WorkspaceFilesContent
            files={files}
            isLoading={isLoading}
            isError={isError}
            isDeleting={isDeleting}
            isZipping={isZipping}
            pendingDelete={pendingDelete}
            onOpen={handleOpen}
            onDownload={handleDownload}
            onRequestDelete={setPendingDelete}
            onConfirmDelete={handleConfirmDelete}
            onCancelDelete={() => setPendingDelete(null)}
            onDownloadAll={handleDownloadAll}
            showHeader={false}
          />
        </StackSection>
      )}
      <SessionActivityCard sessionId={sessionId} />
    </div>
  );
}
