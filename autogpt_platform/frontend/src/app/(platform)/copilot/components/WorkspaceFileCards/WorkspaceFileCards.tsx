"use client";

import { Download01Icon, File02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { ContextPanelExpert } from "../../store";
import { DeleteFileDialog } from "../ContextPanel/components/FilesTab/components/DeleteFileDialog";
import { SessionActivityCard } from "./components/SessionActivityCard";
import { StackSection } from "./components/StackSection";
import { WorkspaceFilesContent } from "./components/WorkspaceFilesContent";
import { useExpertDocuments } from "./useExpertDocuments";
import { useSessionActivity } from "./useSessionActivity";
import { useWorkspaceFileCards } from "./useWorkspaceFileCards";

interface Props {
  sessionId: string;
  expert?: ContextPanelExpert | null;
}

/** The chat's files, then the runs and schedules it set in motion — the
 *  files tab of the docked side panel. */
export function WorkspaceFileCards({ sessionId, expert = null }: Props) {
  const {
    files,
    documentCount: sessionDocumentCount,
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
  const {
    documents: expertDocuments,
    isLoading: isExpertDocumentsLoading,
    isError: isExpertDocumentsError,
  } = useExpertDocuments(expert?.id ?? null, sessionDocumentCount);

  // The files section only shows once there's something in it (or something
  // to report); a chat with nothing at all gets one empty state.
  const showFilesCard = isLoading || isError || files.length > 0;
  const { runs, schedules } = useSessionActivity(sessionId);
  const hasActivity = runs.length > 0 || schedules.length > 0;
  const hasExpertSection = Boolean(expert);

  return (
    <div className="flex flex-col gap-3">
      {!showFilesCard && !hasActivity && !hasExpertSection && (
        <div className="rounded-3xl bg-white/90 px-4 py-3 backdrop-blur smooth-shadow-ring-sm">
          <p className="py-2 text-center text-sm text-zinc-400">
            Nothing here yet.
          </p>
        </div>
      )}
      {showFilesCard && (
        <StackSection
          title="Files in this chat"
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
            isZipping={isZipping}
            onOpen={handleOpen}
            onDownload={handleDownload}
            onRequestDelete={setPendingDelete}
            onDownloadAll={handleDownloadAll}
            showHeader={false}
          />
        </StackSection>
      )}
      {expert && (
        <StackSection
          title={`All ${expert.name}'s documents`}
          icon={File02Icon}
          plain
        >
          <WorkspaceFilesContent
            files={expertDocuments}
            isLoading={isExpertDocumentsLoading}
            isError={isExpertDocumentsError}
            isZipping={false}
            onOpen={handleOpen}
            onDownload={handleDownload}
            onRequestDelete={setPendingDelete}
            onDownloadAll={() => undefined}
            emptyMessage={`${expert.name} hasn't created any documents yet.`}
            showHeader={false}
            rowStyle="detailed"
          />
        </StackSection>
      )}
      <SessionActivityCard sessionId={sessionId} />
      <DeleteFileDialog
        fileName={pendingDelete?.item.name ?? null}
        isDeleting={isDeleting}
        onConfirm={handleConfirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}
