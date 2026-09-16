"use client";

import { Download01Icon, File02Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useAreWorkspaceFileCardsOpen } from "../../useAreWorkspaceFileCardsOpen";
import { SessionActivityCard } from "./components/SessionActivityCard";
import { StackSection } from "./components/StackSection";
import { WorkspaceFilesContent } from "./components/WorkspaceFilesContent";
import { useSessionActivity } from "./useSessionActivity";
import { useWorkspaceFileCards } from "./useWorkspaceFileCards";

interface Props {
  sessionId: string | null;
}

// Matches the chat column's shift transition (see ChatMessagesContainer) so
// the card lands as the messages finish sliding aside.
const CARD_EASE: [number, number, number, number] = [0.32, 0.72, 0, 1];
const CARD_TRANSITION = { duration: 0.3, ease: CARD_EASE };

/**
 * Workspace files as a floating card pinned to the chat's top right. The
 * workspace-files icon toggles ``artifactPanel.isOpen``; this grid answers it
 * inside the chat column rather than pushing a drawer over it.
 *
 * The card body — and its file-list request and transcript scans — mounts
 * only while the card is showing.
 */
export function WorkspaceFileCards({ sessionId }: Props) {
  const isOpen = useAreWorkspaceFileCardsOpen();
  return (
    <AnimatePresence initial={false}>
      {isOpen && sessionId && (
        <OpenWorkspaceFileCards
          key="workspace-file-cards"
          sessionId={sessionId}
        />
      )}
    </AnimatePresence>
  );
}

function OpenWorkspaceFileCards({ sessionId }: { sessionId: string }) {
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

  // An empty-state card is noise floating over the chat — the files card only
  // earns its space once there's something in it (or something to report).
  const showFilesCard = isLoading || isError || files.length > 0;
  const { runs, schedules } = useSessionActivity(sessionId);
  const hasActivity = runs.length > 0 || schedules.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, x: 12, scale: 0.98 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 12, scale: 0.98 }}
      transition={CARD_TRANSITION}
      className="absolute right-8 top-3 z-30 flex w-80 max-w-[calc(100%-2rem)] flex-col gap-3"
    >
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
    </motion.div>
  );
}
