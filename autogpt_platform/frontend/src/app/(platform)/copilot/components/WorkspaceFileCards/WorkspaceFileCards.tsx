"use client";

import { Download01Icon, File02Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SessionActivityCard } from "./components/SessionActivityCard";
import { StackSection } from "./components/StackSection";
import { WorkspaceFilesContent } from "./components/WorkspaceFilesContent";
import { useWorkspaceFileCards } from "./useWorkspaceFileCards";

interface Props {
  sessionId: string | null;
}

// Matches the chat column's shift transition (see ChatMessagesContainer) so
// the card lands as the messages finish sliding aside.
const CARD_EASE: [number, number, number, number] = [0.32, 0.72, 0, 1];
const CARD_TRANSITION = { duration: 0.3, ease: CARD_EASE };

/**
 * Workspace files as a floating card pinned to the chat's top right — the
 * new-tool-UI replacement for the Context Panel's side sheet.
 *
 * Same trigger, same store flag: the workspace-files icon toggles
 * ``artifactPanel.isOpen``; under the flag the panel renders nothing and this
 * grid answers instead, so files stay inside the chat column rather than
 * pushing a drawer over it.
 */
export function WorkspaceFileCards({ sessionId }: Props) {
  const {
    files,
    isOpen,
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

  return (
    <AnimatePresence initial={false}>
      {isOpen && sessionId && (
        <motion.div
          initial={{ opacity: 0, x: 12, scale: 0.98 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={{ opacity: 0, x: 12, scale: 0.98 }}
          transition={CARD_TRANSITION}
          className="absolute right-8 top-18 z-30 flex w-80 max-w-[calc(100%-2rem)] flex-col gap-3"
        >
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
      )}
    </AnimatePresence>
  );
}
