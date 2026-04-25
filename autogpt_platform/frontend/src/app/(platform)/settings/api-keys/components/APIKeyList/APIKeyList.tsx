"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { APIKeyListEmpty } from "../APIKeyListEmpty/APIKeyListEmpty";
import { APIKeyListSkeleton } from "../APIKeyListSkeleton/APIKeyListSkeleton";
import { APIKeyRow } from "../APIKeyRow/APIKeyRow";
import { APIKeySelectionBar } from "../APIKeySelectionBar/APIKeySelectionBar";
import { DeleteAPIKeyDialog } from "../DeleteAPIKeyDialog/DeleteAPIKeyDialog";
import { useAPIKeyListView } from "./useAPIKeyListView";

export function APIKeyList() {
  const {
    keys,
    isLoading,
    isError,
    error,
    refetch,
    isEmpty,
    selection,
    deleteTarget,
    requestDelete,
    closeDeleteDialog,
    handleDeleted,
  } = useAPIKeyListView();
  const reduceMotion = useReducedMotion();

  if (isLoading) return <APIKeyListSkeleton />;
  if (isError) {
    const message = error instanceof Error ? error.message : undefined;
    return (
      <ErrorCard
        context="API keys"
        responseError={message ? { message } : undefined}
        onRetry={() => {
          refetch();
        }}
      />
    );
  }
  if (isEmpty) return <APIKeyListEmpty />;

  return (
    <div className="flex w-full flex-col gap-3">
      <AnimatePresence initial={false}>
        {selection.selectedCount > 0 && (
          <motion.div
            key="selection-bar"
            initial={
              reduceMotion
                ? { opacity: 0 }
                : { opacity: 0, height: 0, marginBottom: -12 }
            }
            animate={
              reduceMotion
                ? { opacity: 1 }
                : { opacity: 1, height: "auto", marginBottom: 0 }
            }
            exit={
              reduceMotion
                ? { opacity: 0 }
                : { opacity: 0, height: 0, marginBottom: -12 }
            }
            transition={{ duration: 0.2, ease: [0, 0, 0.2, 1] }}
            className="sticky top-0 z-20 overflow-hidden bg-[#F9F9FA]"
          >
            <APIKeySelectionBar
              selectedCount={selection.selectedCount}
              allSelected={selection.allSelected}
              onSelectAll={selection.selectAll}
              onDeselectAll={selection.clear}
              onDeleteSelected={() => requestDelete([...selection.selectedIds])}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex flex-col divide-y divide-zinc-200 overflow-hidden rounded-[8px] border border-zinc-200 bg-white">
        {keys.map((key) => (
          <APIKeyRow
            key={key.id}
            apiKey={key}
            selected={selection.isSelected(key.id)}
            onToggleSelected={() => selection.toggle(key.id)}
            onDelete={() => requestDelete([key.id])}
          />
        ))}
      </div>

      {deleteTarget && (
        <DeleteAPIKeyDialog
          open
          keyIds={deleteTarget}
          onOpenChange={closeDeleteDialog}
          onDeleted={handleDeleted}
        />
      )}
    </div>
  );
}
