"use client";

import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { DeleteConfirmDialog } from "../DeleteConfirmDialog/DeleteConfirmDialog";
import { IntegrationsListEmpty } from "../IntegrationsListEmpty/IntegrationsListEmpty";
import { IntegrationsSearch } from "../IntegrationsSearch/IntegrationsSearch";
import { IntegrationsSelectionBar } from "../IntegrationsSelectionBar/IntegrationsSelectionBar";
import { ProviderGroup } from "../ProviderGroup/ProviderGroup";
import { IntegrationsListSkeleton } from "./IntegrationsListSkeleton";
import { useIntegrationsList } from "./useIntegrationsList";

const LIST_CONTAINER_VARIANTS: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.08, delayChildren: 0.05 },
  },
};

const LIST_ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 16 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
};

const REDUCED_MOTION_ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1 },
};

export function IntegrationsList() {
  const {
    query,
    setQuery,
    providers,
    isLoading,
    isError,
    error,
    refetch,
    isEmpty,
    selection,
    requestDelete,
    isDeleting,
    isDeletingId,
    buildTargets,
  } = useIntegrationsList();
  const reduceMotion = useReducedMotion();
  const [pendingDeleteIds, setPendingDeleteIds] = useState<string[]>([]);
  const [pendingForceIds, setPendingForceIds] = useState<string[]>([]);

  function askDelete(ids: string[]) {
    if (ids.length === 0) return;
    setPendingDeleteIds(ids);
  }

  async function confirmDelete() {
    const ids = pendingDeleteIds;
    setPendingDeleteIds([]);
    const { needsConfirmationIds } = await requestDelete(ids);
    if (needsConfirmationIds.length > 0) {
      setPendingForceIds(needsConfirmationIds);
    }
  }

  async function confirmForceDelete() {
    const ids = pendingForceIds;
    setPendingForceIds([]);
    await requestDelete(ids, true);
  }

  const pendingNames = buildTargets(pendingDeleteIds).map(
    (t) => t.name ?? t.provider,
  );
  const pendingForceNames = buildTargets(pendingForceIds).map(
    (t) => t.name ?? t.provider,
  );

  if (isLoading) {
    return (
      <div className="flex w-full flex-col gap-3">
        <IntegrationsSearch value={query} onChange={setQuery} disabled />
        <IntegrationsListSkeleton />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex w-full flex-col gap-3">
        <IntegrationsSearch value={query} onChange={setQuery} disabled />
        <ErrorCard
          context="integrations"
          responseError={
            error instanceof Error ? { message: error.message } : undefined
          }
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  return (
    <div className="flex w-full flex-col gap-3">
      <div className="sticky top-0 z-10 -mx-1 bg-[#F9F9FA] px-1 pb-1 pt-1">
        <IntegrationsSearch value={query} onChange={setQuery} />
      </div>

      <AnimatePresence initial={false}>
        {selection.selectedCount > 0 && (
          <motion.div
            key="integrations-selection-bar"
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
            className="sticky top-2 z-20 bg-[#F9F9FA] sm:top-0"
            style={{ overflow: "hidden" }}
          >
            <IntegrationsSelectionBar
              selectedCount={selection.selectedCount}
              allSelected={selection.allSelected}
              onSelectAll={selection.selectAll}
              onDeselectAll={selection.clear}
              onDeleteSelected={() => askDelete(selection.selectedIds)}
              isDeleting={isDeleting}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {isEmpty ? (
        <IntegrationsListEmpty query={query} />
      ) : (
        <motion.div
          className="flex flex-col gap-3 pb-4"
          initial={reduceMotion ? false : "hidden"}
          animate={reduceMotion ? undefined : "show"}
          variants={reduceMotion ? undefined : LIST_CONTAINER_VARIANTS}
        >
          {providers.map((provider) => (
            <motion.div
              key={provider.id}
              variants={
                reduceMotion ? REDUCED_MOTION_ITEM_VARIANTS : LIST_ITEM_VARIANTS
              }
            >
              <ProviderGroup
                provider={provider}
                isSelected={selection.isSelected}
                onToggleSelected={selection.toggle}
                onDelete={(id) => askDelete([id])}
                isDeletingId={isDeletingId}
              />
            </motion.div>
          ))}
        </motion.div>
      )}

      <DeleteConfirmDialog
        open={pendingDeleteIds.length > 0}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteIds([]);
        }}
        itemNames={pendingNames}
        isPending={isDeleting}
        onConfirm={confirmDelete}
      />

      <DeleteConfirmDialog
        variant="force"
        open={pendingForceIds.length > 0}
        onOpenChange={(open) => {
          if (!open) setPendingForceIds([]);
        }}
        itemNames={pendingForceNames}
        isPending={isDeleting}
        onConfirm={confirmForceDelete}
      />
    </div>
  );
}
