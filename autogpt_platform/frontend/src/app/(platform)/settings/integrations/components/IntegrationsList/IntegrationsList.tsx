"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { IntegrationsListEmpty } from "../IntegrationsListEmpty/IntegrationsListEmpty";
import { IntegrationsSearch } from "../IntegrationsSearch/IntegrationsSearch";
import { IntegrationsSelectionBar } from "../IntegrationsSelectionBar/IntegrationsSelectionBar";
import { ProviderGroup } from "../ProviderGroup/ProviderGroup";
import { IntegrationsListSkeleton } from "./IntegrationsListSkeleton";
import { useIntegrationsList } from "./useIntegrationsList";

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
  } = useIntegrationsList();
  const reduceMotion = useReducedMotion();

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
      <IntegrationsSearch value={query} onChange={setQuery} />

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
            className="sticky top-0 z-20 bg-[#F9F9FA]"
            style={{ overflow: "hidden" }}
          >
            <IntegrationsSelectionBar
              selectedCount={selection.selectedCount}
              allSelected={selection.allSelected}
              onSelectAll={selection.selectAll}
              onDeselectAll={selection.clear}
              onDeleteSelected={() =>
                requestDelete([...selection.selectedIds])
              }
            />
          </motion.div>
        )}
      </AnimatePresence>

      {isEmpty ? (
        <IntegrationsListEmpty query={query} />
      ) : (
        <div className="flex flex-col gap-3">
          {providers.map((provider) => (
            <ProviderGroup
              key={provider.id}
              provider={provider}
              isSelected={selection.isSelected}
              onToggleSelected={selection.toggle}
              onDelete={(id) => requestDelete([id])}
            />
          ))}
        </div>
      )}
    </div>
  );
}
