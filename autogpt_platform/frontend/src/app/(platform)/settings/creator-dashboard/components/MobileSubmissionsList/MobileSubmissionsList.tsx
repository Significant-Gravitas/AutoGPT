"use client";

import { useMemo, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { toast } from "@/components/molecules/Toast/use-toast";

import {
  EASE_OUT,
  type FilterState,
  type SortDir,
  type SortKey,
} from "../../helpers";
import { MobileSubmissionItem } from "../MobileSubmissionItem/MobileSubmissionItem";
import { SortColumnFilter } from "../SubmissionsList/columns/SortColumnFilter";
import { StatusColumnFilter } from "../SubmissionsList/columns/StatusColumnFilter";
import { useSubmissionSelection } from "../SubmissionsList/useSubmissionSelection";
import { MobileSelectionBar } from "./MobileSelectionBar";

interface EditPayload extends StoreSubmissionEditRequest {
  store_listing_version_id: string | undefined;
  graph_id: string;
}

interface Props {
  submissions: StoreSubmission[];
  totalCount: number;
  filterState: FilterState;
  onFilterChange: (next: FilterState) => void;
  onResetFilters: () => void;
  onView: (submission: StoreSubmission) => void;
  onEdit: (payload: EditPayload) => void;
  onDelete: (submissionId: string) => Promise<void>;
  index?: number;
}

export function MobileSubmissionsList({
  submissions,
  totalCount,
  filterState,
  onFilterChange,
  onResetFilters,
  onView,
  onEdit,
  onDelete,
  index = 0,
}: Props) {
  const reduceMotion = useReducedMotion();

  const selectableIds = useMemo(
    () =>
      submissions
        .filter((s) => s.status === SubmissionStatus.PENDING)
        .map((s) => s.listing_version_id),
    [submissions],
  );

  const selection = useSubmissionSelection(selectableIds);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);

  function setStatuses(statuses: SubmissionStatus[]) {
    onFilterChange({ ...filterState, statuses });
  }

  function setSort(sortKey: SortKey | null, sortDir: SortDir) {
    onFilterChange({ ...filterState, sortKey, sortDir });
  }

  async function handleBulkDelete() {
    setIsBulkDeleting(true);
    const ids = [...selection.selectedIds];
    try {
      const results = await Promise.allSettled(ids.map((id) => onDelete(id)));
      const failed = results.filter((r) => r.status === "rejected").length;
      const succeeded = ids.length - failed;
      if (failed > 0) {
        for (const r of results) {
          if (r.status === "rejected") Sentry.captureException(r.reason);
        }
        toast({
          title: `Deleted ${succeeded} of ${ids.length} submissions`,
          description: `${failed} failed to delete. Please try again.`,
          variant: "destructive",
        });
      }
      selection.clear();
      setBulkDeleteOpen(false);
    } finally {
      setIsBulkDeleting(false);
    }
  }

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 0.28, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
      }
      className="flex w-full min-w-0 max-w-full flex-col gap-3"
      data-testid="mobile-submissions-list"
    >
      <div className="flex items-center justify-between gap-3 px-1">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Submissions
        </Text>
        <Text variant="small" className="text-zinc-500">
          {submissions.length} of {totalCount}
        </Text>
      </div>

      <div className="flex flex-wrap items-center gap-2 px-1">
        <FilterChip label="Status">
          <StatusColumnFilter
            value={filterState.statuses}
            onChange={setStatuses}
          />
        </FilterChip>
        <FilterChip label="Date">
          <SortColumnFilter
            sortKey="submitted"
            activeKey={filterState.sortKey}
            activeDir={filterState.sortDir}
            onChange={setSort}
            ascLabel="Oldest first"
            descLabel="Newest first"
          />
        </FilterChip>
        <FilterChip label="Runs">
          <SortColumnFilter
            sortKey="runs"
            activeKey={filterState.sortKey}
            activeDir={filterState.sortDir}
            onChange={setSort}
            ascLabel="Lowest first"
            descLabel="Highest first"
          />
        </FilterChip>
        <FilterChip label="Rating">
          <SortColumnFilter
            sortKey="rating"
            activeKey={filterState.sortKey}
            activeDir={filterState.sortDir}
            onChange={setSort}
            ascLabel="Lowest first"
            descLabel="Highest first"
          />
        </FilterChip>
      </div>

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
            className="overflow-hidden"
          >
            <MobileSelectionBar
              selectedCount={selection.selectedCount}
              allSelected={selection.allSelected}
              onSelectAll={selection.selectAll}
              onDeselectAll={selection.clear}
              onDeleteSelected={() => setBulkDeleteOpen(true)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="w-full min-w-0 max-w-full overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        {submissions.length > 0 ? (
          submissions.map((submission) => (
            <MobileSubmissionItem
              key={submission.listing_version_id}
              submission={submission}
              selected={selection.isSelected(submission.listing_version_id)}
              onToggleSelected={() =>
                selection.toggle(submission.listing_version_id)
              }
              onView={onView}
              onEdit={onEdit}
              onDelete={onDelete}
            />
          ))
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 px-4 py-10 text-center">
            <Text variant="body-medium" className="text-textBlack">
              No submissions match these filters
            </Text>
            <Button variant="secondary" size="small" onClick={onResetFilters}>
              Clear filters
            </Button>
          </div>
        )}
      </div>

      <Dialog
        title="Delete selected submissions?"
        styling={{ maxWidth: "420px" }}
        controlled={{
          isOpen: bulkDeleteOpen,
          set: (open) => setBulkDeleteOpen(open),
        }}
      >
        <Dialog.Content>
          <Text variant="body" className="text-zinc-700">
            This will remove {selection.selectedCount} submission
            {selection.selectedCount === 1 ? "" : "s"} from the store. This
            action cannot be undone.
          </Text>
          <Dialog.Footer>
            <Button
              variant="ghost"
              size="small"
              onClick={() => setBulkDeleteOpen(false)}
              disabled={isBulkDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="small"
              onClick={handleBulkDelete}
              loading={isBulkDeleting}
            >
              {isBulkDeleting ? "Deleting" : "Delete selected"}
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </motion.section>
  );
}

function FilterChip({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="inline-flex items-center gap-1 rounded-full border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-600">
      <span>{label}</span>
      {children}
    </div>
  );
}
