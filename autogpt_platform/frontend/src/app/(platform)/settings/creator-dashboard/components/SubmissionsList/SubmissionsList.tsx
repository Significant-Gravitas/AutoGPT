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
import { SubmissionItem } from "../SubmissionItem/SubmissionItem";
import { SubmissionSelectionBar } from "../SubmissionSelectionBar/SubmissionSelectionBar";
import { ColumnHeader } from "./columns/ColumnHeader";
import { SortColumnFilter } from "./columns/SortColumnFilter";
import { StatusColumnFilter } from "./columns/StatusColumnFilter";
import { useSubmissionSelection } from "./useSubmissionSelection";

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

const COLUMN_COUNT = 7;

export function SubmissionsList({
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
      className="flex w-full flex-col gap-3"
      data-testid="submissions-list"
    >
      <div className="flex items-center justify-between pl-4 pr-1">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Submissions
        </Text>
        <Text variant="small" className="text-zinc-500">
          {submissions.length} of {totalCount}
        </Text>
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
            <SubmissionSelectionBar
              selectedCount={selection.selectedCount}
              allSelected={selection.allSelected}
              onSelectAll={selection.selectAll}
              onDeselectAll={selection.clear}
              onDeleteSelected={() => setBulkDeleteOpen(true)}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <div className="overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-zinc-100 bg-zinc-50/60">
                <th
                  scope="col"
                  className="w-[48px] px-3 py-3"
                  aria-label="Select"
                />
                <ColumnHeader label="Agent" />
                <ColumnHeader
                  label="Status"
                  width="140px"
                  filter={
                    <StatusColumnFilter
                      value={filterState.statuses}
                      onChange={setStatuses}
                    />
                  }
                />
                <ColumnHeader
                  label="Submitted"
                  width="160px"
                  filter={
                    <SortColumnFilter
                      sortKey="submitted"
                      activeKey={filterState.sortKey}
                      activeDir={filterState.sortDir}
                      onChange={setSort}
                      ascLabel="Oldest first"
                      descLabel="Newest first"
                    />
                  }
                />
                <ColumnHeader
                  label="Runs"
                  align="right"
                  width="110px"
                  filter={
                    <SortColumnFilter
                      sortKey="runs"
                      activeKey={filterState.sortKey}
                      activeDir={filterState.sortDir}
                      onChange={setSort}
                      ascLabel="Lowest first"
                      descLabel="Highest first"
                    />
                  }
                />
                <ColumnHeader
                  label="Rating"
                  align="right"
                  width="110px"
                  filter={
                    <SortColumnFilter
                      sortKey="rating"
                      activeKey={filterState.sortKey}
                      activeDir={filterState.sortDir}
                      onChange={setSort}
                      ascLabel="Lowest first"
                      descLabel="Highest first"
                    />
                  }
                />
                <th
                  scope="col"
                  className="w-[60px] px-2 py-3"
                  aria-label="Actions"
                />
              </tr>
            </thead>
            <tbody>
              {submissions.length > 0 ? (
                submissions.map((submission) => (
                  <SubmissionItem
                    key={submission.listing_version_id}
                    submission={submission}
                    selected={selection.isSelected(
                      submission.listing_version_id,
                    )}
                    onToggleSelected={() =>
                      selection.toggle(submission.listing_version_id)
                    }
                    onView={onView}
                    onEdit={onEdit}
                    onDelete={onDelete}
                  />
                ))
              ) : (
                <tr>
                  <td colSpan={COLUMN_COUNT} className="px-4 py-12">
                    <div className="flex flex-col items-center justify-center gap-3 text-center">
                      <Text variant="body-medium" className="text-textBlack">
                        No submissions match these filters
                      </Text>
                      <Button
                        variant="secondary"
                        size="small"
                        onClick={onResetFilters}
                      >
                        Clear filters
                      </Button>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
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
