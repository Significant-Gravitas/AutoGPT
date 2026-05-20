"use client";

import { motion, useReducedMotion } from "framer-motion";
import { CircleNotchIcon } from "@phosphor-icons/react";

import type { Pagination as PaginationModel } from "@/app/api/__generated__/models/pagination";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import type { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import {
  EASE_OUT,
  type FilterState,
  type SortDir,
  type SortKey,
} from "../../helpers";
import { MobileSubmissionItem } from "../MobileSubmissionItem/MobileSubmissionItem";
import { Pagination } from "../Pagination/Pagination";
import { SortColumnFilter } from "../SubmissionsList/columns/SortColumnFilter";
import { StatusColumnFilter } from "../SubmissionsList/columns/StatusColumnFilter";

interface EditPayload extends StoreSubmissionEditRequest {
  store_listing_version_id: string | undefined;
  graph_id: string;
}

interface Props {
  submissions: StoreSubmission[];
  totalCount: number;
  pagination?: PaginationModel;
  onPageChange?: (page: number) => void;
  isFetching?: boolean;
  filterState: FilterState;
  onFilterChange: (next: FilterState) => void;
  onResetFilters: () => void;
  onView: (submission: StoreSubmission) => void;
  onEdit: (payload: EditPayload) => void;
  onDelete: (submissionId: string) => Promise<void>;
  creatorUsername?: string;
  index?: number;
}

export function MobileSubmissionsList({
  submissions,
  totalCount,
  pagination,
  onPageChange,
  isFetching,
  filterState,
  onFilterChange,
  onResetFilters,
  onView,
  onEdit,
  onDelete,
  creatorUsername,
  index = 0,
}: Props) {
  const reduceMotion = useReducedMotion();

  function setStatuses(statuses: SubmissionStatus[]) {
    onFilterChange({ ...filterState, statuses });
  }

  function setSort(sortKey: SortKey | null, sortDir: SortDir) {
    onFilterChange({ ...filterState, sortKey, sortDir });
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
        <div className="flex items-center gap-2">
          <Text variant="body-medium" as="span" className="text-textBlack">
            Submissions
          </Text>
          {isFetching ? (
            <span
              role="status"
              aria-live="polite"
              className="inline-flex items-center gap-1 text-zinc-500"
              data-testid="submissions-fetching"
            >
              <CircleNotchIcon
                size={14}
                weight="bold"
                className="animate-spin"
              />
            </span>
          ) : null}
        </div>
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
      </div>

      <div className="w-full min-w-0 max-w-full overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        {submissions.length > 0 ? (
          submissions.map((submission, rowIndex) => (
            <MobileSubmissionItem
              key={submission.listing_version_id}
              submission={submission}
              rowIndex={rowIndex}
              onView={onView}
              onEdit={onEdit}
              onDelete={onDelete}
              creatorUsername={creatorUsername}
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
        {pagination && onPageChange ? (
          <div className="border-t border-zinc-100">
            <Pagination
              pagination={pagination}
              onPageChange={onPageChange}
              disabled={isFetching}
            />
          </div>
        ) : null}
      </div>
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
