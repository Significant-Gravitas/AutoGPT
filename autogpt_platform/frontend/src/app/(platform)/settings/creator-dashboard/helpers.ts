import {
  CheckCircleIcon,
  ClockIcon,
  ProhibitIcon,
  WarningCircleIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";

import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { SubmissionStats } from "@/app/api/__generated__/models/submissionStats";

export const EASE_OUT = [0.16, 1, 0.3, 1] as const;

export type StatusFilterValue = "all" | SubmissionStatus;

export type SortKey = "submitted" | "runs";
export type SortDir = "asc" | "desc";

export interface FilterState {
  statuses: SubmissionStatus[];
  nameQuery: string;
  sortKey: SortKey | null;
  sortDir: SortDir;
}

export const INITIAL_FILTER_STATE: FilterState = {
  statuses: [],
  nameQuery: "",
  sortKey: null,
  sortDir: "desc",
};

export const STATUS_OPTIONS: { value: SubmissionStatus; label: string }[] = [
  { value: SubmissionStatus.PENDING, label: "In review" },
  { value: SubmissionStatus.APPROVED, label: "Approved" },
  { value: SubmissionStatus.REJECTED, label: "Needs changes" },
  { value: SubmissionStatus.DRAFT, label: "Draft" },
];

export function applyFiltersAndSort(
  submissions: StoreSubmission[],
  state: FilterState,
): StoreSubmission[] {
  let result: StoreSubmission[] = [...submissions];

  if (state.statuses.length > 0) {
    const set = new Set(state.statuses);
    result = result.filter((s) => set.has(s.status));
  }

  if (state.nameQuery.trim()) {
    const q = state.nameQuery.trim().toLowerCase();
    result = result.filter((s) => s.name.toLowerCase().includes(q));
  }

  if (state.sortKey) {
    const dir = state.sortDir === "asc" ? 1 : -1;
    result = [...result].sort((a, b) => {
      const av = sortValue(a, state.sortKey!);
      const bv = sortValue(b, state.sortKey!);
      if (av === bv) return 0;
      return av < bv ? -1 * dir : 1 * dir;
    });
  }

  return result;
}

function sortValue(submission: StoreSubmission, key: SortKey): number {
  if (key === "submitted") {
    const date = submission.submitted_at;
    if (!date) return 0;
    const time =
      date instanceof Date ? date.getTime() : new Date(date).getTime();
    return Number.isNaN(time) ? 0 : time;
  }
  return submission.run_count ?? 0;
}

export function isFiltered(state: FilterState): boolean {
  return (
    state.statuses.length > 0 ||
    state.nameQuery.trim() !== "" ||
    state.sortKey !== null
  );
}

export interface StatusVisual {
  label: string;
  Icon: PhosphorIcon;
  pillClass: string;
  dotClass: string;
}

export function getStatusVisual(status: SubmissionStatus): StatusVisual {
  return STATUS_VISUAL[status] ?? STATUS_VISUAL[SubmissionStatus.DRAFT];
}

export const STATUS_VISUAL: Record<SubmissionStatus, StatusVisual> = {
  [SubmissionStatus.DRAFT]: {
    label: "Draft",
    Icon: ClockIcon,
    pillClass: "bg-zinc-100 text-zinc-700 ring-1 ring-inset ring-zinc-200",
    dotClass: "bg-zinc-400",
  },
  [SubmissionStatus.PENDING]: {
    label: "In review",
    Icon: ClockIcon,
    pillClass: "bg-amber-50 text-amber-800 ring-1 ring-inset ring-amber-200",
    dotClass: "bg-amber-500",
  },
  [SubmissionStatus.APPROVED]: {
    label: "Approved",
    Icon: CheckCircleIcon,
    pillClass:
      "bg-emerald-50 text-emerald-800 ring-1 ring-inset ring-emerald-200",
    dotClass: "bg-emerald-500",
  },
  [SubmissionStatus.REJECTED]: {
    label: "Needs changes",
    Icon: ProhibitIcon,
    pillClass: "bg-rose-50 text-rose-800 ring-1 ring-inset ring-rose-200",
    dotClass: "bg-rose-500",
  },
};

export const STATUS_FILTERS: { value: StatusFilterValue; label: string }[] = [
  { value: "all", label: "All" },
  { value: SubmissionStatus.PENDING, label: "In review" },
  { value: SubmissionStatus.APPROVED, label: "Approved" },
  { value: SubmissionStatus.REJECTED, label: "Needs changes" },
  { value: SubmissionStatus.DRAFT, label: "Drafts" },
];

export interface DashboardStats {
  total: number;
  approved: number;
  pending: number;
  totalRuns: number;
  averageRating: number | null;
}

export const EMPTY_DASHBOARD_STATS: DashboardStats = {
  total: 0,
  approved: 0,
  pending: 0,
  totalRuns: 0,
  averageRating: null,
};

export function toDashboardStats(
  stats: SubmissionStats | undefined,
): DashboardStats {
  if (!stats) return EMPTY_DASHBOARD_STATS;
  return {
    total: stats.total,
    approved: stats.approved,
    pending: stats.pending,
    totalRuns: stats.total_runs,
    averageRating: stats.average_rating ?? null,
  };
}

export function filterSubmissions(
  submissions: StoreSubmission[],
  filter: StatusFilterValue,
): StoreSubmission[] {
  if (filter === "all") return [...submissions];
  return submissions.filter((submission) => submission.status === filter);
}

export function formatRuns(value: number): string {
  if (value >= 999_950_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 999_950) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toLocaleString();
}

export function formatSubmittedAt(value: Date | null | undefined): string {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export const FILTER_EMPTY_ICON: PhosphorIcon = WarningCircleIcon;
