"use client";

import {
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  GlobeIcon,
  LinkSquare01Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { safeHostname } from "./resultHelpers";

interface StatusPillProps {
  status: string;
  className?: string;
}

interface StatusCardProps {
  label: string;
  ok: boolean;
}

interface StatCardProps {
  value: number;
  label: string;
}

interface ChipListProps {
  label: string;
  items: string[];
}

interface LinkCardProps {
  url: string;
  title?: string;
  meta?: string;
  secret?: boolean;
}

export const CARD = "rounded-xl bg-surface ring-1 ring-border-base/70";
export const HALF = "w-full sm:w-1/2";
export const RESULT_GRID =
  // TODO(tokens): no fitting role for scrollbar-thumb-zinc-300 #C5C5C9; #C5C5C9 is foreground-disabled (a text colour) and border-strong (a line colour)
  "grid max-h-64 gap-1.5 overflow-y-auto pr-0.5 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300 sm:grid-cols-2";

const STATUS_STYLES: Record<string, string> = {
  COMPLETED: "bg-success-surface text-success",
  FAILED: "bg-error-surface text-error",
  // TODO(tokens): no exact role for bg-purple-50; no fitting role for text-purple-600 #6C2EDF; #6C2EDF is brand-hover (a fill)
  RUNNING: "bg-purple-50 text-purple-600",
  // TODO(tokens): no exact role for text-amber-600
  QUEUED: "bg-warning-surface text-amber-600",
};

export function StatusPill({ status, className }: StatusPillProps) {
  const normalized = status.toUpperCase();
  const isPending = normalized === "RUNNING" || normalized === "QUEUED";
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-caption-sm font-medium leading-inherit",
        STATUS_STYLES[normalized] ??
          "bg-surface-hover text-foreground-tertiary",
        className,
      )}
    >
      {isPending && (
        <span className="size-2.5 animate-spin rounded-full border border-current border-t-transparent motion-reduce:animate-none" />
      )}
      {normalized.toLowerCase()}
    </span>
  );
}

export function StatusCard({ label, ok }: StatusCardProps) {
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2 p-2.5`}>
      {ok ? (
        <Icon
          icon={CheckmarkCircle02Icon}
          size={16}
          // TODO(tokens): no exact role for text-green-500
          className="shrink-0 text-green-500"
        />
      ) : (
        <Icon
          icon={CancelCircleIcon}
          size={16}
          // TODO(tokens): no exact role for text-red-400
          className="shrink-0 text-red-400"
        />
      )}
      <p className="min-w-0 truncate text-body-sm font-medium leading-inherit text-foreground-body">
        {label}
      </p>
    </div>
  );
}

export function StatCard({ value, label }: StatCardProps) {
  return (
    <div className={`${CARD} ${HALF} flex items-baseline gap-2 p-2.5`}>
      <span className="text-lg font-semibold leading-none text-foreground-emphasis">
        {value.toLocaleString()}
      </span>
      <span className="min-w-0 truncate text-xs text-foreground-tertiary">
        {label}
      </span>
    </div>
  );
}

export function ChipList({ label, items }: ChipListProps) {
  return (
    <div className={`${CARD} ${HALF} p-2.5`}>
      <p className="mb-1.5 text-caption-sm uppercase leading-inherit tracking-wide text-foreground-subtle">
        {label}
      </p>
      <div className="flex flex-wrap gap-1">
        {Array.from(new Set(items)).map((item) => (
          <span
            key={item}
            className="max-w-full truncate rounded-full bg-surface-hover px-2 py-0.5 text-xs text-foreground-secondary"
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

export function LinkCard({ url, title, meta, secret }: LinkCardProps) {
  const domain = safeHostname(url) ?? url;

  return (
    <div
      className={cn(
        CARD,
        HALF,
        "flex items-center gap-2.5 p-2.5",
        secret && "ph-no-capture",
      )}
    >
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
        <Icon
          icon={GlobeIcon}
          size={15}
          className="shrink-0 text-foreground-subtle"
        />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
          {title ?? domain}
        </p>
        <p className="truncate text-xs text-foreground-tertiary">
          {title ? domain : url}
        </p>
      </div>
      {meta && (
        <span className="shrink-0 text-xs text-foreground-subtle">{meta}</span>
      )}
      <Link
        href={url}
        target="_blank"
        rel="noreferrer"
        aria-label="Open link"
        className="shrink-0 rounded-full p-1 text-foreground-subtle transition-colors hover:bg-surface-hover hover:text-foreground-body"
      >
        <Icon icon={LinkSquare01Icon} size={14} />
      </Link>
    </div>
  );
}
