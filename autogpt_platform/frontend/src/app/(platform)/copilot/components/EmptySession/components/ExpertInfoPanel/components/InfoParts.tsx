import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import type { Tone } from "@/components/styles/tones";
import { cn } from "@/lib/utils";
import Link from "next/link";
import type { ReactNode } from "react";

export const CARD = "rounded-panel bg-surface shadow-(--inbox-shadow-card)";
export const FOCUS =
  "focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none";

interface GroupProps {
  label: string;
  count?: number;
  action?: ReactNode;
  children: ReactNode;
}

export function InfoGroup({ label, count, action, children }: GroupProps) {
  return (
    <section aria-label={label} className="flex flex-col gap-2">
      <div className="flex h-4 items-center justify-between px-1">
        <Text variant="small" weight="medium" as="h3" tone="quiet">
          {label}
          {count !== undefined && count > 0 && (
            <span className="ml-1.5 font-normal text-foreground-faint tabular-nums">
              {count}
            </span>
          )}
        </Text>
        {action}
      </div>
      {children}
    </section>
  );
}

export function InfoList({ children }: { children: ReactNode }) {
  return <ul className={cn(CARD, "flex flex-col p-1")}>{children}</ul>;
}

interface RowProps {
  tile: ReactNode;
  title: string;
  subtitle?: string;
  trailing?: ReactNode;
  href?: string;
}

export function InfoRow({ tile, title, subtitle, trailing, href }: RowProps) {
  const body = (
    <>
      <span className="grid size-[26px] shrink-0 place-items-center rounded-md bg-surface-tray text-foreground-text">
        {tile}
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <Text variant="inbox" as="span" tone="ink" className="truncate">
          {title}
        </Text>
        {subtitle && (
          <Text variant="small" as="span" tone="quiet" className="truncate">
            {subtitle}
          </Text>
        )}
      </span>
      {trailing}
    </>
  );
  const rowClass = cn(
    "flex items-center gap-2.5 rounded-lg px-2",
    subtitle ? "min-h-12 py-1.5" : "h-10",
  );

  return (
    <li className="relative not-first:before:absolute not-first:before:-top-px not-first:before:right-2 not-first:before:left-12 not-first:before:h-px not-first:before:bg-border-hairline">
      {href ? (
        <Link
          href={href}
          className={cn(
            rowClass,
            "transition-colors hover:bg-surface-tray",
            FOCUS,
          )}
        >
          {body}
        </Link>
      ) : (
        <div className={rowClass}>{body}</div>
      )}
    </li>
  );
}

export function InfoPill({
  children,
  tone = "text",
}: {
  children: ReactNode;
  tone?: Tone;
}) {
  return (
    <span className="inline-flex h-5 shrink-0 items-center rounded-control bg-surface-wash px-1.5">
      <Text variant="small" weight="medium" as="span" tone={tone}>
        {children}
      </Text>
    </span>
  );
}

export function InfoMeta({ children }: { children: ReactNode }) {
  return (
    <Text variant="small" as="span" tone="quiet" className="shrink-0">
      {children}
    </Text>
  );
}

export function InfoEmpty({ children }: { children: ReactNode }) {
  return (
    <div className={cn(CARD, "px-3 py-3")}>
      <Text variant="inbox" as="p" tone="quiet">
        {children}
      </Text>
    </div>
  );
}

export function InfoLoading() {
  return (
    <div className={cn(CARD, "flex items-center gap-2.5 p-3")}>
      <Skeleton className="size-[26px] rounded-md" />
      <div className="flex flex-1 flex-col gap-1.5">
        <Skeleton className="h-3.5 w-2/3" />
        <Skeleton className="h-3 w-1/3" />
      </div>
    </div>
  );
}
