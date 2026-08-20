"use client";

import { cn } from "@/lib/utils";
import Link from "next/link";
import { PiLockSimple as LockIcon } from "react-icons/pi";

interface Props {
  title: string;
  subtitle?: string;
  notes?: string[];
  isSelected: boolean;
  onSelect: () => void;
  /**
   * The accessible name, when the visible title alone would not identify the
   * choice — a tier reads as "Balanced · sonnet-5" while showing the model on
   * its own line, so the name a screen reader announces stays complete.
   */
  label?: string;
  /** Why this cannot be chosen, and where to go to change that. */
  lock?: { reason: string; href: string | null };
}

export function ChoiceRow({
  title,
  subtitle,
  notes,
  isSelected,
  onSelect,
  label,
  lock,
}: Props) {
  if (lock) {
    return <LockedRow title={title} subtitle={subtitle} lock={lock} />;
  }

  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      aria-label={label}
      onClick={onSelect}
      className={cn(
        "flex w-full items-start gap-2.5 px-3 py-2 text-left transition-colors",
        "focus-visible:bg-muted focus-visible:outline-none",
        isSelected ? "bg-muted/60" : "hover:bg-muted/40",
      )}
    >
      <span
        aria-hidden
        className={cn(
          "mt-[3px] flex h-3.5 w-3.5 flex-none items-center justify-center rounded-full border",
          isSelected ? "border-primary" : "border-muted-foreground/50",
        )}
      >
        {isSelected && (
          <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
        )}
      </span>
      <span className="flex min-w-0 flex-col">
        <span className="text-xs font-medium text-foreground">{title}</span>
        {subtitle && (
          <span className="break-words text-[11px] leading-snug text-muted-foreground">
            {subtitle}
          </span>
        )}
        {notes?.map((note) => (
          <span
            key={note}
            className="mt-0.5 text-[11px] leading-snug text-muted-foreground/80"
          >
            {note}
          </span>
        ))}
      </span>
    </button>
  );
}

interface LockedProps {
  title: string;
  subtitle?: string;
  lock: { reason: string; href: string | null };
}

/**
 * Not a radio: this is not a choice, and offering it as one would invite a
 * click that cannot do anything. It states what the connection is, why it is
 * unavailable, and the one action that changes that.
 */
function LockedRow({ title, subtitle, lock }: LockedProps) {
  return (
    <div className="flex items-start gap-2.5 px-3 py-2">
      <LockIcon
        size={14}
        aria-hidden
        className="mt-[3px] flex-none text-muted-foreground/70"
      />
      <span className="flex min-w-0 flex-col">
        <span className="text-xs font-medium text-muted-foreground">
          {title}
        </span>
        {subtitle && (
          <span className="break-words text-[11px] leading-snug text-muted-foreground/80">
            {subtitle}
          </span>
        )}
        <span className="mt-0.5 text-[11px] leading-snug text-muted-foreground/80">
          {lock.reason}
        </span>
        {lock.href && (
          <Link
            href={lock.href}
            className="mt-1 text-[11px] font-medium text-primary underline underline-offset-2"
          >
            See plans
          </Link>
        )}
      </span>
    </div>
  );
}
