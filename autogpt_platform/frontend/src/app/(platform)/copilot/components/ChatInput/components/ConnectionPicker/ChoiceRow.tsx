"use client";

import { cn } from "@/lib/utils";
import Link from "next/link";
import { LockIcon, Tick02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";

interface Props {
  /** The mark that identifies the row at a glance — a provider logo, a glyph. */
  leading?: React.ReactNode;
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
  /** Shown beside the title, e.g. "Connected". */
  badge?: string;
  /** Why this cannot be chosen, and where to go to change that. */
  lock?: { reason: string; href: string | null };
  /** Identifies the row to the group's arrow-key handler. */
  offerId?: string;
  /**
   * A radio group is one tab stop; only the active row takes it. Supplied by
   * the group so Tab skips past the whole set rather than through it.
   */
  tabIndex?: number;
}

export function ChoiceRow({
  leading,
  title,
  subtitle,
  notes,
  isSelected,
  onSelect,
  label,
  lock,
  badge,
  offerId,
  tabIndex,
}: Props) {
  if (lock) {
    return (
      <LockedRow title={title} subtitle={subtitle} notes={notes} lock={lock} />
    );
  }

  return (
    /* The row is a container rather than the radio itself, so the info mark can
       sit on it without sitting *inside* it: a radio's descendants are
       presentational to assistive technology, and an SVG takes no focus either
       way, so a tooltip nested in the radio was unreachable by keyboard. The
       container carries the hover and focus wash so the row still lights up as
       one thing. */
    <div
      className={cn(
        "flex w-full items-center gap-3 px-3 py-2.5 transition-colors",
        "has-[button:focus-visible]:bg-neutral-50 hover:bg-neutral-50",
      )}
    >
      <button
        type="button"
        role="radio"
        aria-checked={isSelected}
        aria-label={label}
        data-offer={offerId}
        tabIndex={tabIndex}
        onClick={onSelect}
        className="flex min-w-0 flex-1 items-center gap-3 text-left focus-visible:outline-none"
      >
        {leading && (
          <span aria-hidden className="flex-none">
            {leading}
          </span>
        )}
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="flex items-center gap-1.5">
            <span className="text-sm font-medium text-zinc-900">{title}</span>
            {badge && (
              <span className="rounded-full bg-green-500/10 px-1.5 py-px text-[10px] font-medium text-green-700">
                {badge}
              </span>
            )}
          </span>
          {subtitle && (
            <span className="break-words text-sm leading-snug text-zinc-500">
              {subtitle}
            </span>
          )}
        </span>
      </button>
      {/* What applies to a route matters once, when you are weighing it.
          Spelled out on the row it competes with the choice itself, so it
          waits behind the info mark. */}
      {notes && notes.length > 0 && (
        <InformationTooltip description={notes.join("\n\n")} iconSize={22} />
      )}
      {/* The tick alone says which one is live: a row that is not chosen needs
          no marker of its own, and an empty circle beside every option is one
          more thing to read past. */}
      <Icon
        icon={Tick02Icon}
        size={16}
        aria-hidden
        className={cn(
          "flex-none text-primary transition-opacity",
          isSelected ? "opacity-100" : "opacity-0",
        )}
      />
    </div>
  );
}

interface LockedProps {
  title: string;
  subtitle?: string;
  notes?: string[];
  lock: { reason: string; href: string | null };
}

/**
 * Not a radio: this is not a choice, and offering it as one would invite a
 * click that cannot do anything. It states what the connection is, why it is
 * unavailable, and the one action that changes that.
 */
function LockedRow({ title, subtitle, notes, lock }: LockedProps) {
  return (
    <div className="flex items-start gap-2.5 px-3 py-2.5">
      <Icon
        icon={LockIcon}
        size={14}
        aria-hidden
        className="mt-[3px] flex-none text-zinc-400"
      />
      <span className="flex min-w-0 flex-col">
        <span className="text-sm font-medium text-zinc-500">{title}</span>
        {subtitle && (
          <span className="break-words text-sm leading-snug text-zinc-400">
            {subtitle}
          </span>
        )}
        {notes?.map((note) => (
          <span
            key={note}
            className="mt-0.5 text-[11px] leading-snug text-zinc-400"
          >
            {note}
          </span>
        ))}
        <span className="mt-0.5 text-[11px] leading-snug text-zinc-400">
          {lock.reason}
        </span>
        {lock.href && (
          <Link
            href={lock.href}
            className="mt-1 w-fit text-[11px] font-medium text-zinc-900 underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
          >
            See plans
          </Link>
        )}
      </span>
    </div>
  );
}
