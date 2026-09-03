"use client";

import {
  AiBrain01Icon,
  FlashIcon,
  LockIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

import type { CopilotLlmModel } from "../../../../store";
import { nextRovingValue, rovingTabIndex } from "./radioKeys";
import { Swap } from "./Swap";

interface Segment {
  tier: CopilotLlmModel;
  /** "Balanced · Claude Sonnet 5" — the whole choice, for the accessible name. */
  label: string;
  /** "Balanced" — the tier on its own, for the row's title line. */
  name?: string;
  /** "Claude Sonnet 5" — the model, on the line under the tier. */
  model?: string | null;
  lock?: { reason: string; href: string | null };
}

interface Props {
  segments: Segment[];
  value: CopilotLlmModel;
  onSelect: (tier: CopilotLlmModel) => void;
}

/**
 * The tiers as a stacked list, matching the connection rows above it.
 *
 * A segmented control put the two side by side, which halved the width each
 * tier had to name its model in and made the section read as a different kind
 * of control from the one directly above. Stacked, both sections are the same
 * list of rows, and the tick is the only thing saying which one is live.
 */
function tierIcon(tier: CopilotLlmModel) {
  return tier === "advanced" ? AiBrain01Icon : FlashIcon;
}

export function TierToggle({ segments, value, onSelect }: Props) {
  const options = segments.map((segment) => ({
    value: segment.tier,
    disabled: Boolean(segment.lock),
  }));

  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const to = nextRovingValue(options, value, event.key);
    if (to === null) return;
    event.preventDefault();
    onSelect(to);
    // Selection follows focus in a radio group, so the newly chosen segment
    // is also where focus belongs.
    const group = event.currentTarget;
    const next = group.querySelector<HTMLElement>(`[data-tier="${to}"]`);
    next?.focus();
  }

  return (
    <div
      role="radiogroup"
      aria-label="Model tier"
      onKeyDown={handleKeyDown}
      className="divide-y divide-neutral-200"
    >
      {segments.map((segment) =>
        segment.lock ? (
          <LockedSegment key={segment.tier} segment={segment} />
        ) : (
          <button
            key={segment.tier}
            type="button"
            role="radio"
            aria-checked={segment.tier === value}
            data-tier={segment.tier}
            tabIndex={rovingTabIndex(options, { value: segment.tier }, value)}
            aria-label={segment.label}
            onClick={() => onSelect(segment.tier)}
            className={cn(
              "flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors",
              "focus-visible:bg-neutral-50 focus-visible:outline-none",
            )}
          >
            <Icon
              icon={tierIcon(segment.tier)}
              size={18}
              aria-hidden
              className="flex-none text-zinc-500"
            />
            <span className="flex min-w-0 flex-1 flex-col">
              <span className="truncate text-sm font-medium text-zinc-900">
                {segment.name ?? segment.label}
              </span>
              {segment.model && (
                <span className="text-sm leading-snug text-zinc-500">
                  <Swap className="max-w-full truncate">{segment.model}</Swap>
                </span>
              )}
            </span>
            <Icon
              icon={Tick02Icon}
              size={16}
              aria-hidden
              className={cn(
                "flex-none text-primary transition-opacity",
                segment.tier === value ? "opacity-100" : "opacity-0",
              )}
            />
          </button>
        ),
      )}
    </div>
  );
}

/**
 * A tier the plan excludes. It remains a disabled radio so assistive
 * technology receives a complete account of the radiogroup, while the line
 * below the name has room to explain the barrier.
 */
function LockedSegment({ segment }: { segment: Segment }) {
  return (
    /* The upgrade link is a sibling of the radio, not a child of it: a radio's
       descendants are presentational to assistive technology, so a link nested
       inside one can lose its link semantics and strand the one action the row
       exists to offer. */
    <div className="flex flex-col px-3 py-2.5">
      <span
        role="radio"
        aria-checked={false}
        aria-disabled="true"
        aria-label={`${segment.label} — ${segment.lock?.reason ?? "unavailable"}`}
        tabIndex={-1}
        className="flex items-start gap-2.5"
      >
        <Icon
          icon={LockIcon}
          size={14}
          aria-hidden
          className="mt-[3px] flex-none text-zinc-400"
        />
        <span className="flex min-w-0 flex-col">
          <span className="truncate text-sm font-medium text-zinc-500">
            {segment.name ?? segment.label}
          </span>
          {segment.model && (
            <span className="text-sm leading-snug text-zinc-400">
              <Swap className="max-w-full truncate">{segment.model}</Swap>
            </span>
          )}
          {/* A row has room for the lock but the name has none for why. Saying
              only "locked" would leave the user to guess at a barrier they can
              actually clear. */}
          <span className="text-[11px] leading-snug text-zinc-400">
            {segment.lock?.reason}
          </span>
        </span>
      </span>
      {segment.lock?.href && (
        <Link
          href={segment.lock.href}
          // Indented past the lock glyph so it lines up under the reason it
          // answers, the way it reads when it follows that sentence inline.
          className="ml-6 mt-1 w-fit text-[11px] font-medium text-zinc-900 underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
        >
          See plans
        </Link>
      )}
    </div>
  );
}
