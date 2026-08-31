"use client";

import { LockIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

import type { CopilotLlmModel } from "../../../../store";

interface Segment {
  tier: CopilotLlmModel;
  /** "Balanced · Claude Sonnet 5" — the tier and the model it resolves to. */
  label: string;
  lock?: { reason: string; href: string | null };
}

interface Props {
  segments: Segment[];
  value: CopilotLlmModel;
  onSelect: (tier: CopilotLlmModel) => void;
}

/**
 * The two tiers side by side, as one control rather than a list of choices.
 *
 * A list reads as "here are some options"; a segmented control reads as "it is
 * one of these two", which is the shape of the decision — there is no third
 * tier, because absorbing that complexity is what having tiers is for.
 *
 * Each segment names its model inline. That only fits because the model now
 * arrives as a display name rather than a routing slug; while a tier read
 * "anthropic/claude-sonnet-5" the label had to wrap onto a second line.
 */
export function TierToggle({ segments, value, onSelect }: Props) {
  const locked = segments.filter((segment) => segment.lock);

  return (
    <div className="mb-3 mt-1 flex flex-col gap-1.5 px-3">
      <div
        role="radiogroup"
        aria-label="Model tier"
        className="flex items-stretch gap-1 rounded-full bg-muted/70 p-1"
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
              onClick={() => onSelect(segment.tier)}
              className={cn(
                "min-w-0 flex-1 truncate rounded-full px-3 py-1.5 text-[11px] font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/40",
                segment.tier === value
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {segment.label}
            </button>
          ),
        )}
      </div>

      {/* A segment has room for the lock but not for why. Saying only "locked"
          would leave the user to guess at a barrier they can actually clear. */}
      {locked.map((segment) => (
        <p
          key={segment.tier}
          className="px-1 text-[11px] leading-snug text-muted-foreground/80"
        >
          {segment.lock?.reason}{" "}
          {segment.lock?.href && (
            <Link
              href={segment.lock.href}
              className="font-medium text-accent underline-offset-2 hover:underline"
            >
              See plans
            </Link>
          )}
        </p>
      ))}
    </div>
  );
}

/**
 * A tier the plan excludes. It remains a disabled radio so assistive
 * technology receives a complete account of the radiogroup, while the line
 * below has room to explain the barrier.
 */
function LockedSegment({ segment }: { segment: Segment }) {
  return (
    <span
      role="radio"
      aria-checked={false}
      aria-disabled="true"
      aria-label={`${segment.label} — ${segment.lock?.reason ?? "unavailable"}`}
      tabIndex={-1}
      className="flex min-w-0 flex-1 items-center justify-center gap-1 rounded-full px-3 py-1.5 text-[11px] font-medium text-muted-foreground/70"
    >
      <Icon icon={LockIcon} size={11} aria-hidden className="flex-none" />
      <span className="truncate">{segment.label}</span>
    </span>
  );
}
