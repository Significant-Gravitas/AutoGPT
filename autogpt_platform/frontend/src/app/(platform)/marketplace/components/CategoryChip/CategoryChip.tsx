"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import {
  getCategoryAccent,
  type ExpertAccent,
} from "../ExpertsSection/helpers";

export type ChipSize = "small" | "default";

/** The zinc fallback `getCategoryAccent` hands back for "All", research and
 *  development. Its glossy chip is too faint to read as selected next to a
 *  white row, so those chips take a plain fill instead. */
const NEUTRAL_ACCENT = getCategoryAccent(undefined).accent;
const NEUTRAL_SELECTED = "border-transparent bg-zinc-100 text-zinc-900";

interface Props {
  label: string;
  title?: string;
  /** The glyph and accent this category wears everywhere else. Left out for
   *  chips that are not categories, e.g. the hero's search terms. */
  icon?: IconSvgElement | null;
  accent?: ExpertAccent;
  /** "small" tucks a row into a section header; "default" is a page's own
   *  filter. */
  size?: ChipSize;
  isSelected: boolean;
  onClick: () => void;
}

export function CategoryChip({
  label,
  title,
  icon,
  accent = NEUTRAL_ACCENT,
  size = "default",
  isSelected,
  onClick,
}: Props) {
  return (
    // Outline, so a row of filters never reads as a row of actions. The
    // selected one takes the accent's glossy chip, whose gradient sits over
    // the variant's hover wash rather than being replaced by it.
    <Button
      variant="outline"
      size={size === "small" ? "xs" : "small"}
      title={title}
      aria-pressed={isSelected}
      onClick={onClick}
      unmask={false}
      leftIcon={
        icon ? (
          <Icon
            icon={icon}
            size={size === "small" ? 12 : 15}
            className={isSelected ? undefined : accent.icon}
            aria-hidden
          />
        ) : undefined
      }
      className={cn(
        // The variant's #a6a6a6 is for a lone action; a row of chips reads
        // quieter with a lighter edge.
        "min-w-0 rounded-full border-[#e9e9e9] hover:border-[#e9e9e9]",
        size === "small" ? "gap-1 px-2" : "gap-1.5 px-3.5",
        isSelected &&
          (accent === NEUTRAL_ACCENT
            ? NEUTRAL_SELECTED
            : cn(accent.chip, "border-transparent")),
      )}
    >
      {label}
    </Button>
  );
}
