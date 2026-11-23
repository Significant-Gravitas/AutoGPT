"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import Link from "next/link";
import type { ReactNode } from "react";
import {
  CARD,
  FOCUS,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";

interface RowProps {
  /** The small square at the row's start: an icon, a logo, a status mark. */
  tile: ReactNode;
  title: string;
  /** A second, quieter line. */
  subtitle?: string | null;
  /** Lets the subtitle run to two lines instead of one. */
  wrap?: boolean;
  trailing?: ReactNode;
  onClick?: () => void;
  href?: string;
  /** The href leaves the app: it opens in a new tab. */
  external?: boolean;
  disabled?: boolean;
}

/** One row of a block's card: a tile, what it is, an optional second line
 *  and something quiet at the end. With `onClick` or `href` the whole row
 *  opens something. Rows after the first are split by a hairline that starts
 *  after the tile, as in the expert's Profile panel. */
export function BlockRow({
  tile,
  title,
  subtitle,
  wrap = false,
  trailing,
  onClick,
  href,
  external = false,
  disabled = false,
}: RowProps) {
  const body = (
    <>
      <span className="grid size-[26px] shrink-0 place-items-center overflow-hidden rounded-md bg-surface-tray text-foreground-text">
        {tile}
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <Text variant="inbox" as="span" tone="ink" className="truncate">
          {title}
        </Text>
        {subtitle ? (
          <Text
            variant="small"
            as="span"
            tone="quiet"
            className={wrap ? "line-clamp-2 text-pretty" : "truncate"}
          >
            {subtitle}
          </Text>
        ) : null}
      </span>
      {trailing}
    </>
  );
  const rowClass = cn(
    "flex w-full items-center gap-2.5 rounded-lg px-2 text-left",
    subtitle ? "min-h-12 py-1.5" : "h-10",
  );
  const interactive = cn(
    rowClass,
    "transition-colors duration-100 ease-out hover:bg-surface-tray",
    FOCUS,
  );

  return (
    <li className="relative not-first:before:absolute not-first:before:-top-px not-first:before:right-2 not-first:before:left-12 not-first:before:h-px not-first:before:bg-border-hairline">
      {href && external ? (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={interactive}
        >
          {body}
        </a>
      ) : href ? (
        <Link href={href} className={interactive}>
          {body}
        </Link>
      ) : onClick && !disabled ? (
        <button type="button" onClick={onClick} className={interactive}>
          {body}
        </button>
      ) : (
        <div className={cn(rowClass, disabled && "opacity-60")}>{body}</div>
      )}
    </li>
  );
}

interface NoteProps {
  children: ReactNode;
  className?: string;
}

/** A block's card holding free content instead of rows. */
export function BlockNote({ children, className }: NoteProps) {
  return (
    <div className={cn(CARD, "flex flex-col gap-1.5 p-3", className)}>
      {children}
    </div>
  );
}

/** The amber "Needs you" tag. */
export function NeedsYouTag({ children = "Needs you" }: { children?: string }) {
  return (
    <span className="w-fit shrink-0 rounded-sm bg-status-waiting/15 px-1.5 text-caption font-medium text-warning-foreground">
      {children}
    </span>
  );
}
