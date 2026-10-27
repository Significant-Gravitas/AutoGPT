"use client";

import type { ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  /** Left, right and centre tile, in that order. */
  tiles: [ReactNode, ReactNode, ReactNode];
  title: string;
  description: ReactNode;
  children?: ReactNode;
  /** A quieter line under the actions, e.g. a link elsewhere. */
  footer?: ReactNode;
  /** `body-md` inside a profile tab, whose copy reads at 15px. */
  textSize?: "body" | "body-md";
}

const TILE_POSITIONS = [
  "left-2 top-3 -rotate-12 group-hover:-translate-x-1.5 group-hover:-rotate-[16deg]",
  "right-2 top-3 rotate-12 group-hover:translate-x-1.5 group-hover:rotate-[16deg]",
  "left-1/2 top-0 z-10 -translate-x-1/2 group-hover:-translate-y-1",
];

/** The side panel's empty state: three white tiles, the outer two tilted out
 *  (they spread further on hover), over a title and a line of copy. */
export function PanelEmptyState({
  tiles,
  title,
  description,
  children,
  footer,
  textSize = "body",
}: Props) {
  return (
    <div className="group flex flex-col items-center px-6 pb-8 pt-10 text-center">
      <div aria-hidden className="relative mb-7 h-24 w-36">
        {tiles.map((tile, index) => (
          <div
            key={index}
            className={cn(
              "absolute flex h-20 w-16 items-center justify-center rounded-xl bg-surface transition-transform duration-300 ease-out smooth-shadow-ring-sm motion-reduce:transition-none",
              TILE_POSITIONS[index],
            )}
          >
            {tile}
          </div>
        ))}
      </div>
      <Text variant="large-medium" className="!text-foreground-strong">
        {title}
      </Text>
      <Text
        variant={textSize}
        className="mt-1.5 max-w-[17rem] !text-foreground-tertiary"
      >
        {description}
      </Text>
      {children && (
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
          {children}
        </div>
      )}
      {footer && <div className="mt-4">{footer}</div>}
    </div>
  );
}
