"use client";

import { cn } from "@/lib/utils";
import type { MutableRefObject, ReactNode } from "react";
import type { TranscriptRow } from "./helpers";
import {
  useVirtualTranscript,
  type JumpToMessage,
} from "./useVirtualTranscript";

interface Props {
  rows: TranscriptRow[];
  renderRow: (row: TranscriptRow) => ReactNode;
  lastUserMessageID: string | null;
  compact: boolean;
  /** Set to a function that scrolls a row into view by message ID, for rows
   *  the window has unmounted. */
  jumpRef?: MutableRefObject<JumpToMessage | null>;
}

/** The transcript's rows, windowed: only the rows on screen, a few either
 *  side and the newest turn are mounted. Rows keep `key={message.id}`, so a
 *  row's state survives for as long as it stays in the window. */
export function VirtualTranscript({
  rows,
  renderRow,
  lastUserMessageID,
  compact,
  jumpRef,
}: Props) {
  const { listRef, measureRow, segments, padding, isWindowed, isQuiet } =
    useVirtualTranscript({ rows, lastUserMessageID, compact, jumpRef });

  return (
    <div
      ref={listRef}
      className={cn("flex flex-col", compact ? "gap-4" : "gap-6")}
      style={{
        paddingTop: padding.top || undefined,
        paddingBottom: padding.bottom || undefined,
        overflowAnchor: isWindowed ? "none" : undefined,
      }}
    >
      {segments.map((segment) => {
        if (segment.kind === "gap") {
          return (
            <div
              key={`gap-${segment.afterIndex}`}
              aria-hidden
              style={{ height: segment.height }}
            />
          );
        }
        const row = rows[segment.index];
        const id = row.message.id;
        return (
          <div
            key={id}
            ref={measureRow}
            data-index={segment.index}
            data-key={id}
            className={cn("flex flex-col", isQuiet(id) && "[&>*]:animate-none")}
          >
            {renderRow(row)}
          </div>
        );
      })}
    </div>
  );
}
