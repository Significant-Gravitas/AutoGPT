import { findSegmentCutPoints } from "@/lib/text-segments";
import {
  createTextPacer,
  defaultNow,
  defaultTickMs,
  isDocumentHidden,
  type TextPacer,
} from "../copilotStreamPacer";
import type { BlockState, LogRow, TurnLog } from "./turnLog";
import type { Segment } from "./turnTail";

export interface TextRevealOptions {
  now?: () => number;
  isHidden?: () => boolean;
  tickMs?: number;
}

interface Revealing {
  shown: number;
  pacer: TextPacer;
}

/**
 * Paces the text the live POST stream writes: what is shown lags what is
 * applied, so the cursor and the turn log stay exact. It reveals by elapsed
 * time with the AI SDK path's rules (`copilotStreamPacer.ts`): a steady rate,
 * at most `MAX_LAG_MS` behind, whole words (or dictionary words in unspaced
 * scripts), all at once in a hidden tab. A block that ends keeps revealing
 * at its pace and reads as still streaming until it is shown whole.
 */
export class TextReveal {
  private revealing = new Map<string, Revealing>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private displayRows = new WeakMap<LogRow, { shown: number; row: LogRow }>();
  private readonly now: () => number;
  private readonly isHidden: () => boolean;
  private readonly tickMs: number;

  constructor(
    private readonly segments: () => readonly Segment[],
    private readonly onTick: () => void,
    options: TextRevealOptions = {},
  ) {
    this.now = options.now ?? defaultNow;
    this.isHidden = options.isHidden ?? isDocumentHidden;
    this.tickMs = options.tickMs ?? defaultTickMs();
  }

  /** Rows a live block writes into show from where they stood before this entry. */
  track(prev: TurnLog, next: TurnLog) {
    const now = this.now();
    for (const block of Object.values(next.blocks)) {
      if (!block.open || block.row === null) continue;
      const row = next.rows[block.row];
      const before = prev.rows[block.row]?.content.length ?? 0;
      let entry = this.revealing.get(row.key);
      if (!entry) {
        entry = { shown: before, pacer: createTextPacer() };
        entry.pacer.begin(now);
        this.revealing.set(row.key, entry);
      }
      if (row.content.length > before) {
        entry.pacer.arrive(row.content.length - before, now);
      }
    }
    if (this.revealing.size > 0 && !this.timer) {
      this.timer = setInterval(() => this.tick(), this.tickMs);
    }
  }

  /** Show everything: a resumed stream or a stop does not pace. */
  clear() {
    this.revealing.clear();
  }

  /** The segment as shown: rows still revealing are cut where the reveal stands. */
  display(seg: Segment): Segment {
    if (seg.kind !== "turn" || this.revealing.size === 0) return seg;
    const cutRows = new Set<number>();
    const rows = seg.log.rows.map((row, index) => {
      const shown = this.revealing.get(row.key)?.shown;
      if (shown === undefined || shown >= row.content.length) return row;
      cutRows.add(index);
      const cached = this.displayRows.get(row);
      if (cached?.shown === shown) return cached.row;
      const display = { ...row, content: row.content.slice(0, shown) };
      this.displayRows.set(row, { shown, row: display });
      return display;
    });
    if (cutRows.size === 0) return seg;
    const blocks = stillWriting(seg.log.blocks, cutRows);
    return { ...seg, log: { ...seg.log, rows, blocks } };
  }

  dispose() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private tick() {
    const rows = turnRows(this.segments());
    const now = this.now();
    let moved = false;
    for (const [key, entry] of this.revealing) {
      const found = rows.get(key);
      const rest = found?.row.content.slice(entry.shown) ?? "";
      const cuts = findSegmentCutPoints(rest);
      // An ended block has no word still growing: its last one is due too.
      if (found && !found.open && rest && cuts.at(-1) !== rest.length) {
        cuts.push(rest.length);
      }
      const last = cuts.at(-1) ?? 0;
      const cut = this.isHidden() ? last : entry.pacer.take(cuts, now);
      if (cut > 0) {
        entry.shown += cut;
        moved = true;
      }
      if (!found || (!found.open && entry.shown >= found.row.content.length)) {
        this.revealing.delete(key);
        moved = true;
      }
    }
    if (this.revealing.size === 0) this.dispose();
    if (moved) this.onTick();
  }
}

function turnRows(segments: readonly Segment[]) {
  const rows = new Map<string, { row: LogRow; open: boolean }>();
  for (const seg of segments) {
    if (seg.kind !== "turn") continue;
    const open = new Set(
      Object.values(seg.log.blocks)
        .filter((block) => block.open)
        .map((block) => block.row),
    );
    seg.log.rows.forEach((row, index) => {
      rows.set(row.key, { row, open: open.has(index) });
    });
  }
  return rows;
}

// A row still being revealed reads as streaming, whether or not its block ended.
function stillWriting(
  blocks: Readonly<Record<string, BlockState>>,
  cutRows: ReadonlySet<number>,
) {
  let changed = false;
  const next: Record<string, BlockState> = {};
  for (const [id, block] of Object.entries(blocks)) {
    const reopen = !block.open && block.row !== null && cutRows.has(block.row);
    next[id] = reopen ? { ...block, open: true } : block;
    changed ||= reopen;
  }
  return changed ? next : blocks;
}
