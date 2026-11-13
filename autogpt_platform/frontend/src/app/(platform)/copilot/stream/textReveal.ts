import {
  BACKLOG_DRAIN_TICKS,
  findWordCutPoints,
  TICK_DELAY_MS,
} from "../copilotStreamSmoothing";
import type { LogRow, TurnLog } from "./turnLog";
import type { Segment } from "./turnTail";

const REVEAL_TICK_MS = 30;

/**
 * Paces the text the live POST stream writes, as the smoothing transform does
 * on the AI SDK path: what is shown lags what is applied, so the cursor and
 * the turn log stay exact.
 */
export class TextReveal {
  // Characters shown so far of each row a live block is writing.
  private shown = new Map<string, number>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private displayRows = new WeakMap<LogRow, { shown: number; row: LogRow }>();

  constructor(
    private readonly segments: () => readonly Segment[],
    private readonly onTick: () => void,
  ) {}

  /** Rows a live block writes into show from where they stood before this entry. */
  track(prev: TurnLog, next: TurnLog) {
    for (const block of Object.values(next.blocks)) {
      if (!block.open || block.row === null) continue;
      const row = next.rows[block.row];
      if (!this.shown.has(row.key)) {
        this.shown.set(row.key, prev.rows[block.row]?.content.length ?? 0);
      }
    }
    if (this.shown.size > 0 && !this.timer) {
      this.timer = setInterval(() => this.tick(), REVEAL_TICK_MS);
    }
  }

  /** Show everything: a resumed stream or a stop does not pace. */
  clear() {
    this.shown.clear();
  }

  /** The segment as shown: rows still revealing are cut where the reveal stands. */
  display(seg: Segment): Segment {
    if (seg.kind !== "turn" || this.shown.size === 0) return seg;
    let changed = false;
    const rows = seg.log.rows.map((row) => {
      const shown = this.shown.get(row.key);
      if (shown === undefined || shown >= row.content.length) return row;
      changed = true;
      const cached = this.displayRows.get(row);
      if (cached?.shown === shown) return cached.row;
      const display = { ...row, content: row.content.slice(0, shown) };
      this.displayRows.set(row, { shown, row: display });
      return display;
    });
    return changed ? { ...seg, log: { ...seg.log, rows } } : seg;
  }

  dispose() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  // Each tick shows what the transform would have emitted over it: a share of
  // the backlog per 10 ms, a whole word at least.
  private tick() {
    const open = openRows(this.segments());
    let moved = false;
    for (const [key, shown] of this.shown) {
      const row = open.get(key);
      if (!row) {
        this.shown.delete(key);
        moved = true;
        continue;
      }
      let next = shown;
      for (let t = 0; t < REVEAL_TICK_MS / TICK_DELAY_MS; t++) {
        const cuts = findWordCutPoints(row.content.slice(next));
        if (cuts.length === 0) break;
        const words = Math.max(1, Math.ceil(cuts.length / BACKLOG_DRAIN_TICKS));
        next += cuts[Math.min(words, cuts.length) - 1];
      }
      if (next !== shown) {
        this.shown.set(key, next);
        moved = true;
      }
    }
    if (this.shown.size === 0) this.dispose();
    if (moved) this.onTick();
  }
}

function openRows(segments: readonly Segment[]) {
  const rows = new Map<string, LogRow>();
  for (const seg of segments) {
    if (seg.kind !== "turn") continue;
    for (const block of Object.values(seg.log.blocks)) {
      if (!block.open || block.row === null) continue;
      const row = seg.log.rows[block.row];
      if (row) rows.set(row.key, row);
    }
  }
  return rows;
}
