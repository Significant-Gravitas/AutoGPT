import type { ActiveStream } from "../../../api/chat";
import type { Row } from "./rows";
import { seedTurnLog, type TurnLog } from "./turnLog";

export interface ActiveTurnSplit {
  /** True for persisted rows the resumed stream draws itself. */
  belongsToTurn: (row: Row) => boolean;
  seed: TurnLog;
  /** The stream entry to resume after. */
  after: string;
  replayedUserContents: string[];
}

/**
 * Divides loaded history from a turn that is still running, so resuming it
 * neither drops nor duplicates rows.
 *
 * With a checkpoint, the turn's first `rows` rows from `sequence` are already
 * persisted: they seed the turn and the stream resumes after the checkpoint
 * entry. Rows persisted after it come back through the stream.
 *
 * Without one the whole turn is replayed from the start. Assistant-side rows
 * created since the turn started belong to it. The prompt that started the
 * turn stays in history (it is persisted before the turn registers, and the
 * stream never repeats it); follow-ups drained into the turn are replayed
 * and are de-duplicated by content instead.
 */
export function splitActiveTurn(
  rows: readonly Row[],
  active: ActiveStream,
): ActiveTurnSplit {
  const checkpoint = active.checkpoint;
  if (checkpoint) {
    return {
      belongsToTurn: (row) =>
        row.sequence !== null && row.sequence >= checkpoint.sequence,
      seed: seedTurnLog({ turnID: active.turnID, rows, checkpoint }),
      after: checkpoint.entryID,
      replayedUserContents: [],
    };
  }
  const startedAt = active.startedAt
    ? Date.parse(active.startedAt)
    : Number.NaN;
  const createdInTurn = (row: Row) =>
    !Number.isNaN(startedAt) &&
    row.createdAt !== null &&
    Date.parse(row.createdAt) >= startedAt;
  return {
    belongsToTurn: (row) => row.role !== "user" && createdInTurn(row),
    seed: seedTurnLog({ turnID: active.turnID, rows: [], checkpoint: null }),
    after: "0-0",
    replayedUserContents: rows
      .filter((row) => row.role === "user" && createdInTurn(row))
      .map((row) => row.content),
  };
}
