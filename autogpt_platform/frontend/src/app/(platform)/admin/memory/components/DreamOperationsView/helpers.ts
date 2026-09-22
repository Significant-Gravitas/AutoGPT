import type { DreamOperationsSnapshot } from "@/app/api/__generated__/models/dreamOperationsSnapshot";

export function shortenUuid(uuid: string | null | undefined): string {
  if (!uuid) return "—";
  if (uuid.length <= 12) return uuid;
  return `${uuid.slice(0, 12)}…`;
}

export function formatConfidence(c: number | null | undefined): string {
  if (c === null || c === undefined) return "—";
  return `${(c * 100).toFixed(0)}%`;
}

export function sectionCounts(ops: DreamOperationsSnapshot | null | undefined) {
  if (!ops) {
    return { writes: 0, proposals: 0, demotions: 0, entities: 0 };
  }
  return {
    writes: ops.writes?.length ?? 0,
    proposals: ops.proposals?.length ?? 0,
    demotions: ops.demotions?.length ?? 0,
    entities: ops.entity_invalidations?.length ?? 0,
  };
}
