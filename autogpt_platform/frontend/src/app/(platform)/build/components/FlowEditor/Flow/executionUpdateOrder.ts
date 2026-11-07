import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";

// Orders websocket deliveries against execution-details fetches for the
// subscribed run. The backend writes a change before it publishes it, so a
// fetch that started after a delivery arrived has read that change or a newer
// one, while a delivery that arrived after a fetch started may be newer than
// what the fetch read. Each delivery and each fetch start takes the next
// number in one sequence, which tells the two cases apart.

const MAX_TRACKED_SNAPSHOTS = 20;

let sequence = 0;
let fetchStartSequence = 0;
let graphStatusSequence = 0;
const deliveredNodeExecutions = new Map<
  string,
  { result: NodeExecutionResult; sequence: number }
>();
// dataUpdatedAt of a snapshot -> sequence when the fetch behind it started
const snapshotSequences = new Map<number, number>();

export function recordNodeExecutionDelivery(result: NodeExecutionResult) {
  sequence += 1;
  deliveredNodeExecutions.delete(result.node_exec_id);
  deliveredNodeExecutions.set(result.node_exec_id, { result, sequence });
}

export function recordGraphStatusDelivery() {
  sequence += 1;
  graphStatusSequence = sequence;
}

export function recordFetchStart() {
  sequence += 1;
  fetchStartSequence = sequence;
}

export function recordFetchSuccess(dataUpdatedAt: number) {
  snapshotSequences.set(dataUpdatedAt, fetchStartSequence);
  if (snapshotSequences.size > MAX_TRACKED_SNAPSHOTS) {
    const oldest = snapshotSequences.keys().next().value;
    if (oldest !== undefined) snapshotSequences.delete(oldest);
  }
}

/** Forget the deliveries, e.g. when the websocket drops and may miss events. */
export function forgetDeliveries() {
  deliveredNodeExecutions.clear();
  graphStatusSequence = 0;
}

// A snapshot whose fetch start was not seen counts as older than every
// delivery, so the websocket's copies win.
function snapshotSequence(dataUpdatedAt: number) {
  return snapshotSequences.get(dataUpdatedAt) ?? 0;
}

/**
 * The node executions to apply for a snapshot, in order: the snapshot's own,
 * except those the websocket delivered after its fetch started, then those
 * deliveries in arrival order.
 */
export function orderSnapshotExecutions(
  nodeExecutions: NodeExecutionResult[],
  dataUpdatedAt: number,
) {
  const since = snapshotSequence(dataUpdatedAt);
  const newer = Array.from(deliveredNodeExecutions.values())
    .filter((delivery) => delivery.sequence > since)
    .map((delivery) => delivery.result);
  if (newer.length === 0) return nodeExecutions;
  const newerIDs = new Set(newer.map((result) => result.node_exec_id));
  return [
    ...nodeExecutions.filter(
      (nodeExecution) => !newerIDs.has(nodeExecution.node_exec_id),
    ),
    ...newer,
  ];
}

/** False when the websocket sent a run status after this snapshot's fetch started. */
export function isSnapshotStatusCurrent(dataUpdatedAt: number) {
  return graphStatusSequence <= snapshotSequence(dataUpdatedAt);
}
