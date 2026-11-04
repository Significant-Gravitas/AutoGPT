import { beforeEach, describe, expect, it } from "vitest";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import {
  forgetDeliveries,
  isSnapshotStatusCurrent,
  orderSnapshotExecutions,
  recordFetchStart,
  recordFetchSuccess,
  recordGraphStatusDelivery,
  recordNodeExecutionDelivery,
} from "../executionUpdateOrder";

function execution(
  nodeExecID: string,
  status: NodeExecutionResult["status"],
): NodeExecutionResult {
  return {
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    graph_exec_id: "run-1",
    node_exec_id: nodeExecID,
    node_id: `node-of-${nodeExecID}`,
    block_id: "block-1",
    status,
    input_data: {},
    output_data: {},
    add_time: new Date("2026-01-05T09:00:00.000Z"),
    queue_time: null,
    start_time: null,
    end_time: null,
  };
}

let nextUpdatedAt = 1_000;

// A fetch whose start and landing are recorded around `during`.
function fetchSnapshot(during: () => void = () => undefined) {
  recordFetchStart();
  during();
  nextUpdatedAt += 1;
  recordFetchSuccess(nextUpdatedAt);
  return nextUpdatedAt;
}

describe("executionUpdateOrder", () => {
  beforeEach(() => {
    forgetDeliveries();
  });

  it("applies a snapshot as is when nothing arrived over the websocket", () => {
    const snapshot = [execution("a", "RUNNING"), execution("b", "QUEUED")];
    const updatedAt = fetchSnapshot();

    expect(orderSnapshotExecutions(snapshot, updatedAt)).toBe(snapshot);
    expect(isSnapshotStatusCurrent(updatedAt)).toBe(true);
  });

  it("puts deliveries that arrived while the fetch was in flight last", () => {
    const delivered = execution("a", "COMPLETED");
    const updatedAt = fetchSnapshot(() => {
      recordNodeExecutionDelivery(delivered);
      recordGraphStatusDelivery();
    });
    const snapshot = [execution("a", "RUNNING"), execution("b", "QUEUED")];

    expect(orderSnapshotExecutions(snapshot, updatedAt)).toEqual([
      snapshot[1],
      delivered,
    ]);
    expect(isSnapshotStatusCurrent(updatedAt)).toBe(false);
  });

  it("trusts a snapshot fetched after the deliveries arrived", () => {
    recordNodeExecutionDelivery(execution("a", "RUNNING"));
    recordGraphStatusDelivery();
    const snapshot = [execution("a", "COMPLETED")];
    const updatedAt = fetchSnapshot();

    expect(orderSnapshotExecutions(snapshot, updatedAt)).toBe(snapshot);
    expect(isSnapshotStatusCurrent(updatedAt)).toBe(true);
  });

  it("keeps deliveries in arrival order, with the latest copy of each", () => {
    const updatedAt = fetchSnapshot(() => {
      recordNodeExecutionDelivery(execution("a", "RUNNING"));
      recordNodeExecutionDelivery(execution("b", "RUNNING"));
      recordNodeExecutionDelivery(execution("a", "COMPLETED"));
    });

    expect(
      orderSnapshotExecutions([], updatedAt).map(
        (result) => `${result.node_exec_id}:${result.status}`,
      ),
    ).toEqual(["b:RUNNING", "a:COMPLETED"]);
  });

  it("treats a snapshot whose fetch start was not seen as older than every delivery", () => {
    const delivered = execution("a", "COMPLETED");
    recordNodeExecutionDelivery(delivered);
    recordGraphStatusDelivery();

    expect(orderSnapshotExecutions([execution("a", "RUNNING")], 42)).toEqual([
      delivered,
    ]);
    expect(isSnapshotStatusCurrent(42)).toBe(false);
  });

  it("trusts snapshots again once the deliveries are forgotten", () => {
    const updatedAt = fetchSnapshot(() => {
      recordNodeExecutionDelivery(execution("a", "COMPLETED"));
      recordGraphStatusDelivery();
    });
    forgetDeliveries();
    const snapshot = [execution("a", "RUNNING")];

    expect(orderSnapshotExecutions(snapshot, updatedAt)).toBe(snapshot);
    expect(isSnapshotStatusCurrent(updatedAt)).toBe(true);
  });

  it("keeps tracking recent snapshots after older ones are dropped", () => {
    const first = fetchSnapshot(() =>
      recordNodeExecutionDelivery(execution("a", "COMPLETED")),
    );
    for (let index = 0; index < 25; index++) fetchSnapshot();
    const latest = fetchSnapshot();

    expect(
      orderSnapshotExecutions([execution("a", "RUNNING")], latest),
    ).toEqual([execution("a", "RUNNING")]);
    // The first snapshot is no longer tracked, so it counts as the oldest.
    expect(
      orderSnapshotExecutions([execution("a", "RUNNING")], first)[0].status,
    ).toBe("COMPLETED");
  });
});
