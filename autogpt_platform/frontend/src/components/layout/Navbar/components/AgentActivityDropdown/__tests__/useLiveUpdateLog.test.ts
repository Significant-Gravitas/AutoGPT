import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { describe, expect, test } from "vitest";
import { createLiveUpdateLog } from "../useLiveUpdateLog";

function status(id: string, value: AgentExecutionStatus) {
  return { id, status: value } as GraphExecution;
}

const RUN = "run-1";

/** A list fetch that starts now and lands as the list at `dataUpdatedAt`. */
function startList(log: ReturnType<typeof createLiveUpdateLog>) {
  log.recordFetchStart();
  return (dataUpdatedAt: number) => log.recordFetchSuccess(dataUpdatedAt);
}

describe("reads of a single run against live updates and lists", () => {
  test("a live update that arrives during a read stands over it", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    log.recordUpdate(status(RUN, AgentExecutionStatus.REVIEW));

    expect(log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING))).toBe(
      false,
    );
  });

  test("a list fetched after a read started stands over it", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    startList(log)(1);
    log.updatesNewerThan(1, [{ id: RUN }]);

    expect(log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING))).toBe(
      false,
    );
  });

  test("a read that started after a list's fetch is replayed over that list", () => {
    const log = createLiveUpdateLog();
    const land = startList(log);
    log.startRead(RUN);
    const completed = status(RUN, AgentExecutionStatus.COMPLETED);
    expect(log.finishRead(RUN, completed)).toBe(true);
    land(1);

    expect(log.updatesNewerThan(1, [{ id: RUN }])).toEqual([completed]);
  });

  test("a later list forgets a read it already includes", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    expect(log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING))).toBe(
      true,
    );
    startList(log)(1);

    expect(log.updatesNewerThan(1, [{ id: RUN }])).toEqual([]);
  });
});
