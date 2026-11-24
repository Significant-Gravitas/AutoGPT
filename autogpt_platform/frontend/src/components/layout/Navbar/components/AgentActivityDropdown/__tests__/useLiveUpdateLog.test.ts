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

    expect(
      log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING)).accepted,
    ).toBe(false);
  });

  test("a list fetched after a read started stands over it", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    startList(log)(1);
    log.updatesNewerThan(1, [{ id: RUN }], false);

    expect(
      log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING)).accepted,
    ).toBe(false);
  });

  test("a list of every run fetched after a read started stands over it, run or no run", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    startList(log)(1);
    // The list has every run but not this one: the run is gone.
    log.updatesNewerThan(1, [], true);

    expect(
      log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING)).accepted,
    ).toBe(false);
  });

  test("a list of only the newest runs that leaves the run out doesn't stand over a read", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    startList(log)(1);
    log.updatesNewerThan(1, [], false);

    expect(
      log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING)).accepted,
    ).toBe(true);
  });

  test("a read that started after a list's fetch is replayed over that list", () => {
    const log = createLiveUpdateLog();
    const land = startList(log);
    log.startRead(RUN);
    const completed = status(RUN, AgentExecutionStatus.COMPLETED);
    expect(log.finishRead(RUN, completed).accepted).toBe(true);
    land(1);

    expect(log.updatesNewerThan(1, [{ id: RUN }], false)).toEqual([completed]);
  });

  test("a later list forgets a read it already includes", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    expect(
      log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING)).accepted,
    ).toBe(true);
    startList(log)(1);

    expect(log.updatesNewerThan(1, [{ id: RUN }], false)).toEqual([]);
  });

  test("a later capped list that leaves a run out keeps a read's completed answer for it", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    const completed = status(RUN, AgentExecutionStatus.COMPLETED);
    expect(log.finishRead(RUN, completed).accepted).toBe(true);
    // A later list, capped at the newest runs, that leaves the run out: it
    // says nothing about the run, so the read's answer survives.
    const land = startList(log);
    land(1);

    expect(log.updatesNewerThan(1, [], false)).toEqual([completed]);
  });

  test("a read by ID that finds a run gone drops its earlier live-update entry", () => {
    const log = createLiveUpdateLog();
    log.recordUpdate(status(RUN, AgentExecutionStatus.RUNNING));
    log.startRead(RUN);
    expect(log.finishRead(RUN, undefined, true).accepted).toBe(true);
    const land = startList(log);
    land(2);

    // A later capped list that leaves the run out must not replay the
    // stale RUNNING entry: the 404 read settled the run as gone.
    expect(log.updatesNewerThan(2, [], false)).toEqual([]);
  });

  test("a read by ID with no answer that is not gone keeps the earlier entry", () => {
    const log = createLiveUpdateLog();
    log.recordUpdate(status(RUN, AgentExecutionStatus.RUNNING));
    log.startRead(RUN);
    expect(log.finishRead(RUN, undefined).accepted).toBe(true);
    const land = startList(log);
    land(2);

    expect(log.updatesNewerThan(2, [], false)).toEqual([
      status(RUN, AgentExecutionStatus.RUNNING),
    ]);
  });

  test("a later capped list drops a read's answer for a run past the 72-hour retention", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    const old = {
      ...status(RUN, AgentExecutionStatus.COMPLETED),
      ended_at: new Date(Date.now() - 73 * 60 * 60 * 1000),
    };
    expect(log.finishRead(RUN, old).accepted).toBe(true);
    const land = startList(log);
    land(1);

    expect(log.updatesNewerThan(1, [], false)).toEqual([]);
  });

  test("a read that settles after the log is reset is not accepted", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    // Activity unmounted while the read was out: the reset drops the
    // registration, so the late answer must not repopulate the fresh log.
    log.reset();

    expect(
      log.finishRead(RUN, status(RUN, AgentExecutionStatus.RUNNING)).accepted,
    ).toBe(false);
    const land = startList(log);
    land(1);

    expect(log.updatesNewerThan(1, [], false)).toEqual([]);
  });

  test("a gone answer that settles after the log is reset is not accepted", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    log.reset();

    expect(log.finishRead(RUN, undefined, true).accepted).toBe(false);
    // The stale "gone" must not drop an entry the fresh log learned since.
    log.recordUpdate(status(RUN, AgentExecutionStatus.RUNNING));
    const land = startList(log);
    land(1);

    expect(log.updatesNewerThan(1, [], false)).toEqual([
      status(RUN, AgentExecutionStatus.RUNNING),
    ]);
  });

  test("a read started after a reset is accepted", () => {
    const log = createLiveUpdateLog();
    log.startRead(RUN);
    log.reset();
    log.startRead(RUN);
    const completed = status(RUN, AgentExecutionStatus.COMPLETED);
    expect(log.finishRead(RUN, completed).accepted).toBe(true);
    const land = startList(log);
    land(1);

    expect(log.updatesNewerThan(1, [], false)).toEqual([completed]);
  });
});
