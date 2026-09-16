import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import { getSessionActivity } from "./helpers";

function toolPart(overrides: Record<string, unknown>) {
  return {
    type: "tool-run_agent",
    toolCallId: "call-1",
    state: "output-available",
    input: {},
    ...overrides,
  };
}

function messageOf(...parts: Record<string, unknown>[]): UIMessage[] {
  return [{ id: "m1", role: "assistant", parts }] as unknown as UIMessage[];
}

describe("getSessionActivity", () => {
  it("returns nothing for an empty transcript", () => {
    expect(getSessionActivity([])).toEqual({ runs: [], schedules: [] });
  });

  it("ignores non-tool message parts", () => {
    const messages = messageOf({ type: "text", text: "hello" });
    expect(getSessionActivity(messages)).toEqual({ runs: [], schedules: [] });
  });

  it("ignores tool parts with no output that aren't run tools", () => {
    const messages = messageOf(
      toolPart({ type: "tool-something_else", output: undefined }),
    );
    expect(getSessionActivity(messages)).toEqual({ runs: [], schedules: [] });
  });

  it("surfaces an in-flight run from the tool input, deslugifying the name", () => {
    const messages = messageOf(
      toolPart({
        output: undefined,
        state: "input-available",
        input: { username_agent_slug: "creator/daily-digest" },
      }),
    );
    const { runs } = getSessionActivity(messages);
    expect(runs).toHaveLength(1);
    expect(runs[0]).toMatchObject({
      name: "daily digest",
      status: "RUNNING",
      libraryAgentId: null,
      startedAt: null,
      href: null,
    });
  });

  it("surfaces an in-flight run started by library-agent id with a null name", () => {
    const messages = messageOf(
      toolPart({
        output: undefined,
        state: "input-streaming",
        input: { library_agent_id: "agent-123" },
      }),
    );
    const { runs } = getSessionActivity(messages);
    expect(runs).toHaveLength(1);
    expect(runs[0].name).toBeNull();
    expect(runs[0].libraryAgentId).toBe("agent-123");
  });

  it("does not surface a pending row for states outside streaming/available", () => {
    const messages = messageOf(
      toolPart({ output: undefined, state: "output-available", input: {} }),
    );
    expect(getSessionActivity(messages).runs).toEqual([]);
  });

  it("skips a pending run tool whose input is actually a schedule request", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-schedule_agent",
        output: undefined,
        state: "input-available",
        input: { cron: "0 9 * * *" },
      }),
    );
    expect(getSessionActivity(messages).runs).toEqual([]);
  });

  it("skips tool parts with no output entirely when not a run tool", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-list_files",
        output: undefined,
        state: "input-available",
      }),
    );
    expect(getSessionActivity(messages)).toEqual({ runs: [], schedules: [] });
  });

  it("records a completed async run from a top-level execution_started envelope", () => {
    const messages = messageOf(
      toolPart({
        output: {
          execution_id: "exec-1",
          status: "RUNNING",
          graph_name: "Daily Digest",
          graph_id: "graph-1",
          started_at: "2026-05-20T10:00:00Z",
        },
      }),
    );
    const { runs } = getSessionActivity(messages);
    expect(runs).toEqual([
      {
        executionId: "exec-1",
        name: "Daily Digest",
        libraryAgentId: null,
        status: "RUNNING",
        startedAt: "2026-05-20T10:00:00Z",
        href: "/library/agents/graph-1?activeTab=runs&activeItem=exec-1",
      },
    ]);
  });

  it("records a waited run from the nested `execution` envelope", () => {
    const messages = messageOf(
      toolPart({
        output: {
          agent_name: "Report Builder",
          library_agent_link: "/library/agents/graph-2",
          execution: {
            execution_id: "exec-2",
            status: "COMPLETED",
            started_at: "2026-05-20T11:00:00Z",
          },
        },
      }),
    );
    const { runs } = getSessionActivity(messages);
    expect(runs).toEqual([
      {
        executionId: "exec-2",
        name: "Report Builder",
        libraryAgentId: null,
        status: "COMPLETED",
        startedAt: "2026-05-20T11:00:00Z",
        href: "/library/agents/graph-2",
      },
    ]);
  });

  it("drops a run tool response with no execution id", () => {
    const messages = messageOf(
      toolPart({ output: { graph_name: "No Id Agent" } }),
    );
    expect(getSessionActivity(messages).runs).toEqual([]);
  });

  it("treats a SCHEDULED run response as a schedule, not a run", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-schedule_agent",
        input: { schedule_name: "Nightly", cron: "0 0 * * *", timezone: "UTC" },
        output: {
          execution_id: "sched-exec-1",
          status: "SCHEDULED",
          graph_name: "Nightly Job",
          next_run_time: "2026-05-21T00:00:00Z",
        },
      }),
    );
    const activity = getSessionActivity(messages);
    expect(activity.runs).toEqual([]);
    expect(activity.schedules).toEqual([
      {
        scheduleId: "sched-exec-1",
        name: "Nightly",
        nextRunTime: "2026-05-21T00:00:00Z",
        isRecurring: true,
        detail: "Runs Nightly Job",
        cron: "0 0 * * *",
        timezone: "UTC",
      },
    ]);
  });

  it("falls back to the graph name, then 'Schedule', for a SCHEDULED response with no schedule_name", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-schedule_agent",
        input: {},
        output: {
          execution_id: "sched-exec-2",
          status: "SCHEDULED",
          graph_name: "Weekly Job",
          next_run_time: null,
        },
      }),
    );
    const { schedules } = getSessionActivity(messages);
    expect(schedules[0].name).toBe("Weekly Job");
    expect(schedules[0].detail).toBe("Runs Weekly Job");
  });

  it("names an unlabeled SCHEDULED response 'Schedule' with a null detail", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-schedule_agent",
        input: {},
        output: {
          execution_id: "sched-exec-3",
          status: "SCHEDULED",
          next_run_time: null,
        },
      }),
    );
    const { schedules } = getSessionActivity(messages);
    expect(schedules[0].name).toBe("Schedule");
    expect(schedules[0].detail).toBeNull();
  });

  it("records a schedule_created envelope, preferring output.cron over input.cron", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-create_schedule",
        input: { cron: "input-cron", message: "input message" },
        output: {
          type: "schedule_created",
          schedule_id: "sched-3",
          name: "Explicit Name",
          next_run_time: "2026-05-22T00:00:00Z",
          cron: "output-cron",
          message: "output message",
        },
      }),
    );
    const { schedules } = getSessionActivity(messages);
    expect(schedules[0]).toMatchObject({
      scheduleId: "sched-3",
      name: "Explicit Name",
      cron: "output-cron",
      detail: "output message",
      isRecurring: true,
    });
  });

  it("marks a schedule recurring from is_recurring even without a cron string", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-create_schedule",
        input: {},
        output: {
          type: "schedule_created",
          schedule_id: "sched-4",
          name: "Recurring Flag",
          is_recurring: true,
        },
      }),
    );
    const { schedules } = getSessionActivity(messages);
    expect(schedules[0].isRecurring).toBe(true);
    expect(schedules[0].cron).toBeNull();
  });

  it("names a schedule_followup 'Follow-up' and takes its detail from the input prompt", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-schedule_followup",
        input: { prompt: "Check the inbox", timezone: "UTC" },
        output: {
          schedule_id: "sched-5",
          next_run_time: "2026-05-23T09:00:00Z",
        },
      }),
    );
    const { schedules } = getSessionActivity(messages);
    expect(schedules[0]).toMatchObject({
      scheduleId: "sched-5",
      name: "Follow-up",
      detail: "Check the inbox",
      timezone: "UTC",
    });
  });

  it("does not treat a schedule_followup with no next_run_time as a created schedule", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-schedule_followup",
        input: { prompt: "Check the inbox" },
        output: { schedule_id: "sched-6" },
      }),
    );
    expect(getSessionActivity(messages).schedules).toEqual([]);
  });

  it("removes a schedule that was later deleted in the same transcript", () => {
    const messages = messageOf(
      toolPart({
        type: "tool-create_schedule",
        input: {},
        output: {
          type: "schedule_created",
          schedule_id: "sched-7",
          name: "Temp",
          next_run_time: "2026-05-24T00:00:00Z",
        },
      }),
      toolPart({
        type: "tool-delete_schedule",
        input: {},
        output: { type: "schedule_deleted", schedule_id: "sched-7" },
      }),
    );
    expect(getSessionActivity(messages).schedules).toEqual([]);
  });

  it("orders runs and schedules newest-first", () => {
    const messages = messageOf(
      toolPart({
        toolCallId: "call-a",
        output: { execution_id: "exec-old", graph_name: "Old Run" },
      }),
      toolPart({
        toolCallId: "call-b",
        output: { execution_id: "exec-new", graph_name: "New Run" },
      }),
    );
    const { runs } = getSessionActivity(messages);
    expect(runs.map((r) => r.executionId)).toEqual(["exec-new", "exec-old"]);
  });
});
