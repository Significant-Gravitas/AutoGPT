import { describe, expect, it } from "vitest";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { getExpertStatusLine } from "./helpers";

function makeWorkflow(id: string): ExpertWorkflowRef {
  // schedule_cron set + schedule_id missing is what "needs setup" means.
  return {
    id,
    store_listing_version_id: `slv-${id}`,
    library_agent_id: `lib-${id}`,
    graph_id: `graph-${id}`,
    name: `Workflow ${id}`,
    description: null,
    schedule_cron: "0 9 * * *",
    schedule_id: null,
  };
}

function makeExpert(overrides: Partial<Expert> = {}): Expert {
  return {
    id: "expert-1",
    name: "Sales Scout",
    avatar_url: null,
    role: "Sales Researcher",
    bio: null,
    skills: [],
    tagline: null,
    identity: "You are a senior sales researcher.",
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    ...overrides,
  };
}

function makeSchedule(overrides: Partial<GraphExecutionJobInfo> = {}) {
  return {
    id: "sched-1",
    name: "Daily Report",
    user_id: "u-1",
    graph_id: "graph-1",
    graph_version: 1,
    cron: "0 9 * * *",
    input_data: {},
    next_run_time: "2026-08-10T14:30:00Z",
    expert_id: "expert-1",
    ...overrides,
  } as GraphExecutionJobInfo;
}

describe("getExpertStatusLine", () => {
  it("reports a paused expert before anything else", () => {
    const expert = makeExpert({
      schedules_paused_at: new Date("2026-08-05T10:00:00Z"),
      workflows: [makeWorkflow("1")],
    });

    expect(getExpertStatusLine(expert, [])).toBe("Paused");
  });

  it("uses singular grammar for one workflow needing setup", () => {
    const expert = makeExpert({ workflows: [makeWorkflow("1")] });

    expect(getExpertStatusLine(expert, [])).toBe("1 workflow needs setup");
  });

  it("uses plural grammar for several workflows needing setup", () => {
    const expert = makeExpert({
      workflows: [makeWorkflow("1"), makeWorkflow("2")],
    });

    expect(getExpertStatusLine(expert, [])).toBe("2 workflows need setup");
  });

  it("falls back to the last run once nothing needs setup", () => {
    const expert = makeExpert({
      last_run_at: new Date("2026-08-06T10:00:00Z"),
      last_run_status: "COMPLETED",
    });

    expect(getExpertStatusLine(expert, [])).toContain("Last run succeeded");
  });

  it("shows the next scheduled run for an expert that has never run", () => {
    const expert = makeExpert();

    expect(getExpertStatusLine(expert, [makeSchedule()])).toMatch(/^Next run /);
  });

  it("falls back to Idle with no runs and no schedules", () => {
    expect(getExpertStatusLine(makeExpert(), [])).toBe("Idle");
  });
});
