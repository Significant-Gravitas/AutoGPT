import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { describe, expect, test } from "vitest";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  filterExpertSchedules,
  filterExpertWorkflows,
  getAssignToastTitle,
  getExpertRosterStatus,
  groupExpertsByPods,
} from "./helpers";

function makeExpert(id: string, podId: string | null = null): Expert {
  return {
    id,
    name: id,
    avatar_url: null,
    role: "Role",
    tagline: null,
    bio: null,
    skills: [],
    identity: "identity",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    pod_id: podId,
  };
}

function makePod(id: string, name: string): ExpertPod {
  return { id, name, created_at: new Date("2026-08-14T00:00:00Z") };
}

describe("groupExpertsByPods", () => {
  test("places experts under their pod and the rest ungrouped", () => {
    const growth = makePod("pod-growth", "Growth");
    const support = makePod("pod-support", "Support");
    const { groups, ungrouped } = groupExpertsByPods(
      [
        makeExpert("maria", "pod-growth"),
        makeExpert("sam", "pod-support"),
        makeExpert("lee"),
      ],
      [growth, support],
    );

    expect(groups.map((g) => g.pod.name)).toEqual(["Growth", "Support"]);
    expect(groups[0].experts.map((e) => e.id)).toEqual(["maria"]);
    expect(groups[1].experts.map((e) => e.id)).toEqual(["sam"]);
    expect(ungrouped.map((e) => e.id)).toEqual(["lee"]);
  });

  test("keeps an empty pod as a group", () => {
    const { groups } = groupExpertsByPods(
      [makeExpert("lee")],
      [makePod("pod-empty", "Empty")],
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].experts).toEqual([]);
  });

  test("treats a dangling pod_id as ungrouped", () => {
    const { groups, ungrouped } = groupExpertsByPods(
      [makeExpert("maria", "pod-deleted")],
      [makePod("pod-growth", "Growth")],
    );
    expect(groups[0].experts).toEqual([]);
    expect(ungrouped.map((e) => e.id)).toEqual(["maria"]);
  });

  test("returns everything ungrouped when there are no pods", () => {
    const { groups, ungrouped } = groupExpertsByPods(
      [makeExpert("maria"), makeExpert("sam")],
      [],
    );
    expect(groups).toEqual([]);
    expect(ungrouped).toHaveLength(2);
  });
});

describe("getAssignToastTitle", () => {
  test("names the destination pod when it is known", () => {
    expect(
      getAssignToastTitle({ podId: "pod-growth", destinationName: "Growth" }),
    ).toBe("Moved to Growth");
  });

  test("falls back to a generic title when the pod list is stale", () => {
    expect(getAssignToastTitle({ podId: "pod-growth" })).toBe("Expert moved");
  });

  test("reports a detach regardless of the destination name", () => {
    expect(getAssignToastTitle({ podId: null })).toBe("Removed from pod");
  });
});

describe("getExpertRosterStatus", () => {
  test("reports active work before attention states", () => {
    const expert = makeExpert("working");
    expert.last_run_status = "RUNNING";
    expert.schedules_paused_at = new Date("2026-09-03T10:00:00Z");

    expect(getExpertRosterStatus(expert, 1)).toBe("working");
  });

  test.each(["FAILED", "TERMINATED", "REVIEW"])(
    "reports %s as needing attention",
    (lastRunStatus) => {
      const expert = makeExpert("attention");
      expert.last_run_status = lastRunStatus;

      expect(getExpertRosterStatus(expert, 0)).toBe("needs-you");
    },
  );

  test("reports paused and unconfigured experts as needing attention", () => {
    const paused = makeExpert("paused");
    paused.schedules_paused_at = new Date("2026-09-03T10:00:00Z");

    expect(getExpertRosterStatus(paused, 0)).toBe("needs-you");
    expect(getExpertRosterStatus(makeExpert("setup"), 1)).toBe("needs-you");
  });

  test("reports an expert with no active issue as idle", () => {
    const expert = makeExpert("idle");
    expert.last_run_status = "COMPLETED";

    expect(getExpertRosterStatus(expert, 0)).toBe("idle");
  });
});

describe("filterExpertWorkflows", () => {
  const workflows: ExpertWorkflowRef[] = [
    {
      id: "wf-1",
      store_listing_version_id: null,
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "Content Calendar",
      description: "Plans a week of posts",
      schedule_cron: "40 7 * * *",
      schedule_id: "sched-1",
    },
    {
      id: "wf-2",
      store_listing_version_id: null,
      library_agent_id: "lib-2",
      graph_id: "graph-2",
      name: "SEO Audit",
      description: null,
      schedule_cron: "0 9 * * 1",
      schedule_id: null,
    },
    {
      id: "wf-3",
      store_listing_version_id: null,
      library_agent_id: "lib-3",
      graph_id: "graph-3",
      name: "Draft reply",
      description: "Answers inbound posts",
      schedule_cron: null,
      schedule_id: null,
    },
  ];

  test("matches the query against name and description", () => {
    expect(
      filterExpertWorkflows(workflows, "posts", "all").map((w) => w.id),
    ).toEqual(["wf-1", "wf-3"]);
    expect(
      filterExpertWorkflows(workflows, "  seo ", "all").map((w) => w.id),
    ).toEqual(["wf-2"]);
  });

  test("splits scheduled, manual and needs-setup workflows", () => {
    expect(
      filterExpertWorkflows(workflows, "", "scheduled").map((w) => w.id),
    ).toEqual(["wf-1"]);
    expect(
      filterExpertWorkflows(workflows, "", "needs-setup").map((w) => w.id),
    ).toEqual(["wf-2"]);
    expect(
      filterExpertWorkflows(workflows, "", "manual").map((w) => w.id),
    ).toEqual(["wf-3"]);
  });

  test("combines query and filter", () => {
    expect(
      filterExpertWorkflows(workflows, "posts", "manual").map((w) => w.id),
    ).toEqual(["wf-3"]);
  });
});

describe("filterExpertSchedules", () => {
  const now = new Date("2026-09-04T00:00:00Z");
  function schedule(
    id: string,
    name: string,
    hoursAhead: number,
  ): GraphExecutionJobInfo {
    return {
      id,
      name,
      agent_name: `${name} agent`,
      user_id: "user-1",
      graph_id: `graph-${id}`,
      graph_version: 1,
      cron: "0 9 * * *",
      input_data: {},
      next_run_time: new Date(
        now.getTime() + hoursAhead * 3600_000,
      ).toISOString(),
    };
  }
  const schedules = [
    schedule("s1", "Morning digest", 3),
    schedule("s2", "Weekly report", 3 * 24),
    schedule("s3", "Quarterly review", 30 * 24),
  ];

  test("matches the query against schedule and agent names", () => {
    expect(
      filterExpertSchedules(schedules, "weekly", "all", now).map((s) => s.id),
    ).toEqual(["s2"]);
    expect(filterExpertSchedules(schedules, "agent", "all", now)).toHaveLength(
      3,
    );
  });

  test("buckets by next run time", () => {
    expect(
      filterExpertSchedules(schedules, "", "today", now).map((s) => s.id),
    ).toEqual(["s1"]);
    expect(
      filterExpertSchedules(schedules, "", "week", now).map((s) => s.id),
    ).toEqual(["s1", "s2"]);
    expect(
      filterExpertSchedules(schedules, "", "later", now).map((s) => s.id),
    ).toEqual(["s3"]);
  });

  test("treats a missing next run time as later, never as due now", () => {
    const paused = { ...schedule("s4", "Paused sync", 1), next_run_time: "" };
    const all = [...schedules, paused];
    expect(
      filterExpertSchedules(all, "", "today", now).map((s) => s.id),
    ).toEqual(["s1"]);
    expect(
      filterExpertSchedules(all, "", "week", now).map((s) => s.id),
    ).toEqual(["s1", "s2"]);
    expect(
      filterExpertSchedules(all, "", "later", now).map((s) => s.id),
    ).toEqual(["s3", "s4"]);
  });
});
