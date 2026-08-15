import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { describe, expect, it } from "vitest";
import {
  getGraphWorkflowCounts,
  getWorkflowSchedules,
  workflowNeedsSetup,
} from "../../helpers";
import {
  getAdoptableExperts,
  getAdoptTargetKey,
  getAdoptTargetVersionID,
  getFilterView,
  getUnadoptedAgents,
  getVisibleGroups,
  pruneAdoptedTargetKeys,
} from "./helpers";

function makeWorkflow(over: Partial<ExpertWorkflowRef>): ExpertWorkflowRef {
  return {
    id: "wf",
    store_listing_version_id: "slv",
    library_agent_id: "lib",
    graph_id: "graph",
    name: "Workflow",
    description: null,
    ...over,
  };
}

function makeExpert(over: Partial<Expert>): Expert {
  return { id: "expert", name: "Expert", workflows: [], ...over } as Expert;
}

function makeAgent(over: Partial<LibraryAgent>): LibraryAgent {
  return {
    id: "agent",
    graph_id: "graph",
    name: "Agent",
    store_listing_version_id: null,
    ...over,
  } as unknown as LibraryAgent;
}

function makeSchedule(
  over: Partial<GraphExecutionJobInfo>,
): GraphExecutionJobInfo {
  return {
    id: "schedule",
    name: "Schedule",
    user_id: "user",
    graph_id: "graph-scheduled",
    graph_version: 1,
    cron: "40 7 * * *",
    input_data: {},
    next_run_time: "2026-08-15T07:40:00Z",
    expert_id: "a",
    ...over,
  };
}

describe("getFilterView", () => {
  it("shows both groups and agents for all", () => {
    expect(getFilterView("all")).toEqual({
      showGroups: true,
      showAgents: true,
      includeEmptyGroups: true,
      scheduledOnly: false,
    });
  });

  it("hides agents but keeps empty groups for members", () => {
    const view = getFilterView("members");
    expect(view.showGroups).toBe(true);
    expect(view.showAgents).toBe(false);
    expect(view.includeEmptyGroups).toBe(true);
  });

  it("shows only agents for agents", () => {
    expect(getFilterView("agents")).toMatchObject({
      showGroups: false,
      showAgents: true,
    });
  });

  it("drops empty groups for workflows", () => {
    expect(getFilterView("workflows").includeEmptyGroups).toBe(false);
  });

  it("keeps only scheduled rows for scheduled", () => {
    expect(getFilterView("scheduled")).toMatchObject({
      scheduledOnly: true,
      includeEmptyGroups: false,
    });
  });
});

describe("getVisibleGroups", () => {
  const withWorkflows = makeExpert({
    id: "a",
    workflows: [
      makeWorkflow({
        id: "s",
        graph_id: "graph-scheduled",
        schedule_id: "sched",
        schedule_cron: "40 7 * * *",
      }),
      makeWorkflow({ id: "u", graph_id: "graph-unscheduled" }),
    ],
  });
  const empty = makeExpert({ id: "b", workflows: [] });
  const schedules = [makeSchedule({})];

  it("keeps empty groups for all and members", () => {
    expect(
      getVisibleGroups([withWorkflows, empty], schedules, "all"),
    ).toHaveLength(2);
    expect(
      getVisibleGroups([withWorkflows, empty], schedules, "members"),
    ).toHaveLength(2);
  });

  it("drops empty groups for workflows", () => {
    const groups = getVisibleGroups(
      [withWorkflows, empty],
      schedules,
      "workflows",
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].expert.id).toBe("a");
  });

  it("keeps only workflows with scheduler jobs for scheduled", () => {
    const groups = getVisibleGroups(
      [withWorkflows, empty],
      schedules,
      "scheduled",
    );
    expect(groups).toHaveLength(1);
    expect(groups[0].workflows.map((item) => item.workflow.id)).toEqual(["s"]);
  });

  it("returns no groups for agents", () => {
    expect(
      getVisibleGroups([withWorkflows, empty], schedules, "agents"),
    ).toHaveLength(0);
  });

  it("ignores a stale workflow schedule id when no scheduler job exists", () => {
    expect(getVisibleGroups([withWorkflows], [], "scheduled")).toEqual([]);
  });

  it("keeps every scheduler job for a workflow", () => {
    const groups = getVisibleGroups(
      [withWorkflows],
      [makeSchedule({ id: "one" }), makeSchedule({ id: "two" })],
      "scheduled",
    );
    expect(
      groups[0].workflows[0].schedules.map((schedule) => schedule.id),
    ).toEqual(["one", "two"]);
  });

  it("does not assign one graph schedule to multiple workflow snapshots", () => {
    const first = makeWorkflow({
      id: "first",
      graph_id: "shared-graph",
      schedule_id: "first-schedule",
    });
    const second = makeWorkflow({
      id: "second",
      graph_id: "shared-graph",
      schedule_id: null,
    });
    const expert = makeExpert({
      id: "a",
      workflows: [first, second],
    });
    const groups = getVisibleGroups(
      [expert],
      [makeSchedule({ id: "first-schedule", graph_id: "shared-graph" })],
      "all",
    );

    expect(groups[0].workflows[0].schedules).toHaveLength(1);
    expect(groups[0].workflows[1].schedules).toHaveLength(0);
  });

  it("leaves a manual graph schedule unassigned when snapshots are ambiguous", () => {
    const workflows = [
      makeWorkflow({ id: "first", graph_id: "shared-graph" }),
      makeWorkflow({ id: "second", graph_id: "shared-graph" }),
    ];
    const schedule = makeSchedule({ id: "manual", graph_id: "shared-graph" });

    const graphWorkflowCounts = getGraphWorkflowCounts(workflows);
    expect(
      getWorkflowSchedules(workflows[0], [schedule], graphWorkflowCounts),
    ).toEqual([]);
    expect(
      getWorkflowSchedules(workflows[1], [schedule], graphWorkflowCounts),
    ).toEqual([]);
  });
});

describe("getUnadoptedAgents", () => {
  it("excludes an exact version when every expert already has it", () => {
    const experts = [
      makeExpert({
        id: "a",
        workflows: [
          makeWorkflow({
            graph_id: "graph-installed",
            store_listing_version_id: "slv-installed",
          }),
        ],
      }),
    ];
    const agents = [
      makeAgent({
        id: "installed",
        graph_id: "graph-installed",
        store_listing_version_id: "slv-installed",
      }),
      makeAgent({
        id: "free",
        graph_id: "graph-free",
        store_listing_version_id: "slv-free",
      }),
    ];
    const result = getUnadoptedAgents(agents, experts);
    expect(result.map((a) => a.id)).toEqual(["free"]);
  });

  it("keeps a newer snapshot of an installed graph adoptable", () => {
    const expert = makeExpert({
      workflows: [
        makeWorkflow({
          graph_id: "shared-graph",
          store_listing_version_id: "slv-v1",
        }),
      ],
    });
    const newer = makeAgent({
      graph_id: "shared-graph",
      store_listing_version_id: "slv-v2",
    });

    expect(getUnadoptedAgents([newer], [expert])).toEqual([newer]);
  });

  it("keeps an agent visible while another expert can adopt it", () => {
    const installed = makeExpert({
      id: "installed",
      workflows: [makeWorkflow({ store_listing_version_id: "slv-shared" })],
    });
    const available = makeExpert({ id: "available", workflows: [] });
    const agent = makeAgent({ store_listing_version_id: "slv-shared" });

    expect(getUnadoptedAgents([agent], [installed, available])).toEqual([
      agent,
    ]);
    expect(getAdoptableExperts(agent, [installed, available])).toEqual([
      available,
    ]);

    const optimistic = new Set([getAdoptTargetKey(agent, available)]);
    expect(
      getUnadoptedAgents([agent], [installed, available], optimistic),
    ).toEqual([]);
  });
});

describe("getAdoptTargetVersionID", () => {
  it("returns the exact-match version id when present", () => {
    expect(
      getAdoptTargetVersionID(
        makeAgent({ store_listing_version_id: "slv-exact" }),
      ),
    ).toBe("slv-exact");
  });

  it("returns null for a pure-local agent", () => {
    expect(
      getAdoptTargetVersionID(makeAgent({ store_listing_version_id: null })),
    ).toBe(null);
  });
});

describe("pruneAdoptedTargetKeys", () => {
  it("removes confirmed and fired-expert targets before the set grows", () => {
    const pending = makeExpert({ id: "pending", workflows: [] });
    const confirmed = makeExpert({
      id: "confirmed",
      workflows: [makeWorkflow({ store_listing_version_id: "slv-exact" })],
    });
    const fired = makeExpert({ id: "fired", workflows: [] });
    const agent = makeAgent({ store_listing_version_id: "slv-exact" });
    const keys = new Set([
      getAdoptTargetKey(agent, pending),
      getAdoptTargetKey(agent, confirmed),
      getAdoptTargetKey(agent, fired),
    ]);

    expect(pruneAdoptedTargetKeys(keys, [agent], [pending, confirmed])).toEqual(
      new Set([getAdoptTargetKey(agent, pending)]),
    );
  });

  it("preserves the set identity when no targets can be pruned", () => {
    const expert = makeExpert({ workflows: [] });
    const agent = makeAgent({ store_listing_version_id: "slv-exact" });
    const keys = new Set([getAdoptTargetKey(agent, expert)]);

    expect(pruneAdoptedTargetKeys(keys, [agent], [expert])).toBe(keys);
  });
});

describe("workflow schedule helpers", () => {
  it("marks cron without a schedule id as needing setup", () => {
    expect(
      workflowNeedsSetup(makeWorkflow({ schedule_cron: "40 7 * * *" })),
    ).toBe(true);
    expect(
      workflowNeedsSetup(
        makeWorkflow({ schedule_cron: "40 7 * * *", schedule_id: "x" }),
      ),
    ).toBe(false);
  });

  it("uses scheduler jobs instead of a stale schedule id when supplied", () => {
    const workflow = makeWorkflow({
      graph_id: "graph-scheduled",
      schedule_cron: "40 7 * * *",
      schedule_id: "stale",
    });
    expect(workflowNeedsSetup(workflow, [])).toBe(true);
    expect(
      workflowNeedsSetup(
        workflow,
        getWorkflowSchedules(
          workflow,
          [makeSchedule({ id: "active" })],
          getGraphWorkflowCounts([workflow]),
        ),
      ),
    ).toBe(false);
  });
});
