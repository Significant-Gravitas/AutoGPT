import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { describe, expect, it } from "vitest";
import {
  getAdoptTargetVersionId,
  getFilterView,
  getUnadoptedAgents,
  getVisibleGroups,
  getWorkflowScheduleLabel,
  isWorkflowScheduled,
  workflowNeedsSetup,
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
        schedule_id: "sched",
        schedule_cron: "40 7 * * *",
      }),
      makeWorkflow({ id: "u" }),
    ],
  });
  const empty = makeExpert({ id: "b", workflows: [] });

  it("keeps empty groups for all and members", () => {
    expect(getVisibleGroups([withWorkflows, empty], "all")).toHaveLength(2);
    expect(getVisibleGroups([withWorkflows, empty], "members")).toHaveLength(2);
  });

  it("drops empty groups for workflows", () => {
    const groups = getVisibleGroups([withWorkflows, empty], "workflows");
    expect(groups).toHaveLength(1);
    expect(groups[0].expert.id).toBe("a");
  });

  it("keeps only scheduled workflows for scheduled", () => {
    const groups = getVisibleGroups([withWorkflows, empty], "scheduled");
    expect(groups).toHaveLength(1);
    expect(groups[0].workflows.map((w) => w.id)).toEqual(["s"]);
  });

  it("returns no groups for agents", () => {
    expect(getVisibleGroups([withWorkflows, empty], "agents")).toHaveLength(0);
  });
});

describe("getUnadoptedAgents", () => {
  it("excludes agents already installed on any expert by graph id", () => {
    const experts = [
      makeExpert({
        id: "a",
        workflows: [makeWorkflow({ graph_id: "graph-installed" })],
      }),
    ];
    const agents = [
      makeAgent({ id: "installed", graph_id: "graph-installed" }),
      makeAgent({ id: "free", graph_id: "graph-free" }),
    ];
    const result = getUnadoptedAgents(agents, experts);
    expect(result.map((a) => a.id)).toEqual(["free"]);
  });
});

describe("getAdoptTargetVersionId", () => {
  it("returns the exact-match version id when present", () => {
    expect(
      getAdoptTargetVersionId(
        makeAgent({ store_listing_version_id: "slv-exact" }),
      ),
    ).toBe("slv-exact");
  });

  it("returns null for a pure-local agent", () => {
    expect(
      getAdoptTargetVersionId(makeAgent({ store_listing_version_id: null })),
    ).toBe(null);
  });
});

describe("workflow schedule helpers", () => {
  it("marks a workflow scheduled only when it has a schedule id", () => {
    expect(isWorkflowScheduled(makeWorkflow({ schedule_id: "x" }))).toBe(true);
    expect(isWorkflowScheduled(makeWorkflow({ schedule_id: null }))).toBe(
      false,
    );
  });

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

  it("humanizes a cron and falls back to the raw string", () => {
    expect(getWorkflowScheduleLabel(makeWorkflow({}))).toBeNull();
    expect(
      getWorkflowScheduleLabel(makeWorkflow({ schedule_cron: "not-a-cron" })),
    ).toBe("not-a-cron");
  });
});
