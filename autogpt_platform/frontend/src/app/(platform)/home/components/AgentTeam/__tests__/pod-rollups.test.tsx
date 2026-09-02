import {
  getListExpertPodsMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetHomeDashboardResponseMock } from "@/app/api/__generated__/endpoints/home/home.msw";
import { getListTasksMockHandler } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { DelegatedTaskStatus } from "@/app/api/__generated__/models/delegatedTaskStatus";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";
import { AgentTeam } from "../AgentTeam";

function makeAgent(
  id: string,
  name: string,
  spendCents: number,
): HomeAgentStatus {
  return {
    expert: { id, name, role: "Specialist", avatar_url: null },
    status: "ready",
    detail: "Ready for work",
    spend_cents: spendCents,
  };
}

function makeExpert(id: string, name: string, podId: string | null): Expert {
  return {
    id,
    name,
    avatar_url: null,
    role: "Specialist",
    tagline: null,
    bio: null,
    skills: [],
    identity: "",
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

function makeTask(
  id: string,
  ownerId: string,
  ownerName: string,
  status: DelegatedTaskStatus,
): DelegatedTask {
  return {
    id,
    title: `Task ${id}`,
    spec: "Do the thing",
    status,
    acceptance: "ACCEPTED",
    created_by_type: "USER",
    created_by_id: null,
    owner: {
      id: ownerId,
      name: ownerName,
      avatar_url: null,
      role: "Specialist",
    },
    parent_task_id: null,
    root_task_id: null,
    origin_session_id: null,
    ancestor_expert_ids: [],
    handoff_count: 0,
    revision_count: 0,
    spend_total: 0,
    outcome_summary: null,
    amendments: [],
    created_at: new Date("2026-08-30T10:00:00Z"),
    updated_at: new Date("2026-08-30T10:00:00Z"),
  };
}

const pods: ExpertPod[] = [
  { id: "pod-growth", name: "Growth", created_at: new Date("2026-08-01") },
  { id: "pod-ops", name: "Ops", created_at: new Date("2026-08-01") },
];

const sixAgents = [
  makeAgent("e1", "Expert One", 100),
  makeAgent("e2", "Expert Two", 200),
  makeAgent("e3", "Expert Three", 300),
  makeAgent("e4", "Expert Four", 0),
  makeAgent("e5", "Expert Five", 0),
  makeAgent("e6", "Expert Six", 0),
];

const sixExperts = [
  makeExpert("e1", "Expert One", "pod-growth"),
  makeExpert("e2", "Expert Two", "pod-growth"),
  makeExpert("e3", "Expert Three", "pod-growth"),
  makeExpert("e4", "Expert Four", "pod-ops"),
  makeExpert("e5", "Expert Five", "pod-ops"),
  makeExpert("e6", "Expert Six", null),
];

const tasks = [
  makeTask("t1", "e1", "Expert One", "WORKING"),
  makeTask("t2", "e2", "Expert Two", "QUEUED"),
  makeTask("t3", "e3", "Expert Three", "WAITING_USER"),
  makeTask("t4", "e4", "Expert Four", "WORKING"),
  makeTask("t5", "e5", "Expert Five", "DONE"),
];

function dashboardWith(agents: HomeAgentStatus[]) {
  return getGetHomeDashboardResponseMock({
    agents,
    team: {
      total: agents.length,
      ready: agents.length,
      working: 0,
      needs_attention: 0,
      spend_cents: 0,
    },
  });
}

describe("AgentTeam pod rollups", () => {
  test("collapses more than five experts into pod rollups", async () => {
    server.use(
      getListExpertsMockHandler(sixExperts),
      getListExpertPodsMockHandler(pods),
      getListTasksMockHandler(tasks),
    );

    render(<AgentTeam dashboard={dashboardWith(sixAgents)} />);

    const growthRow = await screen.findByRole("button", { name: /Growth/ });
    expect(growthRow.textContent).toContain("2 active");
    expect(growthRow.textContent).toContain("1 need you");
    expect(growthRow.textContent).toContain("$6.00 this week");

    const opsRow = screen.getByRole("button", { name: /Ops/ });
    expect(opsRow.textContent).toContain("1 active");

    expect(screen.getByRole("button", { name: /Unassigned/ })).toBeDefined();
    expect(screen.queryByText("Expert One")).toBeNull();
    expect(screen.queryByText("Expert Six")).toBeNull();
  });

  test("expands a pod rollup to reveal its expert rows", async () => {
    const user = userEvent.setup();
    server.use(
      getListExpertsMockHandler(sixExperts),
      getListExpertPodsMockHandler(pods),
      getListTasksMockHandler(tasks),
    );

    render(<AgentTeam dashboard={dashboardWith(sixAgents)} />);

    const growthRow = await screen.findByRole("button", { name: /Growth/ });
    expect(growthRow.getAttribute("aria-expanded")).toBe("false");
    await user.click(growthRow);

    expect(growthRow.getAttribute("aria-expanded")).toBe("true");
    expect(await screen.findByText("Expert One")).toBeDefined();
    expect(screen.getByText("Expert Two")).toBeDefined();
    expect(screen.getByText("Expert Three")).toBeDefined();
    expect(screen.queryByText("Expert Four")).toBeNull();

    await user.click(growthRow);
    expect(screen.queryByText("Expert One")).toBeNull();
  });

  test("keeps plain expert rows with five or fewer experts", async () => {
    render(<AgentTeam dashboard={dashboardWith(sixAgents.slice(0, 5))} />);

    expect(await screen.findByText("Expert One")).toBeDefined();
    expect(screen.getByText("Expert Two")).toBeDefined();
    expect(screen.getByText("Expert Three")).toBeDefined();
    expect(screen.queryByRole("button", { name: /Growth/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /Unassigned/ })).toBeNull();
  });
});
