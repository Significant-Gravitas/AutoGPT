import { buildLibraryAgents } from "./library-fixture";
import { fulfillJSON, type RouteHandlers } from "./routes";

const RUN_COUNT = 20;
const PAGE_SIZE = 20;
const FIRST_STARTED_AT = Date.parse("2026-01-05T09:00:00.000Z");

function runID(index: number) {
  return `00000000-0000-4000-c300-${String(index).padStart(12, "0")}`;
}

type Agent = ReturnType<typeof buildLibraryAgents>[number];

function buildRun(
  agent: Agent,
  userID: string,
  index: number,
  running = false,
) {
  const startedAt = FIRST_STARTED_AT + index * 60_000;
  return {
    id: runID(index),
    user_id: userID,
    graph_id: agent.graph_id,
    graph_version: agent.graph_version,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: null,
    preset_id: null,
    status: running ? "RUNNING" : "COMPLETED",
    started_at: new Date(startedAt).toISOString(),
    ended_at: running ? null : new Date(startedAt + 30_000).toISOString(),
    is_shared: false,
    share_token: null,
    is_dry_run: false,
    stats: { node_exec_count: 4, node_error_count: 0, cost: 0, duration: 30 },
  };
}

export type LibraryAgentRun = ReturnType<typeof buildRun>;

/**
 * One workflow with 20 runs, newest first. The newest is RUNNING, so its
 * detail view polls; each poll answers with one more finished node, as a run
 * that makes progress would. `addRun()` puts a new RUNNING run at the top of
 * the list, as starting a task does.
 */
export function createLibraryAgentFixture(userID: string) {
  const [agent] = buildLibraryAgents(1);
  let nextIndex = RUN_COUNT;
  const runs: LibraryAgentRun[] = Array.from(
    { length: RUN_COUNT },
    (_, index) => buildRun(agent, userID, index, index === RUN_COUNT - 1),
  ).reverse();
  const selectedRun = runs[0];
  const stats = { detailRequests: 0, listRequests: 0 };

  function addRun() {
    const run = buildRun(agent, userID, nextIndex, true);
    nextIndex += 1;
    runs.unshift(run);
    return run;
  }

  const handlers: RouteHandlers = {
    libraryAgent: async (route, match) => {
      if (match[1] !== agent.id || route.request().method() !== "GET") {
        return route.fallback();
      }
      await fulfillJSON(route, agent);
    },
    graphExecutions: async (route, match) => {
      if (match[1] !== agent.graph_id || route.request().method() !== "GET") {
        return route.fallback();
      }
      stats.listRequests += 1;
      const params = new URL(route.request().url()).searchParams;
      const page = Number(params.get("page") ?? 1);
      const pageSize = Number(params.get("page_size") ?? PAGE_SIZE);
      await fulfillJSON(route, {
        executions: runs.slice((page - 1) * pageSize, page * pageSize),
        pagination: {
          total_items: runs.length,
          total_pages: Math.ceil(runs.length / pageSize),
          current_page: page,
          page_size: pageSize,
        },
      });
    },
    graphExecution: async (route, match) => {
      if (match[1] !== agent.graph_id || match[2] !== selectedRun.id) {
        return route.fallback();
      }
      stats.detailRequests += 1;
      await fulfillJSON(route, {
        ...selectedRun,
        stats: {
          ...selectedRun.stats,
          node_exec_count:
            selectedRun.stats.node_exec_count + stats.detailRequests,
        },
        outputs: {},
        node_executions: [],
      });
    },
    graphSchedules: async (route, match) => {
      if (match[1] !== agent.graph_id || route.request().method() !== "GET") {
        return route.fallback();
      }
      await fulfillJSON(route, []);
    },
    libraryPresets: async (route) => {
      if (route.request().method() !== "GET") return route.fallback();
      await fulfillJSON(route, {
        presets: [],
        pagination: {
          total_items: 0,
          total_pages: 0,
          current_page: 1,
          page_size: 100,
        },
      });
    },
  };
  return { handlers, agent, runs, selectedRun, stats, addRun };
}
