import { fulfillJSON, type RouteHandlers } from "./routes";

const FIXED_TIME = "2026-01-05T09:00:00.000Z";

function fixtureID(group: string, index: number) {
  return `00000000-0000-4000-${group}-${String(index).padStart(12, "0")}`;
}

export function buildLibraryAgents(count = 20) {
  return Array.from({ length: count }, (_, index) => ({
    id: fixtureID("c000", index),
    graph_id: fixtureID("c100", index),
    graph_version: 1,
    image_url: null,
    creator_name: "Perf Harness",
    creator_image_url: "",
    status: "COMPLETED",
    created_at: FIXED_TIME,
    updated_at: FIXED_TIME,
    last_run_at: FIXED_TIME,
    name: `Perf workflow ${String(index + 1).padStart(2, "0")}`,
    description: "Fixture for the library-running scenario",
    instructions: null,
    input_schema: { type: "object", properties: {}, required: [] },
    output_schema: { type: "object", properties: {}, required: [] },
    credentials_input_schema: null,
    has_external_trigger: false,
    has_human_in_the_loop: false,
    has_sensitive_action: false,
    trigger_setup_info: null,
    new_output: false,
    execution_count: 12,
    success_rate: null,
    avg_correctness_score: null,
    recent_executions: [],
    can_access_graph: true,
    is_latest_version: true,
    is_favorite: false,
    folder_id: null,
    folder_name: null,
    is_hidden: false,
    recommended_schedule_cron: null,
    is_scheduled: false,
    next_scheduled_run: null,
    settings: {
      human_in_the_loop_safe_mode: true,
      sensitive_action_safe_mode: false,
      builder_chat_session_id: null,
    },
    marketplace_listing: null,
  }));
}

export function buildRunningExecutions(
  agents: ReturnType<typeof buildLibraryAgents>,
  userID: string,
) {
  return agents.map((agent, index) => ({
    id: fixtureID("c200", index),
    user_id: userID,
    graph_id: agent.graph_id,
    graph_version: agent.graph_version,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: null,
    preset_id: null,
    status: "RUNNING",
    started_at: FIXED_TIME,
    ended_at: null,
    is_shared: false,
    share_token: null,
    is_dry_run: false,
    stats: null,
  }));
}

export function createLibraryFixture(userID: string) {
  const agents = buildLibraryAgents();
  const executions = buildRunningExecutions(agents, userID);
  const handlers: RouteHandlers = {
    libraryAgents: async (route) => {
      if (route.request().method() !== "GET") return route.fallback();
      const params = new URL(route.request().url()).searchParams;
      const pageSize = Number(params.get("page_size") ?? agents.length);
      await fulfillJSON(route, {
        agents: agents.slice(0, pageSize),
        pagination: {
          total_items: agents.length,
          total_pages: 1,
          current_page: 1,
          page_size: pageSize,
        },
      });
    },
    executions: async (route) => {
      if (route.request().method() !== "GET") return route.fallback();
      await fulfillJSON(route, executions);
    },
  };
  return { handlers, agents, executions };
}
