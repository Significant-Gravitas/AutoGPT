import { fulfillJSON, type RouteHandlers } from "./routes";

const FIXED_TIME = "2026-01-05T09:00:00.000Z";
const COLUMNS = 10;
const STORE_VALUE_BLOCK = "1ff065e9-88e8-4358-9d82-8dc91f622ba9";
const OTHER_BLOCKS = [
  { id: "b1ab9b19-67a6-406d-abf5-2dba76d00c79", input: { a: 1, b: 2 } },
  { id: "db7d8f02-2f44-4c55-ab7a-eae0941f0c30", input: { format: "{{a}}" } },
  { id: "3060088f-6ed9-4928-9ba7-9c92823a7ccd", input: { match: "x" } },
  { id: "a912d5c7-6e00-4542-b2a9-8034136930e4", input: {} },
  { id: "715696a0-e1da-45c8-b209-c2fa9c3b0be6", input: {} },
];

export const BUILDER_GRAPH_ID = "00000000-0000-4000-8000-0000000000b1";
export const BUILDER_EXECUTION_ID = "00000000-0000-4000-8000-0000000000b2";

function nodeID(index: number) {
  return `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`;
}

function isStoreValue(index: number) {
  return index % 5 !== 4;
}

export function buildGraph(userID: string, nodeCount = 50) {
  const nodes = Array.from({ length: nodeCount }, (_, index) => {
    const other = OTHER_BLOCKS[Math.floor(index / 5) % OTHER_BLOCKS.length];
    return {
      id: nodeID(index),
      block_id: isStoreValue(index) ? STORE_VALUE_BLOCK : other.id,
      input_default: isStoreValue(index) ? { input: `value ${index}` } : {},
      metadata: {
        position: {
          x: (index % COLUMNS) * 420,
          y: Math.floor(index / COLUMNS) * 360,
        },
      },
      input_links: [],
      output_links: [],
      graph_id: BUILDER_GRAPH_ID,
      graph_version: 1,
      webhook_id: null,
    };
  });
  const links = nodes.slice(0, -1).flatMap((node, index) =>
    isStoreValue(index) && isStoreValue(index + 1)
      ? [
          {
            id: `00000000-0000-4000-9000-${String(index).padStart(12, "0")}`,
            source_id: node.id,
            sink_id: nodes[index + 1].id,
            source_name: "output",
            sink_name: "input",
            is_static: true,
          },
        ]
      : [],
  );
  return {
    id: BUILDER_GRAPH_ID,
    version: 1,
    is_active: true,
    name: "Perf: 50-step workflow",
    description: "Fixture for the builder-drag-running scenario",
    instructions: null,
    recommended_schedule_cron: null,
    forked_from_id: null,
    forked_from_version: null,
    user_id: userID,
    created_at: FIXED_TIME,
    organization_id: null,
    team_id: null,
    nodes,
    links,
    sub_graphs: [],
    input_schema: { type: "object", properties: {}, required: [] },
    output_schema: { type: "object", properties: {}, required: [] },
    has_external_trigger: false,
    has_human_in_the_loop: false,
    has_sensitive_action: false,
    trigger_setup_info: null,
    credentials_input_schema: {
      type: "object",
      properties: {},
      required: [],
    },
  };
}

function nodeStatus(index: number) {
  if (index < 30) return "COMPLETED";
  if (index < 33) return "RUNNING";
  return "QUEUED";
}

export function buildRunningExecution(
  graph: ReturnType<typeof buildGraph>,
  userID: string,
) {
  const node_executions = graph.nodes.slice(0, 40).map((node, index) => ({
    user_id: userID,
    graph_id: graph.id,
    graph_version: graph.version,
    graph_exec_id: BUILDER_EXECUTION_ID,
    node_exec_id: `00000000-0000-4000-a000-${String(index).padStart(12, "0")}`,
    node_id: node.id,
    block_id: node.block_id,
    status: nodeStatus(index),
    input_data: node.input_default,
    output_data:
      nodeStatus(index) === "COMPLETED" ? { output: [`value ${index}`] } : {},
    add_time: FIXED_TIME,
    queue_time: FIXED_TIME,
    start_time: index < 33 ? FIXED_TIME : null,
    end_time: index < 30 ? FIXED_TIME : null,
  }));
  return {
    id: BUILDER_EXECUTION_ID,
    user_id: userID,
    graph_id: graph.id,
    graph_version: graph.version,
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
    outputs: {},
    node_executions,
  };
}

export function createBuilderFixture(userID: string) {
  const graph = buildGraph(userID);
  const execution = buildRunningExecution(graph, userID);
  const handlers: RouteHandlers = {
    graph: async (route, match) => {
      if (match[1] !== graph.id || route.request().method() !== "GET") {
        return route.fallback();
      }
      await fulfillJSON(route, graph);
    },
    graphExecution: async (route, match) => {
      if (match[1] !== graph.id || match[2] !== execution.id) {
        return route.fallback();
      }
      await fulfillJSON(route, execution);
    },
  };
  return { handlers, graph, execution };
}
