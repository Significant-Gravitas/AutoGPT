import { getGetV2GetSpecificBlocksMockHandler200 } from "@/app/api/__generated__/endpoints/default/default.msw";
import { getGetV1GetSpecificGraphQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV1GetExecutionDetailsMockHandler200,
  getGetV1GetSpecificGraphMockHandler200,
  getGetV1GetSpecificGraphResponseMock200,
  getGetV1ListUserGraphsMockHandler200,
  getPostV1ExecuteGraphAgentMockHandler200,
  getPostV1ExecuteGraphAgentResponseMock200,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import type { Graph } from "@/app/api/__generated__/models/graph";
import type { GraphExecutionWithNodes } from "@/app/api/__generated__/models/graphExecutionWithNodes";
import type { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { setTabIntroSeen } from "../../components/TabIntroCard/helpers";
import BuilderPage from "../page";
import { useEdgeStore } from "../stores/edgeStore";
import { useGraphStore } from "../stores/graphStore";
import { useHistoryStore } from "../stores/historyStore";
import { useNodeStore } from "../stores/nodeStore";

const USER_ID = "user-1";
const GRAPH_ID = "graph-1";
const BLANK_GRAPH_ID = "graph-2";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "builder@example.com" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

// The websocket only reports run progress, which these tests don't follow;
// a stand-in keeps the real client from opening one.
vi.mock("@/lib/autogpt-server-api/context", () => {
  const api = {
    connectWebSocket: () => Promise.resolve(),
    disconnectWebSocket: () => undefined,
    onWebSocketMessage: () => () => undefined,
    onWebSocketConnect: () => () => undefined,
    onWebSocketDisconnect: () => () => undefined,
    subscribeToGraphExecution: () => Promise.resolve(),
  };
  return {
    BackendAPIProvider: ({ children }: { children: unknown }) => children,
    useBackendAPI: () => api,
  };
});

// IndexedDB does not exist in happy-dom; drafts live in memory instead.
const drafts = vi.hoisted(() => new Map<string, { id: string }>());
vi.mock("@/lib/dexie/db", () => {
  return {
    DRAFT_EXPIRY_MS: 24 * 60 * 60 * 1000,
    db: {
      drafts: {
        put: async (draft: { id: string }) => {
          drafts.set(draft.id, draft);
        },
        get: async (id: string) => drafts.get(id),
        delete: async (id: string) => {
          drafts.delete(id);
        },
      },
    },
    cleanupExpiredDrafts: async () => 0,
  };
});

const STORE_VALUE_BLOCK: BlockInfo = {
  id: "1ff065e9-88e8-4358-9d82-8dc91f622ba9",
  name: "Store Value",
  description: "Forwards its input as its output.",
  inputSchema: {
    type: "object",
    properties: { input: { title: "Input", type: "string" } },
    required: ["input"],
  },
  outputSchema: { type: "object", properties: { output: { title: "Output" } } },
  costs: [],
  categories: [],
  contributors: [],
  staticOutput: true,
  uiType: "Standard",
};

const AGENT_OUTPUT_BLOCK: BlockInfo = {
  id: "363ae599-353e-4804-937e-b2ee3cef3da4",
  name: "Agent Output",
  description: "Shows a value as the workflow's output.",
  inputSchema: {
    type: "object",
    properties: {
      value: { title: "Value" },
      name: { title: "Name", type: "string" },
    },
    required: ["name"],
  },
  outputSchema: { type: "object", properties: { output: { title: "Output" } } },
  costs: [],
  categories: [],
  contributors: [],
  staticOutput: false,
  uiType: "Output",
};

// The node ids the backend hands out, in order. They sort the other way from
// the canvas's own "1" and "2", as random ids do about half the time, which
// makes the builder see a change and send a save with the canvas's ids.
const SERVER_NODE_IDS = [
  "c019852e-7d1c-4d6e-9a43-0f3b8a2e5d11",
  "1c21c084-35a8-4f0e-8b6f-2d9c7e4a1b22",
  "9b4013b5-6e2f-4a8d-b1c7-5f0e3d9a8c33",
  "673b4233-0a9d-4c5b-a8e6-7b1f2c3d4e44",
];

// What the builder's run poll reads back once a run has been started.
const QUEUED_RUN: GraphExecutionWithNodes = {
  id: "execution-1",
  user_id: USER_ID,
  graph_id: GRAPH_ID,
  graph_version: 1,
  inputs: {},
  credential_inputs: null,
  nodes_input_masks: null,
  preset_id: null,
  status: "QUEUED",
  stats: null,
  outputs: {},
  node_executions: [],
};

// A saved two-block workflow, before the backend gives its nodes their ids.
const SAVED_WORKFLOW: Graph = {
  name: "Saved Agent",
  description: "",
  nodes: [
    {
      id: "a",
      block_id: STORE_VALUE_BLOCK.id,
      input_default: { input: "hello" },
      metadata: { position: { x: 0, y: 0 } },
    },
    {
      id: "b",
      block_id: AGENT_OUTPUT_BLOCK.id,
      input_default: { name: "result" },
      metadata: { position: { x: 400, y: 0 } },
    },
  ],
  links: [
    {
      source_id: "a",
      sink_id: "b",
      source_name: "output",
      sink_name: "value",
      is_static: true,
    },
  ],
};

// What the builder chat creates when it is opened without a workflow.
const BLANK_WORKFLOW: Graph = {
  name: "New Workflow",
  description: "",
  nodes: [],
  links: [],
};

// The graph request for a version, the one for a URL without a version (the
// latest), and the blocks request.
type LoadRequest = "graph" | "latest graph" | "blocks";

// Waits that span requests and renders get room for a slow CI runner.
const NETWORK_WAIT = { timeout: 10_000 };

// A stand-in for the graphs API that saves the way the backend does: every
// save is a new version whose nodes get fresh ids, links follow the nodes
// they name, and a link naming a node the save doesn't have is refused.
const backend = {
  graphs: {} as Record<string, GraphModel[]>,
  nodeIDs: [] as string[],
  creates: 0,
  updates: [] as Graph[],
  refusals: [] as string[],
  executions: [] as { graphID: string; graphVersion: number }[],
  requests: { graph: 0, "latest graph": 0, blocks: 0 },
  held: null as LoadRequest | null,
  release: () => {},
};
let heldRequest = Promise.resolve();

function saveVersion(graph: Graph, graphID = GRAPH_ID) {
  const versions = (backend.graphs[graphID] ??= []);
  const version = versions.length + 1;
  const newIDs = new Map(
    (graph.nodes ?? []).map((node) => [node.id, backend.nodeIDs.shift()!]),
  );
  const nodes = (graph.nodes ?? []).map((node) => ({
    ...node,
    id: newIDs.get(node.id),
    graph_id: graphID,
    graph_version: version,
    input_links: [],
    output_links: [],
  }));
  const links = (graph.links ?? []).map((link, index) => ({
    ...link,
    id: `link-${version}-${index}`,
    source_id: newIDs.get(link.source_id) ?? link.source_id,
    sink_id: newIDs.get(link.sink_id) ?? link.sink_id,
  }));
  const available = [...newIDs.values()];
  for (const link of links) {
    for (const nodeID of [link.source_id, link.sink_id]) {
      if (!available.includes(nodeID)) {
        return `Link ('${link.source_id}', '${link.source_name}') <-> ('${link.sink_id}', '${link.sink_name}'), ${nodeID} is invalid node id, available nodes: [${available.join(", ")}]`;
      }
    }
  }
  const saved = getGetV1GetSpecificGraphResponseMock200({
    id: graphID,
    version,
    is_active: true,
    name: graph.name,
    description: graph.description,
    user_id: USER_ID,
    nodes,
    links,
    sub_graphs: [],
    input_schema: { type: "object", properties: {} },
    output_schema: { type: "object", properties: {} },
    credentials_input_schema: { type: "object", properties: {} },
    has_external_trigger: false,
    has_human_in_the_loop: false,
    has_sensitive_action: false,
    trigger_setup_info: null,
  });
  versions.push(saved);
  return saved;
}

function refuse(method: string, path: string, detail: string) {
  backend.refusals.push(detail);
  return HttpResponse.json(
    {
      message: `Failed to process ${method} ${path}`,
      detail,
      hint: "Adjust the request and retry.",
    },
    { status: 400 },
  );
}

async function answerLoad(request: LoadRequest) {
  backend.requests[request]++;
  if (backend.held === request) await heldRequest;
}

function hold(request: LoadRequest) {
  backend.held = request;
  heldRequest = new Promise((resolve) => {
    backend.release = () => {
      backend.held = null;
      resolve();
    };
  });
}

// Moves the open builder to another workflow without leaving the page, as
// the builder chat does for the blank workflow it creates, or the browser's
// back button does.
function OpenWorkflowButton({
  graphID,
  version = 1,
}: {
  graphID: string;
  version?: number | null;
}) {
  const [, setQueryStates] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });
  return (
    <button
      type="button"
      onClick={() => setQueryStates({ flowID: graphID, flowVersion: version })}
    >
      {version === null
        ? `Open ${graphID} latest`
        : version === 1
          ? `Open ${graphID}`
          : `Open ${graphID} v${version}`}
    </button>
  );
}

function renderBuilder(searchParams = "") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const builder = render(
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter
        searchParams={searchParams}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <OnboardingProvider>
          <TooltipProvider>
            <BuilderPage />
            <OpenWorkflowButton graphID={GRAPH_ID} />
            <OpenWorkflowButton graphID={GRAPH_ID} version={null} />
            <OpenWorkflowButton graphID={GRAPH_ID} version={2} />
            <OpenWorkflowButton graphID={BLANK_GRAPH_ID} />
            <Toaster />
          </TooltipProvider>
        </OnboardingProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>,
  );
  return { ...builder, queryClient };
}

// A two-block workflow, added the way the block menu and a drawn connection
// add them.
function buildWorkflow() {
  act(() => {
    const nodes = useNodeStore.getState();
    const source = nodes.addBlock(STORE_VALUE_BLOCK, { input: "hello" });
    const sink = nodes.addBlock(AGENT_OUTPUT_BLOCK, { name: "result" });
    useEdgeStore.getState().addEdge({
      source: source.id,
      target: sink.id,
      sourceHandle: "output",
      targetHandle: "value",
      data: { isStatic: true },
    });
  });
}

function runButton() {
  const button = document.querySelector<HTMLButtonElement>(
    '[data-id="run-graph-button"]',
  );
  if (!button) throw new Error("The builder shows no Run button");
  return button;
}

async function clickRunOnceItIsEnabled(
  user: ReturnType<typeof userEvent.setup>,
) {
  await waitFor(() => expect(runButton().disabled).toBe(false), NETWORK_WAIT);
  await user.click(runButton());
}

// Node ids that links name but the nodes beside them don't have.
function strayLinkEnds(graph: Graph) {
  const nodeIDs = new Set((graph.nodes ?? []).map((node) => node.id));
  return (graph.links ?? [])
    .flatMap((link) => [link.source_id, link.sink_id])
    .filter((nodeID) => !nodeIDs.has(nodeID));
}

function canvas(): Graph {
  return {
    name: "canvas",
    description: "",
    nodes: useNodeStore.getState().getBackendNodes(),
    links: useEdgeStore.getState().getBackendLinks(),
  };
}

// Moves the Store Value block and edits its input, as a user does before
// saving, and returns the canvas with those edits.
function editTheCanvas() {
  const block = useNodeStore
    .getState()
    .nodes.find((node) => node.data.block_id === STORE_VALUE_BLOCK.id)!;
  act(() => {
    useNodeStore.getState().onNodesChange([
      {
        type: "position",
        id: block.id,
        position: { x: 120, y: 80 },
        dragging: false,
      },
    ]);
    useNodeStore.getState().updateNodeData(block.id, {
      hardcodedValues: { ...block.data.hardcodedValues, input: "edited" },
    });
  });
  const edited = canvas();
  expect(edited.nodes).toContainEqual(
    expect.objectContaining({
      input_default: { input: "edited" },
      metadata: expect.objectContaining({ position: { x: 120, y: 80 } }),
    }),
  );
  return edited;
}

async function waitForTheRun() {
  await waitFor(
    () =>
      expect(
        backend.executions.length + backend.refusals.length,
      ).toBeGreaterThan(0),
    NETWORK_WAIT,
  );
}

beforeEach(() => {
  drafts.clear();
  backend.graphs = {};
  backend.nodeIDs = [...SERVER_NODE_IDS];
  backend.creates = 0;
  backend.updates = [];
  backend.refusals = [];
  backend.executions = [];
  backend.requests = { graph: 0, "latest graph": 0, blocks: 0 };
  backend.held = null;
  setTabIntroSeen("build", USER_ID);

  server.use(
    http.post("/api/proxy/api/graphs", async ({ request }) => {
      backend.creates++;
      const { graph } = (await request.json()) as { graph: Graph };
      const saved = saveVersion(graph);
      return typeof saved === "string"
        ? refuse("POST", "/api/graphs", saved)
        : HttpResponse.json(saved);
    }),
    http.put("/api/proxy/api/graphs/:graphID", async ({ request, params }) => {
      const graphID = String(params.graphID);
      const graph = (await request.json()) as Graph;
      backend.updates.push(graph);
      const saved = saveVersion(graph, graphID);
      return typeof saved === "string"
        ? refuse("PUT", `/api/graphs/${graphID}`, saved)
        : HttpResponse.json({ graph: saved, skipped_webhook_presets: [] });
    }),
    getGetV1GetSpecificGraphMockHandler200(async ({ request, params }) => {
      const version = new URL(request.url).searchParams.get("version");
      await answerLoad(version ? "graph" : "latest graph");
      const versions = backend.graphs[String(params.graphId)] ?? [];
      return versions[version ? Number(version) - 1 : versions.length - 1];
    }),
    getGetV2GetSpecificBlocksMockHandler200(async () => {
      await answerLoad("blocks");
      return [STORE_VALUE_BLOCK, AGENT_OUTPUT_BLOCK];
    }),
    getGetV1ListUserGraphsMockHandler200([]),
    getPostV1ExecuteGraphAgentMockHandler200(({ params }) => {
      const run = {
        graphID: String(params.graphId),
        graphVersion: Number(params.graphVersion),
      };
      backend.executions.push(run);
      return getPostV1ExecuteGraphAgentResponseMock200({
        id: "execution-1",
        user_id: USER_ID,
        graph_id: run.graphID,
        graph_version: run.graphVersion,
        status: "QUEUED",
      });
    }),
    getGetV1GetExecutionDetailsMockHandler200(QUEUED_RUN),
  );
});

afterEach(() => {
  backend.release();
  useNodeStore.getState().setNodes([]);
  useNodeStore.getState().setNodeCounter(0);
  useEdgeStore.getState().setEdges([]);
  useGraphStore.getState().reset();
  useHistoryStore.getState().clear();
});

describe("builder Run while the workflow it runs is still loading", () => {
  test.each<LoadRequest>(["graph", "blocks"])(
    "Run right after a new workflow's first save, with the saved copy's %s request still out, runs that workflow",
    async (request) => {
      const user = userEvent.setup();
      renderBuilder();
      buildWorkflow();
      hold(request);

      await user.click(screen.getByTestId("save-control-save-button"));
      await user.type(
        await screen.findByTestId("save-control-name-input"),
        "Smoke Run Agent",
      );
      await user.click(screen.getByRole("button", { name: "Save Workflow" }));
      await waitFor(
        () => expect(backend.requests[request]).toBeGreaterThan(0),
        NETWORK_WAIT,
      );
      expect(backend.graphs[GRAPH_ID]).toHaveLength(1);
      expect(strayLinkEnds(canvas())).toEqual([]);

      // Pressed twice, as fast as a double-click: the wait disables Run on the
      // first press, so the second one starts nothing.
      await waitFor(
        () => expect(runButton().disabled).toBe(false),
        NETWORK_WAIT,
      );
      await user.dblClick(runButton());
      expect(runButton().disabled, "Run can be pressed again").toBe(true);
      backend.release();
      await waitForTheRun();

      expect(backend.creates, "a second workflow was created").toBe(1);
      expect(backend.updates, "running it saved a new version").toEqual([]);
      expect(backend.refusals).toEqual([]);
      expect(backend.executions).toEqual([
        { graphID: GRAPH_ID, graphVersion: 1 },
      ]);
      expect(screen.queryByText("Error saving workflow")).toBeNull();
    },
  );

  test.each<LoadRequest>(["graph", "blocks"])(
    "Run on a saved workflow that is still opening, with its %s request still out, runs it as saved",
    async (request) => {
      saveVersion(SAVED_WORKFLOW);
      const user = userEvent.setup();
      hold(request);
      renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);

      await waitFor(
        () => expect(backend.requests[request]).toBeGreaterThan(0),
        NETWORK_WAIT,
      );
      expect(strayLinkEnds(canvas())).toEqual([]);

      await clickRunOnceItIsEnabled(user);
      backend.release();
      await waitForTheRun();

      expect(backend.creates, "a new workflow was created").toBe(0);
      expect(backend.updates, "running it saved a new version").toEqual([]);
      expect(backend.executions).toEqual([
        { graphID: GRAPH_ID, graphVersion: 1 },
      ]);
    },
  );
});

describe("builder canvas when another workflow opens in place", () => {
  test("opening an empty workflow empties the canvas, undo included, so Run copies none of the other workflow's blocks into it", async () => {
    saveVersion(SAVED_WORKFLOW);
    saveVersion(BLANK_WORKFLOW, BLANK_GRAPH_ID);
    const user = userEvent.setup();
    renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(() => expect(canvas().nodes).toHaveLength(2), NETWORK_WAIT);
    // Undo starts from the workflow as it loaded.
    await waitFor(
      () => expect(useHistoryStore.getState().canUndo()).toBe(false),
      NETWORK_WAIT,
    );

    await user.click(
      screen.getByRole("button", { name: `Open ${BLANK_GRAPH_ID}` }),
    );
    await waitFor(
      () =>
        expect(useGraphStore.getState().loadedGraph?.id).toBe(BLANK_GRAPH_ID),
      NETWORK_WAIT,
    );
    await user.keyboard("{Control>}z{/Control}");
    expect(canvas().nodes).toEqual([]);
    expect(canvas().links).toEqual([]);

    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(
      backend.updates.flatMap((update) => update.nodes ?? []),
      "a save copied the other workflow's blocks into it",
    ).toEqual([]);
    expect(backend.creates).toBe(0);
    expect(backend.executions).toEqual([
      { graphID: BLANK_GRAPH_ID, graphVersion: 1 },
    ]);
  });

  test("a blank workflow made for an unsaved canvas keeps its blocks, and Run saves them into it", async () => {
    saveVersion(BLANK_WORKFLOW, BLANK_GRAPH_ID);
    const user = userEvent.setup();
    renderBuilder();
    buildWorkflow();

    await user.click(
      screen.getByRole("button", { name: `Open ${BLANK_GRAPH_ID}` }),
    );
    await waitFor(
      () =>
        expect(useGraphStore.getState().loadedGraph?.id).toBe(BLANK_GRAPH_ID),
      NETWORK_WAIT,
    );
    expect(canvas().nodes).toHaveLength(2);

    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(backend.refusals).toEqual([]);
    expect(backend.updates.map((update) => update.nodes?.length)).toEqual([2]);
    expect(backend.executions).toEqual([
      { graphID: BLANK_GRAPH_ID, graphVersion: 2 },
    ]);
  });
});

describe("builder canvas when the version it shows is fetched again", () => {
  // Window focus or an invalidation fetches the open version again, and every
  // response is a new object. The backend returns a version's nodes and links
  // unordered, and can change them in place (a webhook, a model migration).
  test.each<[string, (graph: GraphModel) => GraphModel]>([
    ["unchanged", (graph) => graph],
    [
      "with changed links",
      (graph) => ({
        ...graph,
        links: graph.links?.map((link) => ({
          ...link,
          is_static: !link.is_static,
        })),
      }),
    ],
    [
      "with its nodes in another order",
      (graph) => ({ ...graph, nodes: [...(graph.nodes ?? [])].reverse() }),
    ],
  ])(
    "fetching it again %s keeps the user's unsaved edits",
    async (_, change) => {
      saveVersion(SAVED_WORKFLOW);
      const { queryClient } = renderBuilder(
        `?flowID=${GRAPH_ID}&flowVersion=1`,
      );
      await waitFor(
        () => expect(useGraphStore.getState().loadedGraph?.version).toBe(1),
        NETWORK_WAIT,
      );
      const edited = editTheCanvas();
      const loadedBefore = useGraphStore.getState().loadedGraph;

      backend.graphs[GRAPH_ID][0] = change(backend.graphs[GRAPH_ID][0]);
      await act(() =>
        queryClient.invalidateQueries({
          queryKey: getGetV1GetSpecificGraphQueryKey(GRAPH_ID, { version: 1 }),
        }),
      );
      await waitFor(
        () =>
          expect(useGraphStore.getState().loadedGraph).not.toBe(loadedBefore),
        NETWORK_WAIT,
      );

      expect(canvas().nodes, "the fetch replaced the edited blocks").toEqual(
        edited.nodes,
      );
      expect(canvas().links, "the fetch replaced the canvas's links").toEqual(
        edited.links,
      );
    },
  );
});

describe("builder when fetching the open workflow again fails", () => {
  // A one-off refusal, as a network blip or an expired session gives. A 4xx,
  // so the request isn't tried again.
  function refuseTheNextGraphFetch() {
    server.use(
      http.get(
        `/api/proxy/api/graphs/${GRAPH_ID}`,
        () => HttpResponse.json({ detail: "Not allowed" }, { status: 403 }),
        { once: true },
      ),
    );
  }

  async function fetchVersion1Again(queryClient: QueryClient) {
    const key = getGetV1GetSpecificGraphQueryKey(GRAPH_ID, { version: 1 });
    await act(() => queryClient.invalidateQueries({ queryKey: key }));
    await waitFor(() =>
      expect(queryClient.getQueryState(key)?.status).toBe("error"),
    );
    // React Query tells the builder about it on the next tick.
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
  }

  test("a Run waiting for the canvas still runs once the blocks are in", async () => {
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    hold("blocks");
    const { queryClient } = renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(
      () => expect(backend.requests.blocks).toBeGreaterThan(0),
      NETWORK_WAIT,
    );
    await clickRunOnceItIsEnabled(user);

    refuseTheNextGraphFetch();
    await fetchVersion1Again(queryClient);
    backend.release();
    await waitForTheRun();

    expect(
      screen.queryByText(
        "Your workflow couldn't be loaded. Reload the page and try again.",
      ),
    ).toBeNull();
    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 1 },
    ]);
  });

  test("the builder stays on it with the user's edits, and Run saves and runs them", async () => {
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    const { queryClient } = renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(
      () => expect(useGraphStore.getState().loadedGraph?.version).toBe(1),
      NETWORK_WAIT,
    );
    const edited = editTheCanvas();

    refuseTheNextGraphFetch();
    await fetchVersion1Again(queryClient);

    expect(
      screen.queryByText("Not authorized to view this workflow"),
      "the builder left the workflow",
    ).toBeNull();
    expect(canvas()).toEqual(edited);
    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(backend.updates).toHaveLength(1);
    expect(backend.updates[0].nodes).toContainEqual(
      expect.objectContaining({ input_default: { input: "edited" } }),
    );
    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 2 },
    ]);
  });
});

describe("builder Run whose workflow never loads", () => {
  test("closing the builder while Run waits cancels the run, and reopening the workflow doesn't start it", async () => {
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    hold("blocks");
    const builder = renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(
      () => expect(backend.requests.blocks).toBeGreaterThan(0),
      NETWORK_WAIT,
    );
    await clickRunOnceItIsEnabled(user);
    const runWaitsOn = useGraphStore.getState().loadSignal;

    builder.unmount();
    expect(runWaitsOn.aborted, "the closed builder's Run kept waiting").toBe(
      true,
    );
    backend.release();
    renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(() => expect(canvas().nodes).toHaveLength(2), NETWORK_WAIT);
    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 1 },
    ]);
    expect(screen.queryByText("Error saving workflow")).toBeNull();
  });

  test("moving to another workflow while Run waits cancels the run, and coming back doesn't start it", async () => {
    saveVersion(SAVED_WORKFLOW);
    saveVersion(BLANK_WORKFLOW, BLANK_GRAPH_ID);
    const user = userEvent.setup();
    hold("blocks");
    renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(
      () => expect(backend.requests.blocks).toBeGreaterThan(0),
      NETWORK_WAIT,
    );
    await clickRunOnceItIsEnabled(user);
    const runWaitsOn = useGraphStore.getState().loadSignal;

    await user.click(
      screen.getByRole("button", { name: `Open ${BLANK_GRAPH_ID}` }),
    );
    await waitFor(() =>
      expect(runWaitsOn.aborted, "the Run kept waiting").toBe(true),
    );
    backend.release();
    await waitFor(
      () =>
        expect(useGraphStore.getState().loadedGraph?.id).toBe(BLANK_GRAPH_ID),
      NETWORK_WAIT,
    );
    await user.click(screen.getByRole("button", { name: `Open ${GRAPH_ID}` }));
    await waitFor(() => expect(canvas().nodes).toHaveLength(2), NETWORK_WAIT);
    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 1 },
    ]);
    expect(screen.queryByText("Error saving workflow")).toBeNull();
  });

  test("Run on a workflow whose blocks didn't load says so at once", async () => {
    saveVersion(SAVED_WORKFLOW);
    server.use(
      http.get("/api/proxy/api/builder/blocks/batch", () =>
        HttpResponse.json(
          { detail: "Blocks are unavailable" },
          { status: 500 },
        ),
      ),
    );
    const user = userEvent.setup();
    renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);

    await clickRunOnceItIsEnabled(user);

    expect(
      await screen.findByText(
        "Your workflow couldn't be loaded. Reload the page and try again.",
        {},
        NETWORK_WAIT,
      ),
    ).toBeDefined();
    expect(backend.updates).toEqual([]);
    expect(backend.executions).toEqual([]);
  });
});

describe("builder Run while the workflow it names is still loading", () => {
  test("Run after moving to the workflow's latest version waits for it, and runs that version", async () => {
    saveVersion(SAVED_WORKFLOW);
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    hold("latest graph");
    renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(
      () => expect(useGraphStore.getState().loadedGraph?.version).toBe(1),
      NETWORK_WAIT,
    );

    await user.click(
      screen.getByRole("button", { name: `Open ${GRAPH_ID} latest` }),
    );
    await clickRunOnceItIsEnabled(user);
    backend.release();
    await waitForTheRun();

    expect(backend.updates, "running it saved a new version").toEqual([]);
    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 2 },
    ]);
  });

  test("Run after moving to a URL without a version runs at once when the canvas already shows the latest version", async () => {
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    const { queryClient } = renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    const latestKey = getGetV1GetSpecificGraphQueryKey(GRAPH_ID, {});
    await waitFor(() => {
      expect(useGraphStore.getState().loadedGraph?.version).toBe(1);
      expect(queryClient.getQueryData(latestKey)).toBeDefined();
    }, NETWORK_WAIT);
    // React Query hands back the graph object it already has when an equal
    // one arrives, so the builder can get the very same graph for the latest
    // version. (The API client's Date fields defeat that today, so the cache
    // is given the loaded response here, and refetching it is held.)
    hold("latest graph");
    queryClient.setQueryData(
      latestKey,
      queryClient.getQueryData(
        getGetV1GetSpecificGraphQueryKey(GRAPH_ID, { version: 1 }),
      ),
    );

    await user.click(
      screen.getByRole("button", { name: `Open ${GRAPH_ID} latest` }),
    );
    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(backend.updates, "running it saved a new version").toEqual([]);
    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 1 },
    ]);
  });

  test("Run on a workflow whose graph request fails says so at once", async () => {
    saveVersion(SAVED_WORKFLOW);
    server.use(
      http.get(`/api/proxy/api/graphs/${GRAPH_ID}`, async () => {
        await answerLoad("graph");
        return HttpResponse.json(
          { detail: `Workflow #${GRAPH_ID} not found` },
          { status: 404 },
        );
      }),
    );
    const user = userEvent.setup();
    hold("graph");
    renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);

    await clickRunOnceItIsEnabled(user);
    backend.release();

    expect(
      await screen.findByText(
        "Your workflow couldn't be loaded. Reload the page and try again.",
        {},
        NETWORK_WAIT,
      ),
    ).toBeDefined();
    expect(backend.updates).toEqual([]);
    expect(backend.executions).toEqual([]);
  });
});

describe("builder Run while the version it waits for changes", () => {
  test("moving to another version while Run waits cancels the run, and that version runs on its own Run", async () => {
    saveVersion(SAVED_WORKFLOW);
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    hold("blocks");
    renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(
      () => expect(backend.requests.blocks).toBeGreaterThan(0),
      NETWORK_WAIT,
    );
    await clickRunOnceItIsEnabled(user);
    const runWaitsOn = useGraphStore.getState().loadSignal;

    await user.click(
      screen.getByRole("button", { name: `Open ${GRAPH_ID} v2` }),
    );
    await waitFor(() =>
      expect(runWaitsOn.aborted, "the Run for version 1 kept waiting").toBe(
        true,
      ),
    );
    backend.release();
    await waitFor(
      () => expect(useGraphStore.getState().loadedGraph?.version).toBe(2),
      NETWORK_WAIT,
    );
    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 2 },
    ]);
  });

  test("Run while a workflow opens from a URL without a version waits for it, and runs its latest version", async () => {
    saveVersion(SAVED_WORKFLOW);
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    hold("latest graph");
    renderBuilder(`?flowID=${GRAPH_ID}`);

    await clickRunOnceItIsEnabled(user);
    backend.release();
    await waitForTheRun();

    expect(backend.updates, "running it saved a new version").toEqual([]);
    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 2 },
    ]);
  });

  test("picking a version while Run waits for a URL's latest version cancels the run, and the picked version runs on its own Run", async () => {
    saveVersion(SAVED_WORKFLOW);
    saveVersion(SAVED_WORKFLOW);
    const user = userEvent.setup();
    hold("latest graph");
    renderBuilder(`?flowID=${GRAPH_ID}`);
    await clickRunOnceItIsEnabled(user);
    const runWaitsOn = useGraphStore.getState().loadSignal;

    await user.click(screen.getByRole("button", { name: `Open ${GRAPH_ID}` }));
    await waitFor(() =>
      expect(
        runWaitsOn.aborted,
        "the Run for the latest version kept waiting",
      ).toBe(true),
    );
    backend.release();
    await waitFor(
      () => expect(useGraphStore.getState().loadedGraph?.version).toBe(1),
      NETWORK_WAIT,
    );
    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();

    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 1 },
    ]);
  });

  test("a blocks request that succeeds when tried again clears the load error, and Run then runs", async () => {
    saveVersion(SAVED_WORKFLOW);
    let blockFailures = 1;
    server.use(
      http.get("/api/proxy/api/builder/blocks/batch", () =>
        blockFailures-- > 0
          ? HttpResponse.json(
              { detail: "Blocks are unavailable" },
              { status: 500 },
            )
          : HttpResponse.json([STORE_VALUE_BLOCK, AGENT_OUTPUT_BLOCK]),
      ),
    );
    const user = userEvent.setup();
    const { queryClient } = renderBuilder(`?flowID=${GRAPH_ID}&flowVersion=1`);
    await waitFor(
      () => expect(useGraphStore.getState().loadFailed).toBe(true),
      NETWORK_WAIT,
    );

    await act(() => queryClient.refetchQueries({ type: "active" }));
    await waitFor(() => expect(canvas().nodes).toHaveLength(2), NETWORK_WAIT);
    expect(useGraphStore.getState().loadFailed).toBe(false);

    await clickRunOnceItIsEnabled(user);
    await waitForTheRun();
    expect(backend.executions).toEqual([
      { graphID: GRAPH_ID, graphVersion: 1 },
    ]);
  });
});
