import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import type { RJSFSchema } from "@rjsf/utils";
import {
  ReactFlow,
  useStoreApi,
  type NodeProps,
  type Node,
} from "@xyflow/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OutputHandler } from "../OutputHandler";
import { BlockUIType } from "../../../types";
import { useEdgeStore } from "../../../../stores/edgeStore";
import { useNodeStore } from "../../../../stores/nodeStore";
import { linkToCustomEdge } from "../../../helper";
import { useEffect } from "react";

type OutputNode = Node<
  { schema: RJSFSchema; uiType?: BlockUIType },
  "output-test"
>;
let flowStore: ReturnType<typeof useStoreApi>;

function CaptureStore() {
  const store = useStoreApi();
  useEffect(() => {
    flowStore = store;
  }, [store]);
  return null;
}

function TestOutputNode({ data, id }: NodeProps<OutputNode>) {
  return (
    <OutputHandler
      outputSchema={data.schema}
      nodeId={id}
      uiType={data.uiType ?? BlockUIType.STANDARD}
    />
  );
}

const nodeTypes = { "output-test": TestOutputNode };
const schema: RJSFSchema = {
  properties: {
    response: { type: "object", additionalProperties: true },
    unused: { type: "string" },
  },
};

function Flow({
  outputSchema = schema,
  uiType,
}: {
  outputSchema?: RJSFSchema;
  uiType?: BlockUIType;
}) {
  const edges = useEdgeStore((state) => state.edges);
  return (
    <ReactFlow
      nodes={[
        {
          id: "node-a",
          type: "output-test",
          position: { x: 0, y: 0 },
          data: { schema: outputSchema, uiType },
        },
        { id: "node-b", position: { x: 500, y: 0 }, data: { label: "Sink" } },
      ]}
      edges={edges.map((edge) => ({ ...edge, type: "default" }))}
      nodeTypes={nodeTypes}
    >
      <CaptureStore />
    </ReactFlow>
  );
}

function sourceHandles() {
  return (
    flowStore
      .getState()
      .nodeLookup.get("node-a")
      ?.internals.handleBounds?.source?.map((handle) => handle.id) ?? []
  );
}

function connect(sourceHandle: string) {
  useEdgeStore.getState().setEdges([
    linkToCustomEdge({
      id: "dynamic",
      source_id: "node-a",
      sink_id: "node-b",
      source_name: sourceHandle,
      sink_name: "",
    }),
  ]);
}

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class implements ResizeObserver {
      constructor(private callback: ResizeObserverCallback) {}
      observe(target: Element) {
        queueMicrotask(() =>
          this.callback(
            [
              {
                target,
                contentRect: target.getBoundingClientRect(),
                borderBoxSize: [],
                contentBoxSize: [],
                devicePixelContentBoxSize: [],
              },
            ],
            this,
          ),
        );
      }
      unobserve() {}
      disconnect() {}
    },
  );
  vi.spyOn(HTMLElement.prototype, "offsetWidth", "get").mockReturnValue(300);
  vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(200);
  useEdgeStore.setState({ edges: [] });
  useNodeStore.setState({
    nodes: [],
    nodesInResolutionMode: new Set(),
    nodeResolutionData: new Map(),
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("OutputHandler with React Flow", () => {
  it("keeps disconnected broken outputs mounted while the output section is collapsed", async () => {
    const outputSchema: RJSFSchema = {
      properties: {
        response: {
          type: "object",
          properties: { removed: { type: "string" } },
        },
      },
    };
    useNodeStore.setState({
      nodesInResolutionMode: new Set(["node-a"]),
      nodeResolutionData: new Map([
        [
          "node-a",
          {
            incompatibilities: {
              missingInputs: [],
              missingOutputs: ["response_#_removed"],
              newInputs: [],
              newOutputs: [],
              newRequiredInputs: [],
              inputTypeMismatches: [],
            },
            currentSchema: { input_schema: {}, output_schema: outputSchema },
            pendingUpdate: { input_schema: {}, output_schema: {} },
            pendingHardcodedValues: {},
          },
        ],
      ]),
    });
    render(<Flow outputSchema={outputSchema} />);
    fireEvent.click(await screen.findByRole("button", { name: "Output" }));
    await waitFor(() =>
      expect(sourceHandles()).toContain("response_#_removed"),
    );
    expect(screen.getByText("removed")).toBeDefined();
    expect(
      useEdgeStore.getState().isOutputConnected("node-a", "response_#_removed"),
    ).toBe(false);
  });

  it("renders boolean schemas and malformed schema values without crashing", async () => {
    render(
      <Flow
        outputSchema={
          {
            properties: { allowed: true, denied: false, malformed: null },
          } as unknown as RJSFSchema
        }
      />,
    );
    await waitFor(() =>
      expect(sourceHandles()).toEqual(["allowed", "denied", "malformed"]),
    );
    expect(screen.getAllByText("(any)")).toHaveLength(3);
  });

  it("registers a backend link's dynamic handles and refreshes them when the edge changes at the same node size", async () => {
    const { container } = render(<Flow />);
    await waitFor(() => expect(sourceHandles()).toContain("response"));
    act(() => connect("response_#_reviews_#_author"));
    await waitFor(() =>
      expect(sourceHandles()).toContain("response_#_reviews_#_author"),
    );
    expect(
      useEdgeStore.getState().isOutputConnected("node-a", "response"),
    ).toBe(true);
    expect(
      useEdgeStore.getState().isOutputConnected("node-a", "response_#_reviews"),
    ).toBe(true);
    await waitFor(() =>
      expect(
        container.querySelector(
          '[data-testid="rf__edge-dynamic"] path.react-flow__edge-path',
        ),
      ).not.toBeNull(),
    );

    fireEvent.click(screen.getByRole("button", { name: "Output" }));
    expect(screen.queryByText("unused")).toBeNull();
    expect(screen.getByText("author")).toBeDefined();
    act(() => connect("response_#_reviews_#_editor"));
    await waitFor(() => {
      expect(sourceHandles()).toContain("response_#_reviews_#_editor");
      expect(sourceHandles()).not.toContain("response_#_reviews_#_author");
    });
    act(() => useEdgeStore.getState().setEdges([]));
    await waitFor(() => expect(sourceHandles()).toEqual([]));
    expect(
      useEdgeStore.getState().isOutputConnected("node-a", "response"),
    ).toBe(false);
  });

  it("refreshes manually expanded handles and output-schema changes", async () => {
    const outputSchema: RJSFSchema = {
      properties: {
        response: {
          type: "object",
          properties: { declared: { type: "string" } },
        },
      },
    };
    const { rerender } = render(<Flow outputSchema={outputSchema} />);
    await waitFor(() => expect(sourceHandles()).toEqual(["response"]));
    fireEvent.click(screen.getByRole("button", { name: "Expand" }));
    await waitFor(() =>
      expect(sourceHandles()).toContain("response_#_declared"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    await waitFor(() => expect(sourceHandles()).toEqual(["response"]));
    rerender(
      <Flow
        outputSchema={{ properties: { replacement: { type: "string" } } }}
      />,
    );
    await waitFor(() => expect(sourceHandles()).toEqual(["replacement"]));
  });

  it("renders hostile dictionary keys as handles without prototype mutation", async () => {
    act(() => connect("response_#___proto___#_injected"));
    try {
      render(<Flow />);
      await waitFor(() =>
        expect(sourceHandles()).toContain("response_#___proto___#_injected"),
      );
      expect(Object.hasOwn(Object.prototype, "properties")).toBe(false);
    } finally {
      Reflect.deleteProperty(Object.prototype, "properties");
    }
  });

  it("keeps terminal output blocks free of source handles", async () => {
    act(() => connect("response_#_key"));
    render(<Flow uiType={BlockUIType.OUTPUT} />);
    expect(await screen.findByText("key")).toBeDefined();
    expect(
      document.querySelector('.react-flow__node[data-id="node-a"] .source'),
    ).toBeNull();
  });
});
