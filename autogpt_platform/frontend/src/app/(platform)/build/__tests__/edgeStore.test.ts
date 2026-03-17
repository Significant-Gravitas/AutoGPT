import { describe, it, expect, beforeEach, vi } from "vitest";
import { MarkerType } from "@xyflow/react";
import { useEdgeStore } from "../stores/edgeStore";
import { useNodeStore } from "../stores/nodeStore";
import { useHistoryStore } from "../stores/historyStore";
import type { CustomEdge } from "../components/FlowEditor/edges/CustomEdge";
import type { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import type { Link } from "@/app/api/__generated__/models/link";

function makeEdge(overrides: Partial<CustomEdge> & { id: string }): CustomEdge {
  return {
    type: "custom",
    source: "node-a",
    target: "node-b",
    sourceHandle: "output",
    targetHandle: "input",
    ...overrides,
  };
}

function makeExecutionResult(
  overrides: Partial<NodeExecutionResult>,
): NodeExecutionResult {
  return {
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    graph_exec_id: "gexec-1",
    node_exec_id: "nexec-1",
    node_id: "node-1",
    block_id: "block-1",
    status: "INCOMPLETE",
    input_data: {},
    output_data: {},
    add_time: new Date(),
    queue_time: null,
    start_time: null,
    end_time: null,
    ...overrides,
  };
}

beforeEach(() => {
  useEdgeStore.setState({ edges: [] });
  useNodeStore.setState({ nodes: [] });
  useHistoryStore.setState({ past: [], future: [] });
});

describe("edgeStore", () => {
  describe("setEdges", () => {
    it("replaces all edges", () => {
      const edges = [
        makeEdge({ id: "e1" }),
        makeEdge({ id: "e2", source: "node-c" }),
      ];

      useEdgeStore.getState().setEdges(edges);

      expect(useEdgeStore.getState().edges).toHaveLength(2);
      expect(useEdgeStore.getState().edges[0].id).toBe("e1");
      expect(useEdgeStore.getState().edges[1].id).toBe("e2");
    });
  });

  describe("addEdge", () => {
    it("adds an edge and auto-generates an ID", () => {
      const result = useEdgeStore.getState().addEdge({
        source: "n1",
        target: "n2",
        sourceHandle: "out",
        targetHandle: "in",
      });

      expect(result.id).toBe("n1:out->n2:in");
      expect(useEdgeStore.getState().edges).toHaveLength(1);
      expect(useEdgeStore.getState().edges[0].id).toBe("n1:out->n2:in");
    });

    it("uses provided ID when given", () => {
      const result = useEdgeStore.getState().addEdge({
        id: "custom-id",
        source: "n1",
        target: "n2",
        sourceHandle: "out",
        targetHandle: "in",
      });

      expect(result.id).toBe("custom-id");
    });

    it("sets type to custom and adds arrow marker", () => {
      const result = useEdgeStore.getState().addEdge({
        source: "n1",
        target: "n2",
        sourceHandle: "out",
        targetHandle: "in",
      });

      expect(result.type).toBe("custom");
      expect(result.markerEnd).toEqual({
        type: MarkerType.ArrowClosed,
        strokeWidth: 2,
        color: "#555",
      });
    });

    it("rejects duplicate edges without adding", () => {
      useEdgeStore.getState().addEdge({
        source: "n1",
        target: "n2",
        sourceHandle: "out",
        targetHandle: "in",
      });

      const pushSpy = vi.spyOn(useHistoryStore.getState(), "pushState");

      const duplicate = useEdgeStore.getState().addEdge({
        source: "n1",
        target: "n2",
        sourceHandle: "out",
        targetHandle: "in",
      });

      expect(useEdgeStore.getState().edges).toHaveLength(1);
      expect(duplicate.id).toBe("n1:out->n2:in");
      expect(pushSpy).not.toHaveBeenCalled();

      pushSpy.mockRestore();
    });

    it("pushes previous state to history store", () => {
      const pushSpy = vi.spyOn(useHistoryStore.getState(), "pushState");

      useEdgeStore.getState().addEdge({
        source: "n1",
        target: "n2",
        sourceHandle: "out",
        targetHandle: "in",
      });

      expect(pushSpy).toHaveBeenCalledWith({
        nodes: [],
        edges: [],
      });

      pushSpy.mockRestore();
    });
  });

  describe("removeEdge", () => {
    it("removes an edge by ID", () => {
      useEdgeStore.setState({
        edges: [makeEdge({ id: "e1" }), makeEdge({ id: "e2" })],
      });

      useEdgeStore.getState().removeEdge("e1");

      expect(useEdgeStore.getState().edges).toHaveLength(1);
      expect(useEdgeStore.getState().edges[0].id).toBe("e2");
    });

    it("does nothing when removing a non-existent edge", () => {
      useEdgeStore.setState({ edges: [makeEdge({ id: "e1" })] });

      useEdgeStore.getState().removeEdge("nonexistent");

      expect(useEdgeStore.getState().edges).toHaveLength(1);
    });

    it("pushes previous state to history store", () => {
      const existingEdges = [makeEdge({ id: "e1" })];
      useEdgeStore.setState({ edges: existingEdges });

      const pushSpy = vi.spyOn(useHistoryStore.getState(), "pushState");

      useEdgeStore.getState().removeEdge("e1");

      expect(pushSpy).toHaveBeenCalledWith({
        nodes: [],
        edges: existingEdges,
      });

      pushSpy.mockRestore();
    });
  });

  describe("upsertMany", () => {
    it("inserts new edges", () => {
      useEdgeStore.setState({ edges: [makeEdge({ id: "e1" })] });

      useEdgeStore.getState().upsertMany([makeEdge({ id: "e2" })]);

      expect(useEdgeStore.getState().edges).toHaveLength(2);
    });

    it("updates existing edges by ID", () => {
      useEdgeStore.setState({
        edges: [makeEdge({ id: "e1", source: "old-source" })],
      });

      useEdgeStore
        .getState()
        .upsertMany([makeEdge({ id: "e1", source: "new-source" })]);

      expect(useEdgeStore.getState().edges).toHaveLength(1);
      expect(useEdgeStore.getState().edges[0].source).toBe("new-source");
    });

    it("handles mixed inserts and updates", () => {
      useEdgeStore.setState({
        edges: [makeEdge({ id: "e1", source: "old" })],
      });

      useEdgeStore
        .getState()
        .upsertMany([
          makeEdge({ id: "e1", source: "updated" }),
          makeEdge({ id: "e2", source: "new" }),
        ]);

      const edges = useEdgeStore.getState().edges;
      expect(edges).toHaveLength(2);
      expect(edges.find((e) => e.id === "e1")?.source).toBe("updated");
      expect(edges.find((e) => e.id === "e2")?.source).toBe("new");
    });
  });

  describe("removeEdgesByHandlePrefix", () => {
    it("removes edges targeting a node with matching handle prefix", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({ id: "e1", target: "node-b", targetHandle: "input_foo" }),
          makeEdge({ id: "e2", target: "node-b", targetHandle: "input_bar" }),
          makeEdge({
            id: "e3",
            target: "node-b",
            targetHandle: "other_handle",
          }),
          makeEdge({ id: "e4", target: "node-c", targetHandle: "input_foo" }),
        ],
      });

      useEdgeStore.getState().removeEdgesByHandlePrefix("node-b", "input_");

      const edges = useEdgeStore.getState().edges;
      expect(edges).toHaveLength(2);
      expect(edges.map((e) => e.id).sort()).toEqual(["e3", "e4"]);
    });

    it("does not remove edges where target does not match nodeId", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            source: "node-b",
            target: "node-c",
            targetHandle: "input_x",
          }),
        ],
      });

      useEdgeStore.getState().removeEdgesByHandlePrefix("node-b", "input_");

      expect(useEdgeStore.getState().edges).toHaveLength(1);
    });
  });

  describe("getNodeEdges", () => {
    it("returns edges where node is source", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({ id: "e1", source: "node-a", target: "node-b" }),
          makeEdge({ id: "e2", source: "node-c", target: "node-d" }),
        ],
      });

      const result = useEdgeStore.getState().getNodeEdges("node-a");
      expect(result).toHaveLength(1);
      expect(result[0].id).toBe("e1");
    });

    it("returns edges where node is target", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({ id: "e1", source: "node-a", target: "node-b" }),
          makeEdge({ id: "e2", source: "node-c", target: "node-d" }),
        ],
      });

      const result = useEdgeStore.getState().getNodeEdges("node-b");
      expect(result).toHaveLength(1);
      expect(result[0].id).toBe("e1");
    });

    it("returns edges for both source and target", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({ id: "e1", source: "node-a", target: "node-b" }),
          makeEdge({ id: "e2", source: "node-b", target: "node-c" }),
          makeEdge({ id: "e3", source: "node-d", target: "node-e" }),
        ],
      });

      const result = useEdgeStore.getState().getNodeEdges("node-b");
      expect(result).toHaveLength(2);
      expect(result.map((e) => e.id).sort()).toEqual(["e1", "e2"]);
    });

    it("returns empty array for unconnected node", () => {
      useEdgeStore.setState({
        edges: [makeEdge({ id: "e1", source: "node-a", target: "node-b" })],
      });

      expect(useEdgeStore.getState().getNodeEdges("node-z")).toHaveLength(0);
    });
  });

  describe("isInputConnected", () => {
    it("returns true when target handle is connected", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: "input",
          }),
        ],
      });

      expect(useEdgeStore.getState().isInputConnected("node-b", "input")).toBe(
        true,
      );
    });

    it("returns false when target handle is not connected", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: "input",
          }),
        ],
      });

      expect(useEdgeStore.getState().isInputConnected("node-b", "other")).toBe(
        false,
      );
    });

    it("returns false when node is source not target", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            source: "node-b",
            target: "node-c",
            sourceHandle: "output",
            targetHandle: "input",
          }),
        ],
      });

      expect(useEdgeStore.getState().isInputConnected("node-b", "output")).toBe(
        false,
      );
    });
  });

  describe("isOutputConnected", () => {
    it("returns true when source handle is connected", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            source: "node-a",
            sourceHandle: "output",
          }),
        ],
      });

      expect(
        useEdgeStore.getState().isOutputConnected("node-a", "output"),
      ).toBe(true);
    });

    it("returns false when source handle is not connected", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            source: "node-a",
            sourceHandle: "output",
          }),
        ],
      });

      expect(useEdgeStore.getState().isOutputConnected("node-a", "other")).toBe(
        false,
      );
    });
  });

  describe("getBackendLinks", () => {
    it("converts edges to Link format", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            source: "n1",
            target: "n2",
            sourceHandle: "out",
            targetHandle: "in",
            data: { isStatic: true },
          }),
        ],
      });

      const links = useEdgeStore.getState().getBackendLinks();

      expect(links).toHaveLength(1);
      expect(links[0]).toEqual({
        id: "e1",
        source_id: "n1",
        sink_id: "n2",
        source_name: "out",
        sink_name: "in",
        is_static: true,
      });
    });
  });

  describe("addLinks", () => {
    it("converts Links to edges and adds them", () => {
      const links: Link[] = [
        {
          id: "link-1",
          source_id: "n1",
          sink_id: "n2",
          source_name: "out",
          sink_name: "in",
          is_static: false,
        },
      ];

      useEdgeStore.getState().addLinks(links);

      const edges = useEdgeStore.getState().edges;
      expect(edges).toHaveLength(1);
      expect(edges[0].source).toBe("n1");
      expect(edges[0].target).toBe("n2");
      expect(edges[0].sourceHandle).toBe("out");
      expect(edges[0].targetHandle).toBe("in");
      expect(edges[0].data?.isStatic).toBe(false);
    });

    it("adds multiple links", () => {
      const links: Link[] = [
        {
          id: "link-1",
          source_id: "n1",
          sink_id: "n2",
          source_name: "out",
          sink_name: "in",
        },
        {
          id: "link-2",
          source_id: "n3",
          sink_id: "n4",
          source_name: "result",
          sink_name: "value",
        },
      ];

      useEdgeStore.getState().addLinks(links);

      expect(useEdgeStore.getState().edges).toHaveLength(2);
    });
  });

  describe("getAllHandleIdsOfANode", () => {
    it("returns targetHandle values for edges targeting the node", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({ id: "e1", target: "node-b", targetHandle: "input_a" }),
          makeEdge({ id: "e2", target: "node-b", targetHandle: "input_b" }),
          makeEdge({ id: "e3", target: "node-c", targetHandle: "input_c" }),
        ],
      });

      const handles = useEdgeStore.getState().getAllHandleIdsOfANode("node-b");
      expect(handles).toEqual(["input_a", "input_b"]);
    });

    it("returns empty array when no edges target the node", () => {
      useEdgeStore.setState({
        edges: [makeEdge({ id: "e1", source: "node-b", target: "node-c" })],
      });

      expect(useEdgeStore.getState().getAllHandleIdsOfANode("node-b")).toEqual(
        [],
      );
    });

    it("returns empty string for edges with no targetHandle", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: undefined,
          }),
        ],
      });

      expect(useEdgeStore.getState().getAllHandleIdsOfANode("node-b")).toEqual([
        "",
      ]);
    });
  });

  describe("updateEdgeBeads", () => {
    it("updates bead counts for edges targeting the node", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: "input",
            data: { beadUp: 0, beadDown: 0, beadData: new Map() },
          }),
        ],
      });

      useEdgeStore.getState().updateEdgeBeads(
        "node-b",
        makeExecutionResult({
          node_exec_id: "exec-1",
          status: "COMPLETED",
          input_data: { input: "some-value" },
        }),
      );

      const edge = useEdgeStore.getState().edges[0];
      expect(edge.data?.beadUp).toBe(1);
      expect(edge.data?.beadDown).toBe(1);
    });

    it("counts INCOMPLETE status in beadUp but not beadDown", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: "input",
            data: { beadUp: 0, beadDown: 0, beadData: new Map() },
          }),
        ],
      });

      useEdgeStore.getState().updateEdgeBeads(
        "node-b",
        makeExecutionResult({
          node_exec_id: "exec-1",
          status: "INCOMPLETE",
          input_data: { input: "data" },
        }),
      );

      const edge = useEdgeStore.getState().edges[0];
      expect(edge.data?.beadUp).toBe(1);
      expect(edge.data?.beadDown).toBe(0);
    });

    it("does not modify edges not targeting the node", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-c",
            targetHandle: "input",
            data: { beadUp: 0, beadDown: 0, beadData: new Map() },
          }),
        ],
      });

      useEdgeStore.getState().updateEdgeBeads(
        "node-b",
        makeExecutionResult({
          node_exec_id: "exec-1",
          status: "COMPLETED",
          input_data: { input: "data" },
        }),
      );

      const edge = useEdgeStore.getState().edges[0];
      expect(edge.data?.beadUp).toBe(0);
      expect(edge.data?.beadDown).toBe(0);
    });

    it("does not update edge when input_data has no matching handle", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: "input",
            data: { beadUp: 0, beadDown: 0, beadData: new Map() },
          }),
        ],
      });

      useEdgeStore.getState().updateEdgeBeads(
        "node-b",
        makeExecutionResult({
          node_exec_id: "exec-1",
          status: "COMPLETED",
          input_data: { other_handle: "data" },
        }),
      );

      const edge = useEdgeStore.getState().edges[0];
      expect(edge.data?.beadUp).toBe(0);
      expect(edge.data?.beadDown).toBe(0);
    });

    it("accumulates beads across multiple executions", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: "input",
            data: { beadUp: 0, beadDown: 0, beadData: new Map() },
          }),
        ],
      });

      useEdgeStore.getState().updateEdgeBeads(
        "node-b",
        makeExecutionResult({
          node_exec_id: "exec-1",
          status: "COMPLETED",
          input_data: { input: "data1" },
        }),
      );

      useEdgeStore.getState().updateEdgeBeads(
        "node-b",
        makeExecutionResult({
          node_exec_id: "exec-2",
          status: "INCOMPLETE",
          input_data: { input: "data2" },
        }),
      );

      const edge = useEdgeStore.getState().edges[0];
      expect(edge.data?.beadUp).toBe(2);
      expect(edge.data?.beadDown).toBe(1);
    });

    it("handles static edges by setting beadUp to beadDown + 1", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            target: "node-b",
            targetHandle: "input",
            data: {
              isStatic: true,
              beadUp: 0,
              beadDown: 0,
              beadData: new Map(),
            },
          }),
        ],
      });

      useEdgeStore.getState().updateEdgeBeads(
        "node-b",
        makeExecutionResult({
          node_exec_id: "exec-1",
          status: "COMPLETED",
          input_data: { input: "data" },
        }),
      );

      const edge = useEdgeStore.getState().edges[0];
      expect(edge.data?.beadUp).toBe(2);
      expect(edge.data?.beadDown).toBe(1);
    });
  });

  describe("resetEdgeBeads", () => {
    it("resets all bead data on all edges", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            data: {
              beadUp: 5,
              beadDown: 3,
              beadData: new Map([["exec-1", "COMPLETED"]]),
            },
          }),
          makeEdge({
            id: "e2",
            data: {
              beadUp: 2,
              beadDown: 1,
              beadData: new Map([["exec-2", "INCOMPLETE"]]),
            },
          }),
        ],
      });

      useEdgeStore.getState().resetEdgeBeads();

      const edges = useEdgeStore.getState().edges;
      for (const edge of edges) {
        expect(edge.data?.beadUp).toBe(0);
        expect(edge.data?.beadDown).toBe(0);
        expect(edge.data?.beadData?.size).toBe(0);
      }
    });

    it("preserves other edge data when resetting beads", () => {
      useEdgeStore.setState({
        edges: [
          makeEdge({
            id: "e1",
            data: {
              isStatic: true,
              edgeColorClass: "text-red-500",
              beadUp: 3,
              beadDown: 2,
              beadData: new Map(),
            },
          }),
        ],
      });

      useEdgeStore.getState().resetEdgeBeads();

      const edge = useEdgeStore.getState().edges[0];
      expect(edge.data?.isStatic).toBe(true);
      expect(edge.data?.edgeColorClass).toBe("text-red-500");
      expect(edge.data?.beadUp).toBe(0);
    });
  });
});
