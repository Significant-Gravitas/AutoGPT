import { describe, expect, it } from "vitest";
import { sanitizeImportedGraph, validateGraphStructure } from "../utils";

describe("validateGraphStructure", () => {
  it("returns no errors for a valid graph with nodes and links", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        {
          id: "node-1",
          block_id: "block-1",
          input_default: {},
        },
        {
          id: "node-2",
          block_id: "block-2",
          input_default: {},
        },
      ],
      links: [
        {
          source_id: "node-1",
          sink_id: "node-2",
          source_name: "output",
          sink_name: "input",
        },
      ],
    } as any;

    const errors = validateGraphStructure(graph);
    expect(errors).toEqual([]);
  });

  it("returns error when graph has no nodes", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [],
    } as any;

    const errors = validateGraphStructure(graph);
    expect(errors).toContain("Graph has no nodes");
  });

  it("returns error when nodes array is missing", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
    } as any;

    const errors = validateGraphStructure(graph);
    expect(errors).toContain("Graph has no nodes");
  });

  it("detects missing block_id on a node", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        {
          id: "node-1",
          block_id: "valid-block",
          input_default: {},
        },
        {
          id: "node-2",
          block_id: "",
          input_default: {},
        },
      ],
      links: [],
    } as any;

    const errors = validateGraphStructure(graph);
    expect(errors.some((e) => e.includes("missing block_id"))).toBe(true);
  });

  it("detects missing node id", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        {
          id: "",
          block_id: "block-1",
          input_default: {},
        },
      ],
      links: [],
    } as any;

    const errors = validateGraphStructure(graph);
    expect(errors.some((e) => e.includes("missing node id"))).toBe(true);
  });

  it("detects missing link fields", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        { id: "node-1", block_id: "block-1", input_default: {} },
        { id: "node-2", block_id: "block-2", input_default: {} },
      ],
      links: [
        {
          source_id: "",
          sink_id: "node-2",
          source_name: "output",
          sink_name: "input",
        },
        {
          source_id: "node-1",
          sink_id: "",
          source_name: "output",
          sink_name: "input",
        },
        {
          source_id: "node-1",
          sink_id: "node-2",
          source_name: "",
          sink_name: "",
        },
      ],
    } as any;

    const errors = validateGraphStructure(graph);
    expect(errors.some((e) => e.includes("missing source_id"))).toBe(true);
    expect(errors.some((e) => e.includes("missing sink_id"))).toBe(true);
    expect(errors.some((e) => e.includes("missing source_name"))).toBe(true);
    expect(errors.some((e) => e.includes("missing sink_name"))).toBe(true);
  });

  it("returns all errors together rather than failing on first", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        { id: "n1", block_id: "", input_default: {} },
        { id: "", block_id: "b1", input_default: {} },
      ],
      links: [
        {
          source_id: "",
          sink_id: "",
          source_name: "",
          sink_name: "",
        },
      ],
    } as any;

    const errors = validateGraphStructure(graph);
    expect(errors.length).toBeGreaterThanOrEqual(5);
  });
});

describe("sanitizeImportedGraph", () => {
  it("is a no-op — does not modify credentials in the graph", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        {
          id: "node-1",
          block_id: "block-1",
          input_default: {
            credentials: { api_key: "secret-123" },
            value: "hello",
          },
        },
      ],
      links: [],
    } as any;

    const original = JSON.stringify(graph);
    sanitizeImportedGraph(graph);
    expect(JSON.stringify(graph)).toEqual(original);
  });

  it("is a no-op — does not modify block IDs in the graph", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        {
          id: "node-1",
          block_id: "old-block-id",
          input_default: {},
        },
      ],
      links: [],
    } as any;

    const original = JSON.stringify(graph);
    sanitizeImportedGraph(graph);
    expect(JSON.stringify(graph)).toEqual(original);
  });

  it("is a no-op — preserves the graph structure exactly", () => {
    const graph = {
      id: "test-graph",
      name: "Test",
      description: "Test",
      nodes: [
        { id: "n1", block_id: "b1", input_default: {} },
        { id: "n2", block_id: "b2", input_default: {} },
      ],
      links: [
        {
          source_id: "n1",
          sink_id: "n2",
          source_name: "output",
          sink_name: "input",
        },
      ],
    } as any;

    const snapshot = JSON.stringify(graph);
    sanitizeImportedGraph(graph);
    expect(JSON.stringify(graph)).toEqual(snapshot);
  });
});
