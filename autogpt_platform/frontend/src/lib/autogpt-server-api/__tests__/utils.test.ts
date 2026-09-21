import { describe, expect, it } from "vitest";
import { GraphInput } from "@/app/api/__generated__/models/graphInput";
import {
  formatEdgeID,
  removeAgentInputBlockValues,
  sanitizeImportedGraph,
} from "../utils";
import { Block, BlockUIType } from "../types";

function graph(nodes: GraphInput["nodes"]): GraphInput {
  return { name: "Imported", description: "", nodes, links: [] } as GraphInput;
}

function block(id: string, uiType: BlockUIType): Block {
  return { id, uiType } as Block;
}

describe("sanitizeImportedGraph", () => {
  it("rewrites block ids that were renumbered after #8229", () => {
    const imported = graph([
      { id: "n1", block_id: "a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6" },
      { id: "n2", block_id: "not-a-renamed-block" },
    ] as GraphInput["nodes"]);

    sanitizeImportedGraph(imported);

    expect(imported.nodes?.[0].block_id).toBe(
      "436c3984-57fd-4b85-8e9a-459b356883bd",
    );
    expect(imported.nodes?.[1].block_id).toBe("not-a-renamed-block");
  });

  it("strips credentials at every depth, so an exported graph cannot carry someone else's", () => {
    const imported = graph([
      {
        id: "n1",
        block_id: "b1",
        input_default: {
          credentials: { id: "secret", provider: "github" },
          nested: [{ credentials: { id: "also-secret" }, keep: "kept" }],
        },
      },
    ] as unknown as GraphInput["nodes"]);

    sanitizeImportedGraph(imported);

    const input = imported.nodes?.[0].input_default as Record<string, unknown>;
    expect(input.credentials).toBeUndefined();
    const nested = input.nested as Record<string, unknown>[];
    expect(nested[0].credentials).toBeUndefined();
    expect(nested[0].keep).toBe("kept");
  });

  it("tolerates a graph with no nodes", () => {
    const imported = graph(undefined);
    expect(() => sanitizeImportedGraph(imported)).not.toThrow();
  });
});

describe("removeAgentInputBlockValues", () => {
  it("blanks the value of input blocks and leaves every other node alone", () => {
    const blocks = [
      block("input-block", BlockUIType.INPUT),
      block("other-block", BlockUIType.STANDARD),
    ];
    const source = graph([
      { id: "n1", block_id: "input-block", input_default: { value: "secret" } },
      { id: "n2", block_id: "other-block", input_default: { value: "kept" } },
    ] as unknown as GraphInput["nodes"]);

    const result = removeAgentInputBlockValues(source, blocks);

    expect(result.nodes?.[0].input_default).toMatchObject({ value: "" });
    expect(result.nodes?.[1].input_default).toMatchObject({ value: "kept" });
    // The original is left untouched — callers keep using it after exporting.
    expect(source.nodes?.[0].input_default).toMatchObject({ value: "secret" });
  });
});

describe("formatEdgeID", () => {
  it("formats a backend link", () => {
    expect(
      formatEdgeID({
        source_id: "a",
        source_name: "out",
        sink_id: "b",
        sink_name: "in",
      } as never),
    ).toBe("a_out_b_in");
  });

  it("formats a react-flow connection", () => {
    expect(
      formatEdgeID({
        source: "a",
        sourceHandle: "out",
        target: "b",
        targetHandle: "in",
      }),
    ).toBe("a_out_b_in");
  });
});
