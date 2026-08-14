import { describe, expect, it } from "vitest";
import { getWorkRunMetadata, toPreview } from "../helpers";

describe("WorkCard helpers", () => {
  it("parses a valid run metadata payload", () => {
    const meta = getWorkRunMetadata({
      kind: "expert_run",
      execution_id: "exec-1",
      graph_id: "graph-1",
      library_agent_id: "lib-1",
      graph_name: "Weekly Report",
      status: "completed",
      output_type: "table",
    });
    expect(meta).toEqual({
      executionId: "exec-1",
      graphId: "graph-1",
      libraryAgentId: "lib-1",
      graphName: "Weekly Report",
      status: "completed",
      outputType: "table",
    });
  });

  it("returns null for legacy messages without run metadata", () => {
    expect(getWorkRunMetadata(undefined)).toBeNull();
    expect(getWorkRunMetadata({})).toBeNull();
    expect(getWorkRunMetadata({ kind: "other" })).toBeNull();
  });

  it("requires execution and graph ids", () => {
    expect(
      getWorkRunMetadata({ kind: "expert_run", execution_id: "exec-1" }),
    ).toBeNull();
  });

  it("falls back to unknown for an unrecognized output type", () => {
    const meta = getWorkRunMetadata({
      kind: "expert_run",
      execution_id: "e",
      graph_id: "g",
      output_type: "weird",
    });
    expect(meta?.outputType).toBe("unknown");
    expect(meta?.graphName).toBe("Workflow run");
  });

  it("toPreview strips markdown links, quotes and bold", () => {
    const preview = toPreview(
      "I finished **Report**.\n\n> line one\n> line two\n\n[View the run](/x)",
    );
    expect(preview).toBe("I finished Report. line one line two View the run");
  });
});
