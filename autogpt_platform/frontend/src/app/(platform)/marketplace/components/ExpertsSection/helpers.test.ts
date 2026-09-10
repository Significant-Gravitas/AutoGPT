import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { describe, expect, test } from "vitest";
import {
  getDayOneWorkflow,
  getExpertAccent,
  getExpertFirstName,
} from "./helpers";

describe("getExpertAccent", () => {
  test("themes known roles and falls back to zinc", () => {
    expect(getExpertAccent("Marketing").pill).toContain("violet");
    expect(getExpertAccent("Sales").pill).toContain("amber");
    expect(getExpertAccent("Ops").pill).toContain("sky");
    expect(getExpertAccent("Astrologer").pill).toContain("zinc");
  });
});

function makeWorkflow(
  overrides: Partial<ExpertWorkflowRef>,
): ExpertWorkflowRef {
  return {
    id: "wf-x",
    store_listing_version_id: null,
    library_agent_id: null,
    graph_id: null,
    name: null,
    description: null,
    ...overrides,
  };
}

describe("getDayOneWorkflow", () => {
  test("returns the first workflow with a displayable name", () => {
    const dangling = makeWorkflow({ id: "wf-1", name: null });
    const blank = makeWorkflow({ id: "wf-2", name: "   " });
    const named = makeWorkflow({ id: "wf-3", name: "Content Calendar" });
    expect(getDayOneWorkflow([dangling, blank, named])).toBe(named);
  });

  test("returns null when no workflow has a name", () => {
    expect(getDayOneWorkflow([])).toBe(null);
    expect(getDayOneWorkflow([makeWorkflow({ name: null })])).toBe(null);
    expect(getDayOneWorkflow([makeWorkflow({ name: " " })])).toBe(null);
  });
});

describe("getExpertFirstName", () => {
  test("returns the first token of the name", () => {
    expect(getExpertFirstName("Maria Lopez")).toBe("Maria");
    expect(getExpertFirstName("Max")).toBe("Max");
    expect(getExpertFirstName("  Frankie  ")).toBe("Frankie");
  });

  test("falls back for an empty name", () => {
    expect(getExpertFirstName("")).toBe("Expert");
    expect(getExpertFirstName("   ")).toBe("Expert");
  });
});
