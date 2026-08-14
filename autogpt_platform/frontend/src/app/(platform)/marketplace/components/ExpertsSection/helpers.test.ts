import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { describe, expect, test } from "vitest";
import {
  getDayOneWorkflow,
  getExpertAccent,
  getExpertAvatarUrl,
  getExpertFirstName,
  getHiredExpertsLookup,
} from "./helpers";

describe("getExpertAccent", () => {
  test("themes known roles and falls back to zinc", () => {
    expect(getExpertAccent("Marketing").pill).toContain("violet");
    expect(getExpertAccent("Sales").pill).toContain("amber");
    expect(getExpertAccent("Ops").pill).toContain("sky");
    expect(getExpertAccent("Astrologer").pill).toContain("zinc");
  });
});

describe("getExpertAvatarUrl", () => {
  test("prefers the expert's own avatar_url", () => {
    expect(
      getExpertAvatarUrl({
        avatar_url: "/uploads/custom.png",
        name: "Anybody",
      }),
    ).toBe("/uploads/custom.png");
  });

  test("falls back to known seed persona avatars when avatar_url is null", () => {
    expect(getExpertAvatarUrl({ avatar_url: null, name: "Maria" })).toBe(
      "/experts/maria.svg",
    );
    expect(getExpertAvatarUrl({ avatar_url: null, name: "Max" })).toBe(
      "/experts/max.svg",
    );
    expect(getExpertAvatarUrl({ avatar_url: null, name: "Frankie" })).toBe(
      "/experts/frankie.svg",
    );
  });

  test("does not assign a persona face to an unrelated expert", () => {
    expect(getExpertAvatarUrl({ avatar_url: null, name: "Other Maria" })).toBe(
      null,
    );
  });

  test("treats a whitespace-only avatar_url as absent", () => {
    expect(getExpertAvatarUrl({ avatar_url: "   ", name: "Maria" })).toBe(
      "/experts/maria.svg",
    );
    expect(getExpertAvatarUrl({ avatar_url: "   ", name: "Other" })).toBe(null);
  });
});

describe("getHiredExpertsLookup", () => {
  test("indexes hired experts by source template", () => {
    const hired = { id: "hired-1", source_template_id: "template-1" };
    const lookup = getHiredExpertsLookup([hired], {
      isError: false,
      isFetching: false,
    });

    expect(lookup.state).toBe("loaded");
    expect(lookup.byTemplateId.get("template-1")).toBe(hired);
  });

  test("keeps cached lookup data usable during a refetch error", () => {
    const hired = { id: "hired-1", source_template_id: "template-1" };
    const lookup = getHiredExpertsLookup([hired], {
      isError: true,
      isFetching: false,
    });

    expect(lookup.state).toBe("loaded");
    expect(lookup.byTemplateId.get("template-1")).toBe(hired);
  });

  test("distinguishes unresolved and terminal lookup failures", () => {
    expect(
      getHiredExpertsLookup(undefined, {
        isError: false,
        isFetching: true,
      }).state,
    ).toBe("loading");
    expect(
      getHiredExpertsLookup(undefined, {
        isError: true,
        isFetching: false,
      }).state,
    ).toBe("error");
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
});
