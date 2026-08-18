import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { describe, expect, test } from "vitest";
import {
  getDayOneWorkflow,
  getExpertCardHiredState,
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
  test("accepts secure remote and site-relative avatar URLs", () => {
    expect(
      getExpertAvatarUrl({
        avatar_url: "/uploads/custom.png",
      }),
    ).toBe("/uploads/custom.png");
    expect(
      getExpertAvatarUrl({
        avatar_url: "https://cdn.example.com/avatar.png",
      }),
    ).toBe("https://cdn.example.com/avatar.png");
  });

  test("rejects unsafe avatar URLs", () => {
    expect(getExpertAvatarUrl({ avatar_url: "javascript:alert('xss')" })).toBe(
      null,
    );
    expect(
      getExpertAvatarUrl({ avatar_url: "data:image/svg+xml,<svg />" }),
    ).toBe(null);
    expect(
      getExpertAvatarUrl({ avatar_url: "http://cdn.example.com/avatar.png" }),
    ).toBe(null);
    expect(
      getExpertAvatarUrl({ avatar_url: "//cdn.example.com/avatar.png" }),
    ).toBe(null);
    expect(getExpertAvatarUrl({ avatar_url: "avatar.png" })).toBe(null);
  });

  test("uses initials when the API has no avatar URL", () => {
    expect(getExpertAvatarUrl({ avatar_url: null })).toBe(null);
    expect(getExpertAvatarUrl({ avatar_url: "   " })).toBe(null);
  });
});

describe("getHiredExpertsLookup", () => {
  test("indexes hired experts by source template", () => {
    const hired = {
      id: "hired-1",
      source_template_id: "template-1",
      is_archived: false,
    };
    const lookup = getHiredExpertsLookup([hired], {
      enabled: true,
      isError: false,
      isFetching: false,
    });

    expect(lookup.state).toBe("loaded");
    expect(lookup.byTemplateId.get("template-1")).toBe(hired);
  });

  test("keeps cached lookup data usable during a refetch error", () => {
    const hired = {
      id: "hired-1",
      source_template_id: "template-1",
      is_archived: false,
    };
    const lookup = getHiredExpertsLookup([hired], {
      enabled: true,
      isError: true,
      isFetching: false,
    });

    expect(lookup.state).toBe("loaded");
    expect(lookup.byTemplateId.get("template-1")).toBe(hired);
  });

  test("distinguishes unresolved and terminal lookup failures", () => {
    expect(
      getHiredExpertsLookup(undefined, {
        enabled: true,
        isError: false,
        isFetching: true,
      }).state,
    ).toBe("loading");
    expect(
      getHiredExpertsLookup(undefined, {
        enabled: true,
        isError: true,
        isFetching: false,
      }).state,
    ).toBe("error");
  });

  test("settles when the lookup is disabled", () => {
    expect(
      getHiredExpertsLookup(undefined, {
        enabled: false,
        isError: false,
        isFetching: false,
      }).state,
    ).toBe("loaded");
  });
});

describe("getExpertCardHiredState", () => {
  const hiredTemplateIds = new Set(["template-1"]);

  test("maps lookup state and membership to a card state", () => {
    expect(
      getExpertCardHiredState("template-1", hiredTemplateIds, "loading"),
    ).toBe("unknown");
    expect(
      getExpertCardHiredState("template-1", hiredTemplateIds, "error"),
    ).toBe("error");
    expect(
      getExpertCardHiredState("template-1", hiredTemplateIds, "loaded"),
    ).toBe("hired");
    expect(
      getExpertCardHiredState("template-2", hiredTemplateIds, "loaded"),
    ).toBe("available");
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
