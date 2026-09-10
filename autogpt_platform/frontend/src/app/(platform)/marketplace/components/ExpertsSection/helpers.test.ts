import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { describe, expect, test } from "vitest";
import {
  getDayOneWorkflow,
  getExpertAccent,
  getExpertAccessProviders,
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

describe("getExpertAccessProviders", () => {
  // Every provider the platform pays for, as /api/integrations/providers/system
  // returns them. The roster's real workflows are almost entirely these.
  const systemProviders = ["anthropic", "openai", "jina", "webshare_proxy"];

  test("keeps only what the viewer has to connect themselves", () => {
    // Maria's real providers on dev: three the platform supplies, one not.
    const blogWriter = makeWorkflow({
      id: "wf-1",
      integration_providers: ["openai", "dataforseo", "anthropic"],
    });
    const copyImprover = makeWorkflow({
      id: "wf-2",
      integration_providers: ["openai", "jina"],
    });

    expect(
      getExpertAccessProviders([blogWriter, copyImprover], systemProviders),
    ).toEqual(["dataforseo"]);
  });

  test("reads the untruncated list, not the three-item display chain", () => {
    // The chain caps at three, so a fourth integration is missing from it.
    // Reading the chain here would drop airtable and tell the user to connect
    // three services when they have to connect four.
    const workflow = makeWorkflow({
      id: "wf-1",
      chain: [
        { kind: "integration", provider: "google" },
        { kind: "integration", provider: "dataforseo" },
        { kind: "integration", provider: "openai" },
      ],
      integration_providers: ["airtable", "dataforseo", "google", "openai"],
    });

    expect(getExpertAccessProviders([workflow], systemProviders)).toEqual([
      "airtable",
      "dataforseo",
      "google",
    ]);
  });

  test("dedupes a provider named by more than one workflow", () => {
    const first = makeWorkflow({
      id: "wf-1",
      integration_providers: ["linkedin"],
    });
    const second = makeWorkflow({
      id: "wf-2",
      integration_providers: ["linkedin", "google"],
    });

    expect(getExpertAccessProviders([first, second], systemProviders)).toEqual([
      "linkedin",
      "google",
    ]);
  });

  test("is empty when there is nothing to connect", () => {
    expect(getExpertAccessProviders([], systemProviders)).toEqual([]);
    // A template served before the listing-graph fallback landed carries no
    // providers at all, and the section has to disappear rather than throw.
    expect(
      getExpertAccessProviders([makeWorkflow({ id: "wf-1" })], systemProviders),
    ).toEqual([]);
    expect(
      getExpertAccessProviders(
        [makeWorkflow({ id: "wf-2", integration_providers: ["anthropic"] })],
        systemProviders,
      ),
    ).toEqual([]);
  });

  test("names nothing while the system-provider list is missing", () => {
    // Better to say nothing than to tell someone they must connect Anthropic.
    const workflow = makeWorkflow({
      id: "wf-1",
      integration_providers: ["anthropic", "dataforseo"],
    });

    expect(getExpertAccessProviders([workflow], undefined)).toEqual([]);
  });
});
