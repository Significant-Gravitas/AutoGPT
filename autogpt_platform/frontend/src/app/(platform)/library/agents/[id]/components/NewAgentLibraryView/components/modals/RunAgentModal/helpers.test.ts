import { ORG_HEADER_NAME, TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { describe, expect, it } from "vitest";
import { getLibraryAgentScopeRequest } from "./helpers";

describe("getLibraryAgentScopeRequest", () => {
  it("pins a team-scoped library agent to its own team", () => {
    expect(getLibraryAgentScopeRequest("org-a", "team-a")).toEqual({
      headers: {
        [ORG_HEADER_NAME]: "org-a",
        [TEAM_HEADER_NAME]: "team-a",
      },
    });
  });

  it("forces an org-home agent to stay in org-home", () => {
    expect(getLibraryAgentScopeRequest("org-a", null)).toEqual({
      headers: { [ORG_HEADER_NAME]: "org-a", [TEAM_HEADER_NAME]: "" },
    });
  });
});
