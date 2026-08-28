import { afterEach, describe, expect, it } from "vitest";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";
import {
  getLastUsedTeam,
  getTeamRequestInit,
  getTeamScopedQueryKey,
  getTenantRequestInit,
  setLastUsedTeam,
} from "./helpers";

afterEach(() => {
  window.localStorage.clear();
});

describe("TeamPicker last-used storage", () => {
  it("returns null for a surface that was never used", () => {
    expect(getLastUsedTeam("org-1", "builder-save")).toBeNull();
  });

  it("remembers the last team picked on a surface", () => {
    setLastUsedTeam("org-1", "builder-save", "team-1");
    expect(getLastUsedTeam("org-1", "builder-save")).toBe("team-1");
  });

  it("keeps last-used separate per surface", () => {
    setLastUsedTeam("org-1", "builder-save", "team-1");
    setLastUsedTeam("org-1", "library-folder", "team-2");
    expect(getLastUsedTeam("org-1", "builder-save")).toBe("team-1");
    expect(getLastUsedTeam("org-1", "library-folder")).toBe("team-2");
  });

  it("treats an org-home selection as null (no team)", () => {
    setLastUsedTeam("org-1", "builder-save", "team-1");
    setLastUsedTeam("org-1", "builder-save", null);
    expect(getLastUsedTeam("org-1", "builder-save")).toBeNull();
  });

  it("keeps last-used targets separate per organization", () => {
    setLastUsedTeam("org-1", "builder-save", "team-1");
    setLastUsedTeam("org-2", "builder-save", "team-2");
    expect(getLastUsedTeam("org-1", "builder-save")).toBe("team-1");
    expect(getLastUsedTeam("org-2", "builder-save")).toBe("team-2");
  });

  it("survives a corrupt storage value", () => {
    window.localStorage.setItem("create-surface-teams", "not-json");
    expect(getLastUsedTeam("org-1", "builder-save")).toBeNull();
  });
});

describe("getTeamRequestInit", () => {
  it("sends the org-home sentinel (empty X-Team-Id) for org-home (null team)", () => {
    expect(getTeamRequestInit(null)).toEqual({
      headers: { [TEAM_HEADER_NAME]: "" },
    });
  });

  it("stamps the X-Team-Id header for a team", () => {
    expect(getTeamRequestInit("team-9")).toEqual({
      headers: { [TEAM_HEADER_NAME]: "team-9" },
    });
  });

  it("fails closed while the organization context is loading", () => {
    expect(getTeamRequestInit(null, false)).toEqual({
      headers: { [TEAM_HEADER_NAME]: "__org_context_loading__" },
    });
  });
});

describe("getTenantRequestInit", () => {
  it("pins both persisted organization and team", () => {
    expect(getTenantRequestInit("org-1", "team-9")).toEqual({
      headers: { "X-Org-Id": "org-1", "X-Team-Id": "team-9" },
    });
  });

  it("uses empty sentinels for personal and org-home scope", () => {
    expect(getTenantRequestInit(null, null)).toEqual({
      headers: { "X-Org-Id": "", "X-Team-Id": "" },
    });
  });
});

describe("getTeamScopedQueryKey", () => {
  it("isolates the same request across teams and org-home", () => {
    const generatedKey = ["/api/graphs/graph-1/executions"];

    expect(getTeamScopedQueryKey(generatedKey, "org-1", "team-a")).not.toEqual(
      getTeamScopedQueryKey(generatedKey, "org-1", "team-b"),
    );
    expect(getTeamScopedQueryKey(generatedKey, "org-1", null)).not.toEqual(
      getTeamScopedQueryKey(generatedKey, "org-1", "team-a"),
    );
    expect(getTeamScopedQueryKey(generatedKey, "org-1", null)).toEqual([
      "/api/graphs/graph-1/executions",
      { organizationId: "org-1", teamId: "__org_home__" },
    ]);
  });
});
