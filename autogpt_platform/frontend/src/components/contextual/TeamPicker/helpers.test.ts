import { afterEach, describe, expect, it } from "vitest";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";
import {
  getLastUsedTeam,
  getTeamRequestInit,
  setLastUsedTeam,
} from "./helpers";

afterEach(() => {
  window.localStorage.clear();
});

describe("TeamPicker last-used storage", () => {
  it("returns null for a surface that was never used", () => {
    expect(getLastUsedTeam("builder-save")).toBeNull();
  });

  it("remembers the last team picked on a surface", () => {
    setLastUsedTeam("builder-save", "team-1");
    expect(getLastUsedTeam("builder-save")).toBe("team-1");
  });

  it("keeps last-used separate per surface", () => {
    setLastUsedTeam("builder-save", "team-1");
    setLastUsedTeam("library-folder", "team-2");
    expect(getLastUsedTeam("builder-save")).toBe("team-1");
    expect(getLastUsedTeam("library-folder")).toBe("team-2");
  });

  it("treats an org-home selection as null (no team)", () => {
    setLastUsedTeam("builder-save", "team-1");
    setLastUsedTeam("builder-save", null);
    expect(getLastUsedTeam("builder-save")).toBeNull();
  });

  it("survives a corrupt storage value", () => {
    window.localStorage.setItem("create-surface-teams", "not-json");
    expect(getLastUsedTeam("builder-save")).toBeNull();
  });
});

describe("getTeamRequestInit", () => {
  it("returns undefined for org-home (null team)", () => {
    expect(getTeamRequestInit(null)).toBeUndefined();
  });

  it("stamps the X-Team-Id header for a team", () => {
    expect(getTeamRequestInit("team-9")).toEqual({
      headers: { [TEAM_HEADER_NAME]: "team-9" },
    });
  });
});
