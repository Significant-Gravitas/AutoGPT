import { Key, storage } from "@/services/storage/local-storage";
import { beforeEach, describe, expect, it } from "vitest";

import { useOrgTeamStore } from "../store";

function seedStore() {
  useOrgTeamStore.setState({
    activeOrgID: null,
    activeTeamID: null,
    orgs: [],
    teams: [],
    isLoaded: false,
  });
}

const ORG = {
  id: "org-1",
  name: "Acme",
  slug: "acme",
  avatarUrl: null,
  isPersonal: false,
  memberCount: 3,
};

const TEAM = {
  id: "team-1",
  name: "Engineering",
  slug: "engineering",
  isDefault: true,
  joinPolicy: "OPEN",
  orgId: "org-1",
};

describe("useOrgTeamStore", () => {
  beforeEach(() => {
    window.localStorage.clear();
    seedStore();
  });

  it("setActiveOrg persists the org and resets the active team", () => {
    useOrgTeamStore.getState().setActiveTeam("team-stale");

    useOrgTeamStore.getState().setActiveOrg("org-1");

    const state = useOrgTeamStore.getState();
    expect(state.activeOrgID).toBe("org-1");
    expect(state.activeTeamID).toBeNull();
    expect(storage.get(Key.ACTIVE_ORG)).toBe("org-1");
    expect(storage.get(Key.ACTIVE_TEAM)).toBeNull();
  });

  it("setActiveTeam persists the team id", () => {
    useOrgTeamStore.getState().setActiveTeam("team-1");

    expect(useOrgTeamStore.getState().activeTeamID).toBe("team-1");
    expect(storage.get(Key.ACTIVE_TEAM)).toBe("team-1");
  });

  it("setActiveTeam(null) clears the persisted team", () => {
    useOrgTeamStore.getState().setActiveTeam("team-1");

    useOrgTeamStore.getState().setActiveTeam(null);

    expect(useOrgTeamStore.getState().activeTeamID).toBeNull();
    expect(storage.get(Key.ACTIVE_TEAM)).toBeNull();
  });

  it("setOrgs and setTeams replace the lists", () => {
    useOrgTeamStore.getState().setOrgs([ORG]);
    useOrgTeamStore.getState().setTeams([TEAM]);

    expect(useOrgTeamStore.getState().orgs).toEqual([ORG]);
    expect(useOrgTeamStore.getState().teams).toEqual([TEAM]);
  });

  it("clearContext wipes state and persisted org/team (logout path)", () => {
    useOrgTeamStore.getState().setActiveOrg("org-1");
    useOrgTeamStore.getState().setActiveTeam("team-1");
    useOrgTeamStore.getState().setOrgs([ORG]);
    useOrgTeamStore.getState().setTeams([TEAM]);
    useOrgTeamStore.getState().setLoaded(true);

    useOrgTeamStore.getState().clearContext();

    const state = useOrgTeamStore.getState();
    expect(state.activeOrgID).toBeNull();
    expect(state.activeTeamID).toBeNull();
    expect(state.orgs).toEqual([]);
    expect(state.teams).toEqual([]);
    expect(state.isLoaded).toBe(false);
    expect(storage.get(Key.ACTIVE_ORG)).toBeNull();
    expect(storage.get(Key.ACTIVE_TEAM)).toBeNull();
  });
});
