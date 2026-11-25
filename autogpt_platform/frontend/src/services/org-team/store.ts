import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { Key, storage } from "@/services/storage/local-storage";
import { create } from "zustand";

interface Team {
  id: string;
  name: string;
  slug: string | null;
  isDefault: boolean;
  joinPolicy: string;
  orgId: string;
}

interface OrgTeamState {
  activeOrgID: string | null;
  activeTeamID: string | null;
  orgs: OrgResponse[];
  teams: Team[];
  isLoaded: boolean;

  setActiveOrg(orgID: string): void;
  setActiveTeam(teamID: string | null): void;
  setOrgs(orgs: OrgResponse[]): void;
  setTeams(teams: Team[]): void;
  setLoaded(loaded: boolean): void;
  clearContext(): void;
}

export const useOrgTeamStore = create<OrgTeamState>((set) => ({
  activeOrgID: storage.get(Key.ACTIVE_ORG) || null,
  activeTeamID: storage.get(Key.ACTIVE_TEAM) || null,
  orgs: [],
  teams: [],
  isLoaded: false,

  setActiveOrg(orgID: string) {
    storage.set(Key.ACTIVE_ORG, orgID);
    set({ activeOrgID: orgID, activeTeamID: null });
    // Clear team when switching org — provider will resolve default
    storage.clean(Key.ACTIVE_TEAM);
  },

  setActiveTeam(teamID: string | null) {
    if (teamID) {
      storage.set(Key.ACTIVE_TEAM, teamID);
    } else {
      storage.clean(Key.ACTIVE_TEAM);
    }
    set({ activeTeamID: teamID });
  },

  setOrgs(orgs: OrgResponse[]) {
    set({ orgs });
  },

  setTeams(teams: Team[]) {
    set({ teams });
  },

  setLoaded(loaded: boolean) {
    set({ isLoaded: loaded });
  },

  clearContext() {
    storage.clean(Key.ACTIVE_ORG);
    storage.clean(Key.ACTIVE_TEAM);
    set({
      activeOrgID: null,
      activeTeamID: null,
      orgs: [],
      teams: [],
      isLoaded: false,
    });
  },
}));
