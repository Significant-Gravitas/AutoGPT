import { Key, storage } from "@/services/storage/local-storage";
import { create } from "zustand";

interface Org {
  id: string;
  name: string;
  slug: string;
  avatarUrl: string | null;
  isPersonal: boolean;
  memberCount: number;
}

interface Workspace {
  id: string;
  name: string;
  slug: string | null;
  isDefault: boolean;
  joinPolicy: string;
  orgId: string;
}

interface OrgWorkspaceState {
  activeOrgID: string | null;
  activeWorkspaceID: string | null;
  orgs: Org[];
  workspaces: Workspace[];
  isLoaded: boolean;

  setActiveOrg(orgID: string): void;
  setActiveWorkspace(workspaceID: string | null): void;
  setOrgs(orgs: Org[]): void;
  setWorkspaces(workspaces: Workspace[]): void;
  setLoaded(loaded: boolean): void;
  clearContext(): void;
}

export const useOrgWorkspaceStore = create<OrgWorkspaceState>((set) => ({
  activeOrgID: storage.get(Key.ACTIVE_ORG) || null,
  activeWorkspaceID: storage.get(Key.ACTIVE_WORKSPACE) || null,
  orgs: [],
  workspaces: [],
  isLoaded: false,

  setActiveOrg(orgID: string) {
    storage.set(Key.ACTIVE_ORG, orgID);
    set({ activeOrgID: orgID, activeWorkspaceID: null });
    // Clear workspace when switching org — provider will resolve default
    storage.clean(Key.ACTIVE_WORKSPACE);
  },

  setActiveWorkspace(workspaceID: string | null) {
    if (workspaceID) {
      storage.set(Key.ACTIVE_WORKSPACE, workspaceID);
    } else {
      storage.clean(Key.ACTIVE_WORKSPACE);
    }
    set({ activeWorkspaceID: workspaceID });
  },

  setOrgs(orgs: Org[]) {
    set({ orgs });
  },

  setWorkspaces(workspaces: Workspace[]) {
    set({ workspaces });
  },

  setLoaded(loaded: boolean) {
    set({ isLoaded: loaded });
  },

  clearContext() {
    storage.clean(Key.ACTIVE_ORG);
    storage.clean(Key.ACTIVE_WORKSPACE);
    set({
      activeOrgID: null,
      activeWorkspaceID: null,
      orgs: [],
      workspaces: [],
      isLoaded: false,
    });
  },
}));
