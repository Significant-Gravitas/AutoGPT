export interface DirectoryUser {
  id: string;
  name: string;
  organizations: string[];
}

export interface Organization {
  id: string;
  name: string;
  role: string;
  tools: string[];
}

export interface SyncMapping {
  autoGPTUserID: string;
  autoGPTOrganizationID: string;
  autoGPTTeamID: string;
  syncedAt: string;
}

export interface Session {
  user: {
    id: string;
    email: string;
    name: string;
  };
  activeOrganization: Organization;
  organizations: Organization[];
  sync: SyncMapping | null;
}

export interface TokenResponse {
  access_token: string;
}
