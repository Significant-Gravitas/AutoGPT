import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";

import type { PartnerIdentity } from "./assertion.js";

interface UserRow {
  id: string;
  email: string;
  name: string;
}

interface OrganizationRow {
  id: string;
  name: string;
  role: string;
  tools_json: string;
}

interface SessionRow {
  id: string;
  user_id: string;
  active_organization_id: string;
  expires_at: number;
}

interface MappingRow {
  autogpt_user_id: string;
  autogpt_organization_id: string;
  autogpt_team_id: string;
  synced_at: string;
}

export interface OrganizationMembership {
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

export interface SessionView {
  user: UserRow;
  activeOrganization: OrganizationMembership;
  organizations: OrganizationMembership[];
  sync: SyncMapping | null;
}

export interface DirectoryUser {
  id: string;
  name: string;
  organizations: string[];
}

export class PartnerDatabase {
  private readonly database: DatabaseSync;

  constructor(path: string) {
    if (path !== ":memory:") mkdirSync(dirname(path), { recursive: true });
    this.database = new DatabaseSync(path);
    this.initialize();
  }

  directory(): DirectoryUser[] {
    const users = this.database
      .prepare("SELECT id, name FROM users ORDER BY name")
      .all() as unknown as Pick<UserRow, "id" | "name">[];
    return users.map((user) => ({
      ...user,
      organizations: this.organizationsForUser(user.id).map((org) => org.name),
    }));
  }

  createSession(userID: string): string | undefined {
    const organization = this.organizationsForUser(userID)[0];
    if (!organization) return undefined;
    const sessionID = randomUUID();
    this.database
      .prepare(
        "INSERT INTO sessions (id, user_id, active_organization_id, expires_at) VALUES (?, ?, ?, ?)",
      )
      .run(
        sessionID,
        userID,
        organization.id,
        Math.floor(Date.now() / 1000) + 86_400,
      );
    return sessionID;
  }

  deleteSession(sessionID: string): void {
    this.database.prepare("DELETE FROM sessions WHERE id = ?").run(sessionID);
  }

  session(sessionID: string): SessionView | undefined {
    const session = this.sessionRow(sessionID);
    if (!session) return undefined;
    const user = this.database
      .prepare("SELECT id, email, name FROM users WHERE id = ?")
      .get(session.user_id) as unknown as UserRow | undefined;
    const organizations = this.organizationsForUser(session.user_id);
    const activeOrganization = organizations.find(
      (organization) => organization.id === session.active_organization_id,
    );
    if (!user || !activeOrganization) return undefined;
    return {
      user,
      activeOrganization,
      organizations,
      sync: this.mapping(user.id, activeOrganization.id),
    };
  }

  switchOrganization(sessionID: string, organizationID: string): boolean {
    const session = this.sessionRow(sessionID);
    if (!session) return false;
    const allowed = this.organizationsForUser(session.user_id).some(
      (organization) => organization.id === organizationID,
    );
    if (!allowed) return false;
    this.database
      .prepare("UPDATE sessions SET active_organization_id = ? WHERE id = ?")
      .run(organizationID, sessionID);
    return true;
  }

  identity(sessionID: string): PartnerIdentity | undefined {
    const view = this.session(sessionID);
    if (!view) return undefined;
    return {
      subject: view.user.id,
      accountID: view.activeOrganization.id,
      email: view.user.email,
      name: view.user.name,
      accountName: view.activeOrganization.name,
      roles: [view.activeOrganization.role],
      capabilities: view.activeOrganization.tools,
    };
  }

  saveMapping(
    userID: string,
    organizationID: string,
    mapping: Omit<SyncMapping, "syncedAt">,
  ): SyncMapping {
    const syncedAt = new Date().toISOString();
    this.database
      .prepare(
        "INSERT INTO sync_mappings (user_id, organization_id, autogpt_user_id, autogpt_organization_id, autogpt_team_id, synced_at) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, organization_id) DO UPDATE SET autogpt_user_id = excluded.autogpt_user_id, autogpt_organization_id = excluded.autogpt_organization_id, autogpt_team_id = excluded.autogpt_team_id, synced_at = excluded.synced_at",
      )
      .run(
        userID,
        organizationID,
        mapping.autoGPTUserID,
        mapping.autoGPTOrganizationID,
        mapping.autoGPTTeamID,
        syncedAt,
      );
    return { ...mapping, syncedAt };
  }

  close(): void {
    this.database.close();
  }

  private sessionRow(sessionID: string): SessionRow | undefined {
    return this.database
      .prepare(
        "SELECT id, user_id, active_organization_id, expires_at FROM sessions WHERE id = ? AND expires_at > ?",
      )
      .get(sessionID, Math.floor(Date.now() / 1000)) as unknown as
      | SessionRow
      | undefined;
  }

  private organizationsForUser(userID: string): OrganizationMembership[] {
    const rows = this.database
      .prepare(
        "SELECT organizations.id, organizations.name, memberships.role, memberships.tools_json FROM organizations JOIN memberships ON memberships.organization_id = organizations.id WHERE memberships.user_id = ? ORDER BY organizations.id",
      )
      .all(userID) as unknown as OrganizationRow[];
    return rows.map((row) => ({
      id: row.id,
      name: row.name,
      role: row.role,
      tools: JSON.parse(row.tools_json) as string[],
    }));
  }

  private mapping(userID: string, organizationID: string): SyncMapping | null {
    const row = this.database
      .prepare(
        "SELECT autogpt_user_id, autogpt_organization_id, autogpt_team_id, synced_at FROM sync_mappings WHERE user_id = ? AND organization_id = ?",
      )
      .get(userID, organizationID) as unknown as MappingRow | undefined;
    if (!row) return null;
    return {
      autoGPTUserID: row.autogpt_user_id,
      autoGPTOrganizationID: row.autogpt_organization_id,
      autoGPTTeamID: row.autogpt_team_id,
      syncedAt: row.synced_at,
    };
  }

  private initialize(): void {
    this.database.exec(
      "PRAGMA foreign_keys = ON;" +
        "CREATE TABLE IF NOT EXISTS organizations (id TEXT PRIMARY KEY, name TEXT NOT NULL);" +
        "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL);" +
        "CREATE TABLE IF NOT EXISTS memberships (user_id TEXT NOT NULL REFERENCES users(id), organization_id TEXT NOT NULL REFERENCES organizations(id), role TEXT NOT NULL, tools_json TEXT NOT NULL, PRIMARY KEY (user_id, organization_id));" +
        "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), active_organization_id TEXT NOT NULL REFERENCES organizations(id), expires_at INTEGER NOT NULL);" +
        "CREATE TABLE IF NOT EXISTS sync_mappings (user_id TEXT NOT NULL REFERENCES users(id), organization_id TEXT NOT NULL REFERENCES organizations(id), autogpt_user_id TEXT NOT NULL, autogpt_organization_id TEXT NOT NULL, autogpt_team_id TEXT NOT NULL, synced_at TEXT NOT NULL, PRIMARY KEY (user_id, organization_id));",
    );
    this.seed();
  }

  private seed(): void {
    const insertOrganization = this.database.prepare(
      "INSERT OR IGNORE INTO organizations (id, name) VALUES (?, ?)",
    );
    insertOrganization.run("fd-account-77", "Northstar Freight");
    insertOrganization.run("fd-account-88", "Harbour & Rail Logistics");

    const insertUser = this.database.prepare(
      "INSERT OR IGNORE INTO users (id, email, name) VALUES (?, ?, ?)",
    );
    insertUser.run("fd-user-1042", "alex@northstarfreight.com", "Alex Morgan");
    insertUser.run("fd-user-2077", "priya@northstarfreight.com", "Priya Shah");

    const insertMembership = this.database.prepare(
      "INSERT INTO memberships (user_id, organization_id, role, tools_json) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, organization_id) DO UPDATE SET role = excluded.role, tools_json = excluded.tools_json",
    );
    insertMembership.run(
      "fd-user-1042",
      "fd-account-77",
      "manager",
      JSON.stringify([
        "jobs.read",
        "reports.read",
        "documents.read",
        "documents.write",
        "autogpt:block:b1ab9b19-67a6-406d-abf5-2dba76d00c79",
      ]),
    );
    insertMembership.run(
      "fd-user-1042",
      "fd-account-88",
      "operator",
      JSON.stringify(["jobs.read", "reports.read", "documents.read"]),
    );
    insertMembership.run(
      "fd-user-2077",
      "fd-account-77",
      "operator",
      JSON.stringify(["jobs.read", "documents.read"]),
    );
  }
}
