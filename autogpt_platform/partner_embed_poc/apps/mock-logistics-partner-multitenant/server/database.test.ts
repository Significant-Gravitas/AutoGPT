import { afterEach, describe, expect, it } from "vitest";

import { PartnerDatabase } from "./database.js";

const databases: PartnerDatabase[] = [];

afterEach(() => {
  for (const database of databases.splice(0)) database.close();
});

function database() {
  const result = new PartnerDatabase(":memory:");
  databases.push(result);
  return result;
}

describe("PartnerDatabase", () => {
  it("keeps the active organization inside the signed-in user's memberships", () => {
    const store = database();
    const sessionID = store.createSession("fd-user-2077");

    expect(sessionID).toBeDefined();
    expect(store.switchOrganization(sessionID as string, "fd-account-88")).toBe(
      false,
    );
    expect(store.session(sessionID as string)?.activeOrganization.id).toBe(
      "fd-account-77",
    );
  });

  it("stores a separate AutoGPT mapping for each tenant", () => {
    const store = database();
    const sessionID = store.createSession("fd-user-1042") as string;
    store.saveMapping("fd-user-1042", "fd-account-77", {
      autoGPTUserID: "00000000-0000-4000-8000-000000000001",
      autoGPTOrganizationID: "00000000-0000-4000-8000-000000000002",
      autoGPTTeamID: "00000000-0000-4000-8000-000000000003",
    });

    expect(store.session(sessionID)?.sync?.autoGPTOrganizationID).toBe(
      "00000000-0000-4000-8000-000000000002",
    );
    expect(store.switchOrganization(sessionID, "fd-account-88")).toBe(true);
    expect(store.session(sessionID)?.sync).toBeNull();

    store.saveMapping("fd-user-1042", "fd-account-88", {
      autoGPTUserID: "00000000-0000-4000-8000-000000000001",
      autoGPTOrganizationID: "00000000-0000-4000-8000-000000000004",
      autoGPTTeamID: "00000000-0000-4000-8000-000000000005",
    });
    expect(store.session(sessionID)?.sync?.autoGPTOrganizationID).toBe(
      "00000000-0000-4000-8000-000000000004",
    );
  });

  it("builds assertions from the current user and tenant", () => {
    const store = database();
    const sessionID = store.createSession("fd-user-1042") as string;
    expect(store.identity(sessionID)).toMatchObject({
      subject: "fd-user-1042",
      accountID: "fd-account-77",
      accountName: "Northstar Freight",
      roles: ["manager"],
      capabilities: [
        "jobs.read",
        "reports.read",
        "documents.read",
        "documents.write",
        "agents.create",
        "agents.run",
        "agents.schedule",
        "autogpt:block:c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
        "autogpt:block:363ae599-353e-4804-937e-b2ee3cef3da4",
        "autogpt:block:b1ab9b19-67a6-406d-abf5-2dba76d00c79",
      ],
    });
  });
});
