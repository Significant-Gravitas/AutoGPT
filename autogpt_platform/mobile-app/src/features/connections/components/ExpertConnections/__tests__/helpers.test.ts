import type { ConnectableService, ExpertNeed } from "../../../listHelpers";
import { needRows } from "../helpers";

const hubspot: ConnectableService = {
  key: "hubspot",
  name: "HubSpot",
  description: null,
  providers: ["hubspot"],
  action: { kind: "oauth", provider: "hubspot" },
};

const connectNeed: ExpertNeed = {
  key: "connect:hubspot",
  resolution: "connect",
  providers: ["hubspot"],
  credentialID: null,
  workflowNames: ["Weekly report", "Daily digest"],
};

const allowNeed: ExpertNeed = {
  key: "allow:c9",
  resolution: "allow",
  providers: ["github"],
  credentialID: "c9",
  workflowNames: [],
};

describe("needRows", () => {
  it("offers the service's own way in, keyed by the need", () => {
    const [row] = needRows({ needs: [connectNeed], services: [hubspot], grantedIDs: [] });
    expect(row).toMatchObject({
      key: "connect:hubspot",
      name: "HubSpot",
      detail: "Weekly report, Daily digest",
      label: "Sign in",
      action: { kind: "connect", service: { key: "connect:hubspot", action: hubspot.action } },
    });
  });

  it("sends a need its service's lead provider can't meet to the web", () => {
    const need = { ...connectNeed, key: "connect:hubspot_crm", providers: ["hubspot_crm"] };
    const service = { ...hubspot, providers: ["hubspot", "hubspot_crm"] };
    expect(needRows({ needs: [need], services: [service], grantedIDs: [] })[0]).toMatchObject({
      label: "Open",
      action: { kind: "connect", service: { action: { kind: "web" } } },
    });
  });

  it("offers Allow for an account, and drops one allowed since", () => {
    expect(needRows({ needs: [allowNeed], services: null, grantedIDs: [] })[0]).toMatchObject({
      name: "GitHub",
      detail: "A scheduled workflow",
      label: "Allow",
      action: { kind: "allow", credentialID: "c9" },
    });
    expect(needRows({ needs: [allowNeed], services: [], grantedIDs: ["c9"] })).toEqual([]);
  });

  it("waits for the services, then sends an unknown one to the web", () => {
    expect(needRows({ needs: [connectNeed], services: null, grantedIDs: [] })[0]).toMatchObject({
      label: null,
      action: null,
    });
    expect(needRows({ needs: [connectNeed], services: [], grantedIDs: [] })[0]).toMatchObject({
      name: "HubSpot",
      label: "Open",
      action: { kind: "connect", service: { action: { kind: "web" } } },
    });
  });
});
