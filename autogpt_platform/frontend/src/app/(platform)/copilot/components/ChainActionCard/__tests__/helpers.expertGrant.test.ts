import { describe, expect, it, vi } from "vitest";
import {
  intersectGrants,
  toConnectorRows,
  type ConnectorRequest,
} from "../helpers";

function request(
  id: string,
  credentials: { id: string; title: string; type: string }[],
): ConnectorRequest {
  return {
    id,
    fields: [
      [
        "credentials",
        {
          credentials_provider: ["github"],
          credentials_types: ["oauth2"],
          expert_grant: { expertId: "expert-a", credentials },
        },
      ],
    ],
    selected: {},
    onChange: vi.fn(),
    onConnected: vi.fn(),
  };
}

describe("expert grant candidates on merged connector rows", () => {
  it("offers only accounts eligible for every merged requirement", () => {
    const shared = { id: "both", title: "Both", type: "oauth2" };
    const rows = toConnectorRows(
      [
        request("a", [shared, { id: "only-a", title: "A", type: "oauth2" }]),
        request("b", [shared, { id: "only-b", title: "B", type: "oauth2" }]),
      ],
      [],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0].expertGrant?.credentials.map((c) => c.id)).toEqual(["both"]);
  });

  it("keeps the defined side when only one requirement carries grant info", () => {
    const grant = {
      expertId: "e",
      credentials: [{ id: "x", title: "X", type: "oauth2" }],
    };
    expect(intersectGrants(undefined, grant)).toEqual(grant);
    expect(intersectGrants(grant, undefined)).toEqual(grant);
  });
});
