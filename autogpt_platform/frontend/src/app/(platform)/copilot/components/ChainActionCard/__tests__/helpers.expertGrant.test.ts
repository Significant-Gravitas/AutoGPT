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

describe("a merged row reports whether every card it answers is answered", () => {
  const picked = {
    id: "cred-1",
    provider: "github",
    type: "oauth2" as const,
    title: "GH",
  };

  function withSelection(
    id: string,
    selected: ConnectorRequest["selected"],
  ): ConnectorRequest {
    return { ...request(id, []), selected };
  }

  it("is unanswered while a second card's own field is still empty", () => {
    // The row reports the FIRST target's value, so without this it reads as
    // answered and the second card sits blocked on a credential it has.
    const rows = toConnectorRows(
      [withSelection("a", { credentials: picked }), withSelection("b", {})],
      [],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0].selected?.id).toBe("cred-1");
    expect(rows[0].hasUnansweredTarget).toBe(true);
  });

  it("is answered once both cards hold the same credential", () => {
    const rows = toConnectorRows(
      [
        withSelection("a", { credentials: picked }),
        withSelection("b", { credentials: picked }),
      ],
      [],
    );
    expect(rows[0].hasUnansweredTarget).toBe(false);
  });

  it("is answered when nothing is picked yet", () => {
    const rows = toConnectorRows(
      [withSelection("a", {}), withSelection("b", {})],
      [],
    );
    expect(rows[0].hasUnansweredTarget).toBe(false);
  });
});

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

  it("offers nothing when only one requirement carries grant info", () => {
    const grant = {
      expertId: "e",
      credentials: [{ id: "x", title: "X", type: "oauth2" }],
    };
    const closed = { expertId: "e", credentials: [] };
    expect(intersectGrants(undefined, grant)).toEqual(closed);
    expect(intersectGrants(grant, undefined)).toEqual(closed);
  });

  it("offers nothing when the merged requirements name different experts", () => {
    const kept = {
      expertId: "expert-a",
      credentials: [{ id: "x", title: "X", type: "oauth2" }],
    };
    const incoming = {
      expertId: "expert-b",
      credentials: [{ id: "x", title: "X", type: "oauth2" }],
    };
    expect(intersectGrants(kept, incoming)).toEqual({
      expertId: "expert-a",
      credentials: [],
    });
  });

  it("leaves a row without any grant info in personal mode", () => {
    expect(intersectGrants(undefined, undefined)).toBeUndefined();
  });
});
