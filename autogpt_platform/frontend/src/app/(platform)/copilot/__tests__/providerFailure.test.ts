import { describe, expect, it } from "vitest";

import {
  describeProviderFailure,
  parseProviderFailurePart,
  type ProviderFailure,
} from "../providerFailure";

function part(data: Record<string, unknown>) {
  return { type: "data-provider-failure", data };
}

function failure(over: Partial<ProviderFailure> = {}): ProviderFailure {
  return {
    kind: "usage_limit",
    message: "",
    authProvider: "codex",
    credentialId: "cred-1",
    resetsAt: null,
    retryable: false,
    reconnectFixesIt: false,
    ...over,
  };
}

describe("parseProviderFailurePart", () => {
  it("reads a failure the server sent", () => {
    const parsed = parseProviderFailurePart(
      part({
        kind: "auth_expired",
        message: "token expired",
        authProvider: "codex",
        credentialId: "cred-1",
        resetsAt: null,
        retryable: false,
        reconnectFixesIt: true,
      }),
    );

    expect(parsed?.kind).toBe("auth_expired");
    expect(parsed?.reconnectFixesIt).toBe(true);
  });

  it("ignores other data parts", () => {
    expect(
      parseProviderFailurePart({
        type: "data-mode-changed",
        data: { mode: "fast" },
      }),
    ).toBeNull();
  });

  it("refuses a kind it does not know", () => {
    // A kind added server-side without client copy would otherwise render
    // as an empty toast.
    expect(parseProviderFailurePart(part({ kind: "invented" }))).toBeNull();
  });

  it("does not infer advice the server withheld", () => {
    // Absent means "not retryable", never "probably fine to retry".
    const parsed = parseProviderFailurePart(part({ kind: "policy_denied" }));
    expect(parsed?.retryable).toBe(false);
    expect(parsed?.reconnectFixesIt).toBe(false);
  });
});

describe("describeProviderFailure", () => {
  it("does not tell someone to retry what cannot succeed", () => {
    // The whole point of the envelope: three failures that used to get
    // "Press Try Again" now say what would actually help.
    for (const kind of [
      "auth_expired",
      "usage_limit",
      "model_unavailable",
    ] as const) {
      const copy = describeProviderFailure(failure({ kind }));
      expect(copy.description.toLowerCase()).not.toContain("try again");
    }
  });

  it("names the reset when the provider reported one", () => {
    const copy = describeProviderFailure(
      failure({ resetsAt: Math.floor(Date.now() / 1000) + 20 * 60 }),
    );
    expect(copy.description).toContain("about 20 minutes");
  });

  it("names no time when the provider reported no reset", () => {
    // The kind's own copy may mention resetting in general; what must not
    // appear is a specific time nobody reported.
    const copy = describeProviderFailure(failure({ resetsAt: null }));
    expect(copy.description).not.toMatch(/Resets in/i);
  });

  it("drops a reset that has already passed", () => {
    const copy = describeProviderFailure(
      failure({ resetsAt: Math.floor(Date.now() / 1000) - 60 }),
    );
    expect(copy.description).not.toMatch(/Resets in/i);
  });

  it("prefers the provider's own words for the detail", () => {
    const copy = describeProviderFailure(
      failure({
        kind: "policy_denied",
        message: "This request was blocked by the provider's safety policy.",
      }),
    );
    expect(copy.description).toContain("safety policy");
  });
});
