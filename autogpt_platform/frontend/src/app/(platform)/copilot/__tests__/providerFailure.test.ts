import { describe, expect, it } from "vitest";

import {
  describeProviderFailure,
  latestProviderFailure,
  parseProviderFailurePart,
  PROVIDER_FAILURE_KEY,
  providerFailureFingerprint,
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

describe("latestProviderFailure", () => {
  const envelope = {
    kind: "usage_limit",
    message: "Your ChatGPT plan is out of turns.",
    authProvider: "codex",
    credentialId: "cred-1",
    resetsAt: 1767225600,
    retryable: false,
    reconnectFixesIt: false,
  };

  it("recovers the failure a reopened chat is still sitting on", () => {
    // The backend persists the envelope onto the marker row precisely so a
    // reload can still offer the way out. Reading it only from the live
    // stream meant a refresh silently removed the switch-connection control
    // and left the chat latched to the connection that had just refused it.
    const failure = latestProviderFailure([
      { role: "user", content: "hi", metadata: null },
      {
        role: "assistant",
        content: "[__COPILOT_ERROR_f7a1__] out of turns",
        metadata: { [PROVIDER_FAILURE_KEY]: envelope },
      },
    ]);
    expect(failure?.kind).toBe("usage_limit");
    expect(failure?.credentialId).toBe("cred-1");
  });

  it("takes the newest one, so a recovered-from failure stays history", () => {
    const older = { ...envelope, credentialId: "old" };
    const newer = { ...envelope, credentialId: "new" };
    const failure = latestProviderFailure([
      {
        role: "assistant",
        content: "[__COPILOT_ERROR_f7a1__] old",
        metadata: { [PROVIDER_FAILURE_KEY]: older },
      },
      {
        role: "assistant",
        content: "[__COPILOT_ERROR_f7a1__] new",
        metadata: { [PROVIDER_FAILURE_KEY]: newer },
      },
    ]);
    expect(failure?.credentialId).toBe("new");
  });

  it("does not resurrect a failure the chat has since recovered from", () => {
    // A successful turn after the failure means the chat moved on. Walking
    // past it would re-offer "continue on ..." for a limit that no longer
    // applies -- on a reload, indistinguishable from failing again.
    expect(
      latestProviderFailure([
        {
          role: "assistant",
          content: "[__COPILOT_ERROR_f7a1__] out of turns",
          metadata: { [PROVIDER_FAILURE_KEY]: envelope },
        },
        { role: "user", content: "try again please", metadata: null },
        { role: "assistant", content: "Done.", metadata: null },
      ]),
    ).toBeNull();
  });

  it("does not resurrect a failure after the chat switched connections", () => {
    expect(
      latestProviderFailure(
        [
          {
            role: "assistant",
            content: "[__COPILOT_ERROR_f7a1__] out of turns",
            metadata: { [PROVIDER_FAILURE_KEY]: envelope },
          },
        ],
        { authProvider: "platform", credentialId: null },
      ),
    ).toBeNull();
  });

  it("keeps a failure when the chat is still on that connection", () => {
    expect(
      latestProviderFailure(
        [
          {
            role: "assistant",
            content: "[__COPILOT_ERROR_f7a1__] out of turns",
            metadata: { [PROVIDER_FAILURE_KEY]: envelope },
          },
        ],
        { authProvider: "codex", credentialId: "cred-1" },
      )?.kind,
    ).toBe("usage_limit");
  });

  it("is null for a chat that never failed", () => {
    expect(
      latestProviderFailure([
        { role: "user", content: "hi", metadata: null },
        {
          role: "assistant",
          content: "sure",
          metadata: { kind: "expert_run" },
        },
      ]),
    ).toBeNull();
  });
});

describe("providerFailureFingerprint", () => {
  it("changes when a new connection or reset window fails", () => {
    expect(providerFailureFingerprint(failure())).not.toBe(
      providerFailureFingerprint(failure({ credentialId: "cred-2" })),
    );
    expect(providerFailureFingerprint(failure())).not.toBe(
      providerFailureFingerprint(failure({ resetsAt: 123 })),
    );
  });
});
