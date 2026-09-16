import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { describe, expect, it } from "vitest";

import type { ProviderFailure } from "../../../providerFailure";
import { alternativeConnection, formatResetHint } from "../helpers";

function offer(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
  return {
    offer_id: "platform:deployment",
    provider_family: "autogpt",
    display_name: "AutoGPT Platform",
    auth_method: "deployment",
    credential_id: null,
    backed_by_label: "Your AutoGPT plan",
    description: "Runs on your AutoGPT plan.",
    state: "ready",
    selectable: true,
    is_default: true,
    tiers: [],
    limitations: [],
    lock_reason: null,
    unlock_href: null,
    ...over,
  } as AIConnectionOffer;
}

function chatgpt(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
  return offer({
    offer_id: "codex:cred-1",
    provider_family: "openai",
    display_name: "ChatGPT",
    auth_method: "chatgpt_oauth",
    credential_id: "cred-1",
    is_default: false,
    ...over,
  });
}

function limitOn(
  authProvider: string | null,
  credentialId: string | null = null,
): ProviderFailure {
  return {
    kind: "usage_limit",
    message: "Your plan's limit was reached.",
    authProvider,
    credentialId,
    resetsAt: null,
    retryable: false,
    reconnectFixesIt: false,
  };
}

describe("alternativeConnection", () => {
  it("offers the platform when a linked plan runs out", () => {
    const found = alternativeConnection(
      [offer(), chatgpt()],
      limitOn("codex", "cred-1"),
    );

    expect(found?.display_name).toBe("AutoGPT Platform");
    expect(found?.auth_provider).toBe("platform");
    expect(found?.credential_id).toBeNull();
  });

  it("never offers the connection that just refused the turn", () => {
    // It would fail again immediately, which is worse than saying there is
    // nothing to switch to.
    const found = alternativeConnection(
      [chatgpt()],
      limitOn("codex", "cred-1"),
    );

    expect(found).toBeNull();
  });

  it("keeps a second account of the same provider, which has its own quota", () => {
    const found = alternativeConnection(
      [
        chatgpt(),
        chatgpt({ offer_id: "codex:cred-2", credential_id: "cred-2" }),
      ],
      limitOn("codex", "cred-1"),
    );

    expect(found?.credential_id).toBe("cred-2");
  });

  it("excludes the whole provider when the failure named no account", () => {
    const found = alternativeConnection([offer()], limitOn("platform", null));

    expect(found).toBeNull();
  });

  it("ignores a connection the user cannot select", () => {
    const found = alternativeConnection(
      [offer({ selectable: false }), chatgpt()],
      limitOn("codex", "cred-1"),
    );

    expect(found).toBeNull();
  });

  it("has nothing to offer when there are no other connections", () => {
    expect(alternativeConnection([], limitOn("codex", "cred-1"))).toBeNull();
    expect(
      alternativeConnection(undefined, limitOn("codex", "cred-1")),
    ).toBeNull();
  });
});

describe("formatResetHint", () => {
  it("says nothing when the provider reported no reset time", () => {
    // An invented "try again in an hour" is worse than silence, because
    // people plan around it.
    expect(formatResetHint(null)).toBeNull();
  });

  it("says nothing for a reset that has already passed", () => {
    expect(formatResetHint(Math.floor(Date.now() / 1000) - 60)).toBeNull();
  });

  it("rounds to something a person would say", () => {
    const now = Math.floor(Date.now() / 1000);
    expect(formatResetHint(now + 30)).toBe("It resets in under a minute.");
    expect(formatResetHint(now + 20 * 60)).toBe(
      "It resets in about 20 minutes.",
    );
    expect(formatResetHint(now + 60 * 60)).toBe("It resets in about an hour.");
    expect(formatResetHint(now + 3 * 60 * 60)).toBe(
      "It resets in about 3 hours.",
    );
  });
});
