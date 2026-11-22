import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { describe, expect, it } from "vitest";

import type { ProviderFailure } from "../../../providerFailure";
import { alternativeConnection, formatResetHint } from "../helpers";

function offer(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
  return {
    offer_id: "platform:deployment",
    provider_family: "autogpt",
    auth_provider: "platform",
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
  };
}

function chatgpt(over: Partial<AIConnectionOffer> = {}): AIConnectionOffer {
  return offer({
    offer_id: "codex:cred-1",
    provider_family: "openai",
    auth_provider: "codex",
    display_name: "ChatGPT",
    auth_method: "chatgpt_oauth",
    credential_id: "cred-1",
    is_default: false,
    ...over,
  });
}

function microsoft365(): AIConnectionOffer {
  return offer({
    offer_id: "microsoft_365_copilot:cred-9",
    provider_family: "microsoft",
    auth_provider: "microsoft_365_copilot",
    display_name: "Microsoft 365 Copilot",
    auth_method: "oauth",
    credential_id: "cred-9",
    is_default: false,
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

  it("continues an out-of-budget platform chat on ChatGPT, never Microsoft 365 Copilot", () => {
    expect(
      alternativeConnection([offer(), microsoft365()], limitOn("platform")),
    ).toBeNull();
    expect(
      alternativeConnection(
        [offer(), microsoft365(), chatgpt()],
        limitOn("platform"),
      )?.display_name,
    ).toBe("ChatGPT");
  });

  it("never offers Microsoft 365 Copilot when ChatGPT runs out either", () => {
    // It answers plain messages: none of AutoPilot's tools, none of the task.
    expect(
      alternativeConnection(
        [offer({ selectable: false }), microsoft365(), chatgpt()],
        limitOn("codex", "cred-1"),
      ),
    ).toBeNull();
  });

  it("knows the connection that failed by its provider, not its id", () => {
    expect(
      alternativeConnection(
        [
          offer({ selectable: false }),
          chatgpt({ offer_id: "personal-account" }),
        ],
        limitOn("codex", "cred-1"),
      ),
    ).toBeNull();
  });

  it("goes by each connection's provider, not the shape of its id", () => {
    const workAccount = { ...microsoft365(), offer_id: "work-account" };
    const personal = chatgpt({ offer_id: "personal-account" });

    expect(
      alternativeConnection(
        [offer(), workAccount, personal],
        limitOn("platform"),
      ),
    ).toEqual({
      display_name: "ChatGPT",
      auth_provider: "codex",
      credential_id: "cred-1",
    });
    expect(
      alternativeConnection(
        [offer({ selectable: false }), workAccount],
        limitOn("codex", "cred-1"),
      ),
    ).toBeNull();
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
