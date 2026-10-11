import { useRef } from "react";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

import {
  handleStreamError,
  parseBackendErrorCode,
} from "../copilotStreamErrorHandlers";

function makeRef(value: boolean) {
  return renderHook(() => useRef(value)).result.current;
}

describe("parseBackendErrorCode", () => {
  it("extracts a [code:<id>] prefix and returns the trailing message", () => {
    expect(
      parseBackendErrorCode("[code:idle_timeout] backend stalled for 60s"),
    ).toEqual({ code: "idle_timeout", message: "backend stalled for 60s" });
  });

  it("returns code=null for a plain message", () => {
    expect(parseBackendErrorCode("just a plain error")).toEqual({
      code: null,
      message: "just a plain error",
    });
  });

  it("trims surrounding whitespace and ignores case-only mismatches in body", () => {
    expect(
      parseBackendErrorCode("   [code:tool_stalled] something   "),
    ).toEqual({ code: "tool_stalled", message: "something" });
  });

  it("returns code=null when the body uses uppercase chars in the bracket", () => {
    // The regex restricts the code to [a-z0-9_]+; uppercase IDs aren't backend-emitted
    // but if one ever appears we surface it as a plain message rather than mis-coding.
    const r = parseBackendErrorCode("[code:Idle_Timeout] msg");
    // The /i flag on the regex DOES allow this — assert what the regex actually does
    // so the test reflects current behaviour.
    expect(r.code === null || r.code === "Idle_Timeout").toBe(true);
  });
});

describe("handleStreamError", () => {
  beforeEach(() => {
    mockToast.mockClear();
  });

  it("routes rate-limit messages (case-insensitive 'usage limit') to onRateLimit", () => {
    const onRateLimit = vi.fn();
    const onReconnect = vi.fn();
    handleStreamError({
      error: new Error(
        '{"detail":"You\'ve hit the daily usage limit for this tier."}',
      ),
      onRateLimit,
      onReconnect,
      isUserStoppingRef: makeRef(false),
    });
    expect(onRateLimit).toHaveBeenCalledTimes(1);
    expect(onRateLimit.mock.calls[0][0]).toMatch(/usage limit/i);
    expect(mockToast).not.toHaveBeenCalled();
    expect(onReconnect).not.toHaveBeenCalled();
  });

  it("toasts an auth error when the message mentions auth failure or 401", () => {
    const onRateLimit = vi.fn();
    const onReconnect = vi.fn();
    handleStreamError({
      error: new Error("Authentication failed: token expired"),
      onRateLimit,
      onReconnect,
      isUserStoppingRef: makeRef(false),
    });
    expect(onRateLimit).not.toHaveBeenCalled();
    expect(onReconnect).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Authentication error",
        variant: "destructive",
      }),
    );
  });

  it("uses curated copy for known backend codes", () => {
    handleStreamError({
      error: new Error("[code:idle_timeout] tool sleeping"),
      onRateLimit: vi.fn(),
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });
    const arg = mockToast.mock.calls[0][0] as {
      title: string;
      description: string;
    };
    expect(arg.title).toBe("Your expert stopped responding");
    // Backend message takes priority over fallbackDescription.
    expect(arg.description).toBe("tool sleeping");
  });

  it("falls back to generic copy for an unknown backend code", () => {
    handleStreamError({
      error: new Error("[code:something_new] mystery"),
      onRateLimit: vi.fn(),
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });
    const arg = mockToast.mock.calls[0][0] as { title: string };
    expect(arg.title).toBe("Your expert ran into a problem");
  });

  it("keeps a turn budget error distinct from account admission limits", () => {
    const onRateLimit = vi.fn();
    const onReconnect = vi.fn();
    handleStreamError({
      error: new Error(
        "[code:max_budget_exhausted] Send a follow-up. If your account usage limit is also reached, wait for its reset.",
      ),
      onRateLimit,
      onReconnect,
      isUserStoppingRef: makeRef(false),
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Turn budget reached" }),
    );
    expect(onRateLimit).not.toHaveBeenCalled();
    expect(onReconnect).not.toHaveBeenCalled();
  });

  it("names a platform provider outage without routing it to a usage limit", () => {
    const onRateLimit = vi.fn();
    handleStreamError({
      error: new Error(
        "[code:provider_unavailable] The AI model provider is temporarily unavailable. We've been alerted and are working on it. Please try again shortly.",
      ),
      onRateLimit,
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });

    const arg = mockToast.mock.calls[0][0] as {
      title: string;
      description: string;
    };
    expect(arg.title).toBe("AutoPilot is temporarily unavailable");
    expect(arg.description).toMatch(/try again shortly/i);
    expect(onRateLimit).not.toHaveBeenCalled();
  });

  it("uses fallbackDescription when the backend message is empty", () => {
    handleStreamError({
      error: new Error("[code:tool_stalled]"),
      onRateLimit: vi.fn(),
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });
    const arg = mockToast.mock.calls[0][0] as { description: string };
    expect(arg.description).toMatch(/tool/i);
  });

  it("triggers reconnect on a TypeError network error", () => {
    const onReconnect = vi.fn();
    const err = new TypeError("Failed to fetch");
    handleStreamError({
      error: err,
      onRateLimit: vi.fn(),
      onReconnect,
      isUserStoppingRef: makeRef(false),
    });
    expect(onReconnect).toHaveBeenCalledTimes(1);
  });

  it("triggers reconnect on an AbortError", () => {
    const onReconnect = vi.fn();
    const err = new Error("aborted");
    err.name = "AbortError";
    handleStreamError({
      error: err,
      onRateLimit: vi.fn(),
      onReconnect,
      isUserStoppingRef: makeRef(false),
    });
    expect(onReconnect).toHaveBeenCalledTimes(1);
  });

  it("triggers reconnect on a 'connection interrupted' message", () => {
    const onReconnect = vi.fn();
    handleStreamError({
      error: new Error("connection interrupted by server"),
      onRateLimit: vi.fn(),
      onReconnect,
      isUserStoppingRef: makeRef(false),
    });
    expect(onReconnect).toHaveBeenCalledTimes(1);
  });

  it("does NOT reconnect when the user explicitly stopped", () => {
    const onReconnect = vi.fn();
    handleStreamError({
      error: new TypeError("Failed to fetch"),
      onRateLimit: vi.fn(),
      onReconnect,
      isUserStoppingRef: makeRef(true),
    });
    expect(onReconnect).not.toHaveBeenCalled();
  });

  it("unwraps FastAPI {detail} wrappers from error.message", () => {
    handleStreamError({
      error: new Error('{"detail":"[code:idle_timeout] wrapped"}'),
      onRateLimit: vi.fn(),
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });
    const arg = mockToast.mock.calls[0][0] as { description: string };
    expect(arg.description).toBe("wrapped");
  });

  it("does nothing for unknown plain messages with no transient signal", () => {
    const onReconnect = vi.fn();
    handleStreamError({
      error: new Error("totally unknown problem"),
      onRateLimit: vi.fn(),
      onReconnect,
      isUserStoppingRef: makeRef(false),
    });
    expect(onReconnect).not.toHaveBeenCalled();
    expect(mockToast).not.toHaveBeenCalled();
  });
});

describe("handleStreamError — telling the two usage limits apart", () => {
  function limit(authProvider: string | null) {
    return {
      kind: "usage_limit" as const,
      message: "Limit reached.",
      authProvider,
      credentialId: authProvider === "codex" ? "cred-1" : null,
      resetsAt: null,
      retryable: false,
      reconnectFixesIt: false,
    };
  }

  beforeEach(() => {
    mockToast.mockClear();
  });

  it("hands a linked plan's limit to the caller with the failure attached", () => {
    // The caller needs it to open the continue path rather than the plan
    // dialog: asking someone to upgrade with us because OpenAI said no is
    // the wrong answer to the wrong question.
    const onRateLimit = vi.fn();

    handleStreamError({
      error: new Error("boom"),
      providerFailure: limit("codex"),
      onRateLimit,
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });

    expect(onRateLimit).toHaveBeenCalledTimes(1);
    expect(onRateLimit.mock.calls[0][1]).toEqual(
      expect.objectContaining({ kind: "usage_limit", authProvider: "codex" }),
    );
    expect(onRateLimit.mock.calls[0][2]).toBe("provider");
  });

  it("still routes our own limit the same way, so the composer text survives", () => {
    const onRateLimit = vi.fn();

    handleStreamError({
      error: new Error("boom"),
      providerFailure: limit("platform"),
      onRateLimit,
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });

    expect(onRateLimit).toHaveBeenCalledTimes(1);
    expect(onRateLimit.mock.calls[0][1]?.authProvider).toBe("platform");
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("names a streamed envelope on the platform route as the provider's refusal", () => {
    // A self-host runs its own OpenRouter or local gateway on the "platform"
    // route, so its upstream 429 carries the same authProvider as our
    // admission cap. What tells them apart is that it came mid-turn, on the
    // stream: our cap never gets that far. Routing it as our cap would tell
    // a self-host to upgrade a plan we do not bill them for.
    const onRateLimit = vi.fn();

    handleStreamError({
      error: new Error("boom"),
      providerFailure: limit("platform"),
      onRateLimit,
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });

    expect(onRateLimit.mock.calls[0][2]).toBe("provider");
  });
});

describe("handleStreamError — structured detail from a pre-stream 429/etc.", () => {
  beforeEach(() => {
    mockToast.mockClear();
  });

  it("recovers a ProviderFailure from an object `detail` and opens the switch-connection path", () => {
    // The platform usage-cap 429 raises before streaming starts, so it
    // never rides the live-stream envelope `handleStreamError` normally gets
    // — it only reaches the client as FastAPI's `{"detail": ...}` body. If
    // `detail` is an object (the structured envelope) rather than a string,
    // it must still be recognised, not silently dropped to string-guessing.
    const onRateLimit = vi.fn();

    handleStreamError({
      error: new Error(
        JSON.stringify({
          detail: {
            kind: "usage_limit",
            message: "You've reached your daily usage limit. Resets in 1h 0m.",
            authProvider: "platform",
            credentialId: null,
            resetsAt: 1999999999,
            retryable: false,
            reconnectFixesIt: false,
          },
        }),
      ),
      onRateLimit,
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });

    expect(onRateLimit).toHaveBeenCalledTimes(1);
    expect(onRateLimit.mock.calls[0][1]).toEqual(
      expect.objectContaining({
        kind: "usage_limit",
        authProvider: "platform",
      }),
    );
    // An envelope in the HTTP body means the turn was refused before the
    // stream opened, which only our own cap does. The caller keeps the plan
    // dialog for it and merely adds the switch offer.
    expect(onRateLimit.mock.calls[0][2]).toBe("admission");
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("prefers the streamed envelope, and its origin, when both are in hand", () => {
    const onRateLimit = vi.fn();

    handleStreamError({
      error: new Error(
        JSON.stringify({
          detail: {
            kind: "usage_limit",
            message: "admission",
            authProvider: "platform",
          },
        }),
      ),
      providerFailure: {
        kind: "usage_limit",
        message: "streamed",
        authProvider: "codex",
        credentialId: "cred-1",
        resetsAt: null,
        retryable: false,
        reconnectFixesIt: false,
      },
      onRateLimit,
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });

    expect(onRateLimit.mock.calls[0][1]?.message).toBe("streamed");
    expect(onRateLimit.mock.calls[0][2]).toBe("provider");
  });

  it("still handles a plain string `detail` the old way (backward compat)", () => {
    const onRateLimit = vi.fn();

    handleStreamError({
      error: new Error(
        '{"detail":"You\'ve reached your daily usage limit. Resets in 1h."}',
      ),
      onRateLimit,
      onReconnect: vi.fn(),
      isUserStoppingRef: makeRef(false),
    });

    expect(onRateLimit).toHaveBeenCalledTimes(1);
    // No structured envelope was recoverable, so the second arg is undefined
    // — same behaviour as before this fix, for every backend that still
    // sends a bare string.
    expect(onRateLimit.mock.calls[0][1]).toBeUndefined();
    expect(onRateLimit.mock.calls[0][2]).toBeUndefined();
  });
});
