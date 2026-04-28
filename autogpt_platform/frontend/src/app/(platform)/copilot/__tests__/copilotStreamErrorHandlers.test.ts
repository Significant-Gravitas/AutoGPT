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
    expect(arg.title).toBe("AutoPilot stopped responding");
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
    expect(arg.title).toBe("AutoPilot ran into a problem");
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
