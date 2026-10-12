import { afterEach, describe, expect, it, vi } from "vitest";

const setExtras = vi.fn();
vi.mock("@sentry/nextjs", () => ({
  withScope: (callback: (scope: { setExtras: (e: unknown) => void }) => void) =>
    callback({ setExtras }),
}));

import {
  describeCloseEvent,
  describeErrorEvent,
  logWebSocketIssue,
} from "../websocket-logging";

afterEach(() => {
  vi.restoreAllMocks();
  setExtras.mockClear();
});

describe("describeCloseEvent", () => {
  it("puts the code, its name, the reason and wasClean in the summary", () => {
    const { summary } = describeCloseEvent(
      { code: 4002, reason: "Invalid token", wasClean: true },
      "wss://ws.example.com/ws",
      "connecting",
    );

    expect(summary).toContain("4002");
    expect(summary).toContain("invalid token");
    expect(summary).toContain("Invalid token");
    expect(summary).toContain("wasClean true");
  });

  it("names an abnormal close and marks an empty reason", () => {
    // 1006 with an empty reason is the case that used to log as
    // "WebSocket failed to connect:  [object CloseEvent]".
    const { summary, extra } = describeCloseEvent(
      { code: 1006, reason: "", wasClean: false },
      "wss://ws.example.com/ws",
      "connecting",
    );

    expect(summary).toBe(
      'code 1006 (abnormal closure), reason "(none)", wasClean false',
    );
    expect(extra).toEqual({
      ws_close_code: 1006,
      ws_close_code_name: "abnormal closure",
      ws_close_reason: "",
      ws_was_clean: false,
      ws_phase: "connecting",
      ws_url: "wss://ws.example.com/ws",
    });
  });

  it("labels a code it does not know instead of dropping it", () => {
    const { summary, extra } = describeCloseEvent(
      { code: 4999, reason: "", wasClean: false },
      "wss://ws.example.com/ws",
      "connected",
    );

    expect(summary).toContain("4999");
    expect(summary).toContain("unknown code");
    expect(extra.ws_phase).toBe("connected");
  });

  it("strips the query string so the auth token is never reported", () => {
    const { extra } = describeCloseEvent(
      { code: 1000, reason: "", wasClean: true },
      "wss://ws.example.com/ws?token=secret-token",
      "connected",
    );

    expect(extra.ws_url).toBe("wss://ws.example.com/ws");
    expect(JSON.stringify(extra)).not.toContain("secret-token");
  });

  it("strips the token wherever it sits in the query, even from a non-URL", () => {
    const { summary, extra } = describeCloseEvent(
      { code: 1000, reason: "", wasClean: true },
      "not-a-url?x=1&token=secret-token",
      "connected",
    );

    expect(extra.ws_url).toBe("not-a-url");
    expect(summary + JSON.stringify(extra)).not.toContain("secret-token");
  });

  it("treats a missing reason as empty", () => {
    const { summary, extra } = describeCloseEvent(
      {
        code: 1000,
        reason: undefined as unknown as string,
        wasClean: true,
      },
      "wss://ws.example.com/ws",
      "connected",
    );

    expect(summary).toContain('reason "(none)"');
    expect(extra.ws_close_reason).toBe("");
  });
});

describe("describeErrorEvent", () => {
  it("reports the event type, the socket state and the target URL", () => {
    const { summary, extra } = describeErrorEvent(
      {
        type: "error",
        target: { readyState: 3, url: "wss://ws.example.com/ws?token=secret" },
      },
      "wss://fallback.example.com/ws",
    );

    expect(summary).toBe(
      'type "error", readyState CLOSED, url wss://ws.example.com/ws',
    );
    expect(extra).toEqual({
      ws_event_type: "error",
      ws_ready_state: 3,
      ws_ready_state_name: "CLOSED",
      ws_url: "wss://ws.example.com/ws",
    });
    expect(JSON.stringify(extra)).not.toContain("secret");
  });

  it("labels a readyState it does not recognise", () => {
    const { summary, extra } = describeErrorEvent(
      {
        type: "error",
        target: { readyState: 7, url: "wss://ws.example.com/ws" },
      },
      "wss://ws.example.com/ws",
    );

    expect(summary).toContain("readyState unknown");
    expect(extra.ws_ready_state).toBe(7);
  });

  it("falls back to the configured URL when the event has no target", () => {
    const { summary, extra } = describeErrorEvent(
      { type: "", target: null },
      "wss://ws.example.com/ws",
    );

    expect(summary).toBe(
      'type "error", readyState unknown, url wss://ws.example.com/ws',
    );
    expect(extra.ws_ready_state).toBeNull();
  });
});

describe("logWebSocketIssue", () => {
  it("logs the message alone and hands the details to Sentry as extras", () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);

    logWebSocketIssue("error", "[BackendAPI] boom", { ws_close_code: 4002 });

    // Only the string: passing the event object here is what Sentry's console
    // integration used to stringify into "[object CloseEvent]".
    expect(consoleError).toHaveBeenCalledWith("[BackendAPI] boom");
    expect(consoleError.mock.calls[0]).toHaveLength(1);
    expect(setExtras).toHaveBeenCalledWith({ ws_close_code: 4002 });
  });

  it("logs at warn level when asked to", () => {
    const consoleWarn = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);

    logWebSocketIssue("warn", "[BackendAPI] closed", {});

    expect(consoleWarn).toHaveBeenCalledWith("[BackendAPI] closed");
  });
});
