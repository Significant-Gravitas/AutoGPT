import * as Sentry from "@sentry/nextjs";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import { describeCloseEvent, logWebSocketIssue } from "../websocket-logging";

// Pins the actual mechanism the fix relies on: the Sentry event is created by
// captureConsoleIntegration from our console call, so it has to pick up both
// the summary (as its title) and the scope extras set around that call.
const events: Sentry.ErrorEvent[] = [];

beforeAll(() => {
  // Spy before init so the console patch wraps the spy: Sentry's integration
  // only sees calls made through the console it patched.
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  Sentry.init({
    dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
    integrations: [
      Sentry.captureConsoleIntegration({ levels: ["error", "warn"] }),
    ],
    beforeSend(event) {
      events.push(event);
      return null;
    },
  });
});

afterAll(() => {
  vi.restoreAllMocks();
});

describe("websocket logging reaching Sentry", () => {
  it("sends the close code and reason as the event title, with extras", async () => {
    const { summary, extra } = describeCloseEvent(
      { code: 4002, reason: "Invalid token", wasClean: true },
      "wss://ws.example.com/ws?token=secret-token",
      "connecting",
    );

    logWebSocketIssue(
      "error",
      `[BackendAPI] WebSocket failed to connect: ${summary}`,
      extra,
    );
    await Sentry.flush(2000);

    expect(events).toHaveLength(1);
    const event = events[0];
    expect(event.message).toBe(
      '[BackendAPI] WebSocket failed to connect: code 4002 (invalid token), reason "Invalid token", wasClean true',
    );
    expect(event.extra).toMatchObject({
      ws_close_code: 4002,
      ws_close_code_name: "invalid token",
      ws_close_reason: "Invalid token",
      ws_was_clean: true,
      ws_phase: "connecting",
      ws_url: "wss://ws.example.com/ws",
    });
    expect(JSON.stringify(event)).not.toContain("secret-token");
  });
});
