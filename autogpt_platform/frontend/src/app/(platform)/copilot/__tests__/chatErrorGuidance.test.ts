import { describe, expect, it } from "vitest";

import { describeChatError } from "../chatErrorGuidance";
import { CopilotAuthError } from "../helpers";
import {
  ChatUnavailableError,
  MessageTooLongError,
  SendRefusedError,
} from "../stream/turnRuntime";

const RESEND = /send (it|your message) again|try sending/i;

describe("describeChatError", () => {
  it("a message over the length limit says to shorten it", () => {
    const guidance = describeChatError(
      new MessageTooLongError('{"detail":[…]}', 422),
    );
    expect(guidance).toEqual({
      title: "Your message is too long. Shorten it and send again.",
      action: null,
    });
  });

  it.each([
    ["no session token", new CopilotAuthError(), "sign-in"],
    [
      "a send refused 401",
      new SendRefusedError('{"detail":"Unauthorized"}', 401),
      "sign-in",
    ],
    [
      "the flag-off path's auth failure",
      new Error("Not authenticated"),
      "sign-in",
    ],
    ["a chat that is gone", new ChatUnavailableError(), "new-chat"],
    [
      "a send refused 404",
      new SendRefusedError('{"detail":"Session not found"}', 404),
      "new-chat",
    ],
    [
      "a conversation over the model's limit",
      new Error("SDK stream error: Prompt is too long"),
      "new-chat",
    ],
    [
      "an account without a plan",
      new SendRefusedError(
        '{"detail":"A subscription is required to use this feature."}',
        402,
      ),
      "plans",
    ],
  ])("%s points to what works instead of a resend", (_state, error, action) => {
    const guidance = describeChatError(error);
    expect(guidance.action).toBe(action);
    expect(guidance.title).not.toMatch(RESEND);
  });

  it.each([
    [
      "a usage limit",
      new SendRefusedError('{"detail":"usage limit reached"}', 429),
    ],
    [
      "a plan refusal",
      new SendRefusedError('{"detail":"advanced_tier_not_entitled"}', 403),
    ],
  ])("%s does not promise a resend", (_state, error) => {
    const guidance = describeChatError(error);
    expect(guidance.title).not.toMatch(RESEND);
    expect(guidance.action).toBeNull();
  });

  const REFUSED =
    "The request was refused. Reload the page and try again; if it keeps happening, contact support.";
  const NOTIFIED = "The system maintainer has been notified of the issue";

  it.each([
    [
      "a 403",
      new SendRefusedError('{"detail":"advanced_tier_not_entitled"}', 403),
    ],
    ["a 422", new SendRefusedError('{"detail":"Message too long"}', 422)],
    [
      "a 400 with a non-JSON body",
      new SendRefusedError("Couldn't reach the server (HTTP 400).", 400),
    ],
  ])(
    "%s the app never provokes reads as a refusal we were notified of",
    (_state, error) => {
      expect(describeChatError(error)).toEqual({
        title: REFUSED,
        action: null,
        note: NOTIFIED,
      });
    },
  );

  it.each([
    ["a dropped connection", new TypeError("Failed to fetch")],
    [
      "a proxy's 502",
      new SendRefusedError("Couldn't reach the server (HTTP 502).", 502),
    ],
    ["a 500", new SendRefusedError('{"detail":"Internal error"}', 500)],
    [
      "a conflict with a running turn",
      new SendRefusedError('{"detail":"busy"}', 409),
    ],
    ["an unknown failure", new Error("kaboom")],
  ])("%s says to send again, since that can work", (_state, error) => {
    const guidance = describeChatError(error);
    expect(guidance.title).toMatch(RESEND);
    expect(guidance.action).toBeNull();
  });
});
