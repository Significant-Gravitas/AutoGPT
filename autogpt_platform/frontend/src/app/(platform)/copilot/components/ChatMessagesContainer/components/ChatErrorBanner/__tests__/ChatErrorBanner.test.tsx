import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it } from "vitest";

import { CopilotAuthError } from "../../../../../helpers";
import {
  ChatUnavailableError,
  SendRefusedError,
} from "../../../../../stream/turnRuntime";
import { ChatErrorBanner } from "../ChatErrorBanner";

afterEach(() => {
  window.history.replaceState(null, "", "/");
});

describe("ChatErrorBanner", () => {
  it("signed out: offers sign-in that returns to this chat", () => {
    window.history.replaceState(null, "", "/copilot?sessionId=s-1");
    render(<ChatErrorBanner error={new CopilotAuthError()} />);

    expect(
      screen.getByText("You're signed out. Sign in again to keep chatting."),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Sign in" }).getAttribute("href"),
    ).toBe(`/login?next=${encodeURIComponent("/copilot?sessionId=s-1")}`);
  });

  it("a chat that is gone: offers a new chat", () => {
    render(<ChatErrorBanner error={new ChatUnavailableError()} />);

    expect(
      screen.getByText(
        "This chat no longer exists. Start a new chat to keep going.",
      ),
    ).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Start a new chat" })
        .getAttribute("href"),
    ).toBe("/copilot");
  });

  it("a refusal the app never provokes: the refusal, the server's reason, and the notified line", () => {
    render(
      <ChatErrorBanner
        error={
          new SendRefusedError(
            '{"detail":"advanced_tier_not_entitled"}',
            403,
            "advanced_tier_not_entitled",
          )
        }
      />,
    );

    const lines = Array.from(
      screen.getByTestId("chat-error-banner").querySelectorAll("p"),
    ).map((p) => p.textContent);
    expect(lines).toEqual([
      "The request was refused. Reload the page and try again; if it keeps happening, contact support.",
      "advanced_tier_not_entitled",
      "The system maintainer has been notified of the issue",
    ]);
  });

  it("a refusal with no reason leaves the reason line out", () => {
    render(
      <ChatErrorBanner
        error={
          new SendRefusedError("Couldn't reach the server (HTTP 400).", 400)
        }
      />,
    );

    const lines = Array.from(
      screen.getByTestId("chat-error-banner").querySelectorAll("p"),
    ).map((p) => p.textContent);
    expect(lines).toEqual([
      "The request was refused. Reload the page and try again; if it keeps happening, contact support.",
      "The system maintainer has been notified of the issue",
    ]);
  });

  it("a dropped connection: says to send again, with no action", () => {
    render(<ChatErrorBanner error={new TypeError("Failed to fetch")} />);

    expect(
      screen.getByText(
        "The connection dropped before your message was sent. Check your connection and send it again.",
      ),
    ).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
  });
});
