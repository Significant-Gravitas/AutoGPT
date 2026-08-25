import { afterEach, describe, expect, it } from "vitest";

import {
  AutoGPTEmbeddedChatElement,
  registerAutoGPTEmbeddedChatElement,
} from "./index";

afterEach(() => {
  document.body.replaceChildren();
});

describe("AutoGPTEmbeddedChatElement", () => {
  it("registers once under the documented tag name", () => {
    registerAutoGPTEmbeddedChatElement();
    registerAutoGPTEmbeddedChatElement();

    expect(customElements.get("autogpt-embedded-chat")).toBe(
      AutoGPTEmbeddedChatElement,
    );
  });

  it("renders an accessible configuration error without a token provider", async () => {
    const element = new AutoGPTEmbeddedChatElement();
    document.body.append(element);

    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(element.querySelector('[role="alert"]')?.textContent).toContain(
      "accessTokenProvider",
    );
  });

  it("accepts a write-only token-provider property", () => {
    const element = new AutoGPTEmbeddedChatElement();
    const provider = async () => "token";

    const descriptor = Object.getOwnPropertyDescriptor(
      AutoGPTEmbeddedChatElement.prototype,
      "accessTokenProvider",
    );

    element.accessTokenProvider = provider;

    expect(descriptor?.set).toEqual(expect.any(Function));
    expect(descriptor?.get).toBeUndefined();
  });

  it("exposes host-provided prompt suggestions", () => {
    const element = new AutoGPTEmbeddedChatElement();
    element.suggestedPrompts = ["Review today's exceptions"];

    expect(element.suggestedPrompts).toEqual(["Review today's exceptions"]);
  });

  it("observes public branding and tenant attributes", () => {
    expect(AutoGPTEmbeddedChatElement.observedAttributes).toEqual([
      "api-base-url",
      "appearance",
      "artifacts-enabled",
      "brand-name",
      "chat-title",
      "sessions-enabled",
      "tenant-key",
    ]);
  });
});
