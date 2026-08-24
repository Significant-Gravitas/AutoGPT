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

  it("exposes a typed token-provider property", () => {
    const element = new AutoGPTEmbeddedChatElement();
    const provider = async () => "token";

    element.accessTokenProvider = provider;

    expect(element.accessTokenProvider).toBe(provider);
  });

  it("observes public branding and tenant attributes", () => {
    expect(AutoGPTEmbeddedChatElement.observedAttributes).toEqual([
      "api-base-url",
      "brand-name",
      "chat-title",
      "tenant-key",
    ]);
  });
});
