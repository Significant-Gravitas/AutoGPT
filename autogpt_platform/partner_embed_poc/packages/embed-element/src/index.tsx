import { AutoGPTEmbeddedChat } from "@autogpt/embedded-chat";
import "@autogpt/embedded-chat/styles.css";
import { createRoot, type Root } from "react-dom/client";

export type AccessTokenProvider = () => Promise<string>;

const DEFAULT_TAG_NAME = "autogpt-embedded-chat";

export class AutoGPTEmbeddedChatElement extends HTMLElement {
  static readonly observedAttributes = [
    "api-base-url",
    "brand-name",
    "chat-title",
    "tenant-key",
  ];

  private reactRoot: Root | undefined;
  private tokenProvider: AccessTokenProvider | undefined;

  get accessTokenProvider(): AccessTokenProvider | undefined {
    return this.tokenProvider;
  }

  set accessTokenProvider(value: AccessTokenProvider | undefined) {
    this.tokenProvider = value;
    this.renderChat();
  }

  connectedCallback() {
    if (!this.reactRoot) this.reactRoot = createRoot(this);
    this.renderChat();
  }

  disconnectedCallback() {
    this.reactRoot?.unmount();
    this.reactRoot = undefined;
  }

  attributeChangedCallback() {
    this.renderChat();
  }

  private renderChat() {
    if (!this.reactRoot) return;
    if (!this.tokenProvider) {
      this.reactRoot.render(
        <p className="agpt-embed__configuration-error" role="alert">
          Embedded chat requires an accessTokenProvider callback.
        </p>,
      );
      return;
    }
    this.reactRoot.render(
      <AutoGPTEmbeddedChat
        key={this.getAttribute("tenant-key") ?? "default"}
        apiBaseURL={this.getAttribute("api-base-url") ?? ""}
        brandName={this.getAttribute("brand-name") ?? "AI assistant"}
        getAccessToken={this.tokenProvider}
        title={this.getAttribute("chat-title") ?? "Assistant"}
      />,
    );
  }
}

export function registerAutoGPTEmbeddedChatElement(
  tagName = DEFAULT_TAG_NAME,
): void {
  if (!customElements.get(tagName)) {
    customElements.define(tagName, AutoGPTEmbeddedChatElement);
  }
}

registerAutoGPTEmbeddedChatElement();

declare global {
  interface HTMLElementTagNameMap {
    "autogpt-embedded-chat": AutoGPTEmbeddedChatElement;
  }
}
