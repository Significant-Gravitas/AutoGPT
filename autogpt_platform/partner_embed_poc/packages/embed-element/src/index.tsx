import { AutoGPTEmbeddedChat } from "@autogpt/embedded-chat";
import "@autogpt/embedded-chat/styles.css";
import { createRoot, type Root } from "react-dom/client";

export type AccessTokenProvider = () => Promise<string>;
export type SuggestedPrompts = string[];

const DEFAULT_TAG_NAME = "autogpt-embedded-chat";

export class AutoGPTEmbeddedChatElement extends HTMLElement {
  static readonly observedAttributes = [
    "api-base-url",
    "appearance",
    "artifacts-enabled",
    "brand-name",
    "chat-title",
    "sessions-enabled",
    "tenant-key",
  ];

  private reactRoot: Root | undefined;
  private tokenProvider: AccessTokenProvider | undefined;
  private promptSuggestions: SuggestedPrompts = [];

  set accessTokenProvider(value: AccessTokenProvider | undefined) {
    this.tokenProvider = value;
    this.renderChat();
  }

  get suggestedPrompts(): SuggestedPrompts {
    return this.promptSuggestions;
  }

  set suggestedPrompts(value: SuggestedPrompts) {
    this.promptSuggestions = Array.isArray(value) ? value : [];
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
        appearance={
          this.getAttribute("appearance") === "dark" ? "dark" : "light"
        }
        artifactsEnabled={this.getAttribute("artifacts-enabled") !== "false"}
        brandName={this.getAttribute("brand-name") ?? "AI assistant"}
        getAccessToken={this.tokenProvider}
        onNavigate={(href) =>
          this.dispatchEvent(
            new CustomEvent("autogpt-navigate", {
              bubbles: true,
              composed: true,
              detail: { href },
            }),
          )
        }
        suggestedPrompts={this.promptSuggestions}
        title={this.getAttribute("chat-title") ?? "Assistant"}
        sessionsEnabled={this.getAttribute("sessions-enabled") !== "false"}
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
