import type { UIMessage } from "ai";
import { useId, useState, type FormEvent } from "react";

import "./embedded-chat.css";
import { textFromMessage } from "./transport";
import { useEmbeddedChat } from "./useEmbeddedChat";

export interface AutoGPTEmbeddedChatProps {
  apiBaseURL: string;
  getAccessToken: () => Promise<string>;
  brandName?: string;
  title?: string;
}

export function AutoGPTEmbeddedChat({
  apiBaseURL,
  getAccessToken,
  brandName = "AI assistant",
  title = "Assistant",
}: AutoGPTEmbeddedChatProps) {
  const composerID = useId();
  const [draft, setDraft] = useState("");
  const {
    messages,
    sendMessage,
    stop,
    status,
    error,
    initializationError,
    isInitialized,
  } = useEmbeddedChat({ apiBaseURL, getAccessToken });
  const isBusy = status === "submitted" || status === "streaming";

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || !isInitialized || isBusy) return;
    setDraft("");
    void sendMessage({ text: message });
  }

  return (
    <section className="agpt-embed" aria-label={title}>
      <header className="agpt-embed__header">
        <div>
          <p className="agpt-embed__eyebrow">{brandName}</p>
          <h2>{title}</h2>
        </div>
        <span className="agpt-embed__status">
          {isBusy ? "Working" : isInitialized ? "Ready" : "Connecting"}
        </span>
      </header>

      <div className="agpt-embed__messages" aria-live="polite">
        {messages.length === 0 ? (
          <EmptyState isInitialized={isInitialized} />
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))
        )}
      </div>

      {(initializationError || error) && (
        <p className="agpt-embed__error" role="alert">
          {initializationError || error?.message}
        </p>
      )}

      <form className="agpt-embed__composer" onSubmit={handleSubmit}>
        <label htmlFor={composerID}>Message {title}</label>
        <textarea
          id={composerID}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Ask about jobs, reports, documents, or exceptions…"
          rows={3}
          disabled={!isInitialized}
        />
        <div className="agpt-embed__actions">
          {isBusy ? (
            <button
              type="button"
              className="agpt-embed__secondary"
              onClick={stop}
            >
              Stop
            </button>
          ) : null}
          <button
            type="submit"
            disabled={!draft.trim() || !isInitialized || isBusy}
          >
            Send
          </button>
        </div>
      </form>
    </section>
  );
}

function EmptyState({ isInitialized }: { isInitialized: boolean }) {
  return (
    <div className="agpt-embed__empty">
      <strong>
        {isInitialized ? "What would you like to do?" : "Connecting securely…"}
      </strong>
      <p>
        Review arrivals, create reports, or investigate operational exceptions.
      </p>
    </div>
  );
}

function MessageBubble({ message }: { message: UIMessage }) {
  const text = textFromMessage(message);
  if (!text) return null;
  return (
    <article
      className={`agpt-embed__message agpt-embed__message--${message.role}`}
    >
      <span>{message.role === "user" ? "You" : "Assistant"}</span>
      <p>{text}</p>
    </article>
  );
}
