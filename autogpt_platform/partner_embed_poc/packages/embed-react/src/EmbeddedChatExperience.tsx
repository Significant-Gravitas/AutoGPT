import type { UIMessage } from "ai";
import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { createEmbedSession } from "./api";
import {
  downloadEmbedArtifact,
  getEmbedSession,
  listEmbedArtifacts,
  listEmbedSessions,
  type EmbedArtifact,
  type EmbedSessionDetail,
  type EmbedSessionSummary,
} from "./session-api";
import { persistedMessagesToUI } from "./session-messages";
import { useSessionChat } from "./useSessionChat";

import "./embedded-chat.css";

export interface AutoGPTEmbeddedChatTheme {
  background?: string;
  foreground?: string;
  surface?: string;
  surfaceMuted?: string;
  accent?: string;
  accentForeground?: string;
  border?: string;
  danger?: string;
  radius?: string;
  fontFamily?: string;
  shadow?: string;
}

export interface AutoGPTEmbeddedChatProps {
  apiBaseURL: string;
  getAccessToken: () => Promise<string>;
  brandName?: string;
  title?: string;
  theme?: AutoGPTEmbeddedChatTheme;
  sessionsEnabled?: boolean;
  artifactsEnabled?: boolean;
  appearance?: "light" | "dark";
}

export function AutoGPTEmbeddedChat({
  apiBaseURL,
  getAccessToken,
  brandName = "AI assistant",
  title = "Assistant",
  theme,
  sessionsEnabled = true,
  artifactsEnabled = true,
  appearance = "light",
}: AutoGPTEmbeddedChatProps) {
  const [sessions, setSessions] = useState<EmbedSessionSummary[]>([]);
  const [selectedID, setSelectedID] = useState<string | null>(null);
  const [detail, setDetail] = useState<EmbedSessionDetail | null>(null);
  const [artifacts, setArtifacts] = useState<EmbedArtifact[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const [artifactsOpen, setArtifactsOpen] = useState(false);
  const [shellError, setShellError] = useState<string | null>(null);
  const tokenProviderRef = useRef(getAccessToken);
  tokenProviderRef.current = getAccessToken;

  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      setIsLoading(true);
      setShellError(null);
      try {
        const existing = sessionsEnabled
          ? await listEmbedSessions(apiBaseURL, () =>
              tokenProviderRef.current(),
            )
          : [];
        if (cancelled) return;
        setSessions(existing);
        if (existing[0]) {
          setSelectedID(existing[0].id);
        } else {
          const created = await createEmbedSession(apiBaseURL, () =>
            tokenProviderRef.current(),
          );
          if (cancelled) return;
          const summary = newSessionSummary(created.id, created.createdAt);
          setSessions([summary]);
          setSelectedID(created.id);
        }
      } catch (error) {
        if (!cancelled) setShellError(errorMessage(error));
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void initialize();
    return () => {
      cancelled = true;
    };
  }, [apiBaseURL, sessionsEnabled]);

  useEffect(() => {
    if (!selectedID) return;
    let cancelled = false;
    setDetail(null);
    setArtifacts([]);
    setIsLoading(true);
    setShellError(null);

    async function load() {
      try {
        const [nextDetail, nextArtifacts] = await Promise.all([
          getEmbedSession(apiBaseURL, selectedID as string, () =>
            tokenProviderRef.current(),
          ),
          artifactsEnabled
            ? listEmbedArtifacts(apiBaseURL, selectedID as string, () =>
                tokenProviderRef.current(),
              ).catch(() => [])
            : Promise.resolve([]),
        ]);
        if (cancelled) return;
        setDetail(nextDetail);
        setArtifacts(nextArtifacts);
      } catch (error) {
        if (!cancelled) setShellError(errorMessage(error));
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [apiBaseURL, artifactsEnabled, selectedID]);

  async function createNewSession() {
    setShellError(null);
    try {
      const created = await createEmbedSession(apiBaseURL, () =>
        tokenProviderRef.current(),
      );
      const summary = newSessionSummary(created.id, created.createdAt);
      setSessions((current) => [summary, ...current]);
      setSelectedID(created.id);
      setNavigationOpen(false);
    } catch (error) {
      setShellError(errorMessage(error));
    }
  }

  async function refreshAfterTurn() {
    if (!selectedID) return;
    try {
      const [nextSessions, nextArtifacts] = await Promise.all([
        sessionsEnabled
          ? listEmbedSessions(apiBaseURL, () => tokenProviderRef.current())
          : Promise.resolve(sessions),
        artifactsEnabled && detail?.capabilities.includes("documents.read")
          ? listEmbedArtifacts(apiBaseURL, selectedID, () =>
              tokenProviderRef.current(),
            ).catch(() => artifacts)
          : Promise.resolve(artifacts),
      ]);
      setSessions(nextSessions);
      setArtifacts(nextArtifacts);
    } catch (error) {
      setShellError(errorMessage(error));
    }
  }

  const blockCapabilities =
    detail?.capabilities.filter((capability) =>
      capability.startsWith("autogpt:block:"),
    ) ?? [];
  const canShowArtifacts =
    artifactsEnabled && detail?.capabilities.includes("documents.read");

  return (
    <section
      className="agpt-embed"
      data-slot="embedded-chat"
      data-appearance={appearance}
      style={themeStyle(theme)}
      aria-label={title}
    >
      <header className="agpt-embed__header" data-slot="chat-header">
        <button
          type="button"
          className="agpt-embed__icon-button"
          aria-label="Open chat sessions"
          aria-expanded={navigationOpen}
          onClick={() => setNavigationOpen((open) => !open)}
          hidden={!sessionsEnabled}
        >
          ☰
        </button>
        <div className="agpt-embed__identity">
          <p className="agpt-embed__eyebrow">{brandName}</p>
          <h2>{title}</h2>
        </div>
        <div className="agpt-embed__header-actions">
          {canShowArtifacts ? (
            <ArtifactPanel
              artifacts={artifacts}
              isOpen={artifactsOpen}
              onToggle={() => setArtifactsOpen((open) => !open)}
              onDownload={(artifact) => {
                if (!selectedID) return;
                void downloadEmbedArtifact(
                  apiBaseURL,
                  selectedID,
                  artifact,
                  () => tokenProviderRef.current(),
                ).catch((error) => setShellError(errorMessage(error)));
              }}
            />
          ) : null}
          <span
            className="agpt-embed__status"
            data-state={isLoading ? "loading" : "ready"}
          >
            {isLoading ? "Connecting" : "Ready"}
          </span>
        </div>
      </header>

      <div className="agpt-embed__body">
        {sessionsEnabled ? (
          <SessionNavigation
            sessions={sessions}
            selectedID={selectedID}
            isOpen={navigationOpen}
            blockCount={blockCapabilities.length}
            onClose={() => setNavigationOpen(false)}
            onCreate={() => void createNewSession()}
            onSelect={(sessionID) => {
              setSelectedID(sessionID);
              setNavigationOpen(false);
            }}
          />
        ) : null}

        <main className="agpt-embed__main">
          {shellError ? (
            <p className="agpt-embed__error" role="alert">
              {shellError}
            </p>
          ) : null}
          {detail ? (
            <SessionChat
              key={detail.id}
              apiBaseURL={apiBaseURL}
              detail={detail}
              getAccessToken={() => tokenProviderRef.current()}
              title={title}
              onFinish={() => void refreshAfterTurn()}
            />
          ) : (
            <LoadingState />
          )}
        </main>
      </div>
    </section>
  );
}

interface SessionChatProps {
  apiBaseURL: string;
  detail: EmbedSessionDetail;
  getAccessToken: () => Promise<string>;
  title: string;
  onFinish: () => void;
}

function SessionChat({
  apiBaseURL,
  detail,
  getAccessToken,
  title,
  onFinish,
}: SessionChatProps) {
  const composerID = useId();
  const endRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState("");
  const { messages, sendMessage, stop, status, error } = useSessionChat({
    apiBaseURL,
    sessionID: detail.id,
    initialMessages: persistedMessagesToUI(detail.messages),
    getAccessToken,
    onFinish,
  });
  const isBusy = status === "submitted" || status === "streaming";
  const messageSegments = visibleMessageSegments(messages);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, status]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    sendDraft();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendDraft();
    }
  }

  function sendDraft() {
    const message = draft.trim();
    if (!message || isBusy) return;
    setDraft("");
    void sendMessage({ text: message });
  }

  return (
    <>
      <div
        className="agpt-embed__messages"
        data-slot="conversation"
        role="log"
        aria-live="polite"
        aria-busy={isBusy}
      >
        {messageSegments.length === 0 ? (
          <EmptyState />
        ) : (
          messageSegments.map(({ message, parts, showLabel }) => (
            <NativeMessage
              key={message.id}
              message={message}
              parts={parts}
              showLabel={showLabel}
            />
          ))
        )}
        {isBusy ? (
          <div className="agpt-embed__working" data-state={status}>
            <span aria-hidden="true" />
            AutoPilot is working
          </div>
        ) : null}
        <div ref={endRef} />
      </div>

      {error ? (
        <p className="agpt-embed__error" role="alert">
          {error.message}
        </p>
      ) : null}

      <form
        className="agpt-embed__composer"
        data-slot="composer"
        onSubmit={handleSubmit}
      >
        <label htmlFor={composerID}>Message {title}</label>
        <textarea
          id={composerID}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask AutoPilot about jobs, reports, documents, or exceptions…"
          rows={2}
        />
        <div className="agpt-embed__actions">
          {isBusy ? (
            <button type="button" data-variant="secondary" onClick={stop}>
              Stop
            </button>
          ) : null}
          <button type="submit" disabled={!draft.trim() || isBusy}>
            Send
          </button>
        </div>
      </form>
    </>
  );
}

interface NativeMessageProps {
  message: UIMessage;
  parts: UIMessage["parts"];
  showLabel: boolean;
}

function NativeMessage({ message, parts, showLabel }: NativeMessageProps) {
  return (
    <article
      className="agpt-embed__message"
      data-slot="message"
      data-role={message.role}
    >
      {showLabel ? (
        <span className="agpt-embed__message-label">
          {message.role === "user" ? "You" : "AutoPilot"}
        </span>
      ) : null}
      <div className="agpt-embed__message-content">
        {parts.map((part, index) => (
          <MessagePart key={partKey(part, index)} part={part} />
        ))}
      </div>
    </article>
  );
}

function visibleMessageSegments(messages: UIMessage[]) {
  const segments: {
    message: UIMessage;
    parts: UIMessage["parts"];
    showLabel: boolean;
  }[] = [];
  let previousRole: UIMessage["role"] | null = null;

  for (const message of messages) {
    const parts = message.parts.filter(isRenderablePart);
    if (parts.length === 0) continue;
    segments.push({
      message,
      parts,
      showLabel: message.role !== previousRole,
    });
    previousRole = message.role;
  }

  return segments;
}

function isRenderablePart(part: UIMessage["parts"][number]): boolean {
  if (part.type === "text" || part.type === "reasoning") {
    return part.text.trim().length > 0;
  }
  return part.type === "dynamic-tool" || part.type.startsWith("tool-");
}

function MessagePart({ part }: { part: UIMessage["parts"][number] }) {
  if (part.type === "text") {
    return (
      <div className="agpt-embed__prose" data-slot="message-markdown">
        <ReactMarkdown disallowedElements={["img"]} remarkPlugins={[remarkGfm]}>
          {part.text}
        </ReactMarkdown>
      </div>
    );
  }
  if (part.type === "reasoning") {
    return (
      <details className="agpt-embed__reasoning" data-slot="reasoning">
        <summary>Reasoning</summary>
        <p>{part.text}</p>
      </details>
    );
  }

  const value = part as unknown as Record<string, unknown>;
  if (part.type === "dynamic-tool" || part.type.startsWith("tool-")) {
    const toolName =
      typeof value.toolName === "string"
        ? value.toolName
        : part.type.replace(/^tool-/, "");
    return (
      <details
        className="agpt-embed__tool"
        data-slot="tool-call"
        data-state={typeof value.state === "string" ? value.state : "pending"}
      >
        <summary>
          <span>{humanize(toolName)}</span>
          <span>{toolState(value.state)}</span>
        </summary>
        <div>
          {"input" in value ? (
            <pre aria-label="Tool input">{prettyJSON(value.input)}</pre>
          ) : null}
          {"output" in value ? (
            <pre aria-label="Tool output">{prettyJSON(value.output)}</pre>
          ) : null}
          {typeof value.errorText === "string" ? (
            <p className="agpt-embed__tool-error">{value.errorText}</p>
          ) : null}
        </div>
      </details>
    );
  }
  return null;
}

interface SessionNavigationProps {
  sessions: EmbedSessionSummary[];
  selectedID: string | null;
  isOpen: boolean;
  blockCount: number;
  onClose: () => void;
  onCreate: () => void;
  onSelect: (sessionID: string) => void;
}

function SessionNavigation({
  sessions,
  selectedID,
  isOpen,
  blockCount,
  onClose,
  onCreate,
  onSelect,
}: SessionNavigationProps) {
  return (
    <nav
      className="agpt-embed__sessions"
      data-slot="session-navigation"
      data-state={isOpen ? "open" : "closed"}
      aria-label="Chat sessions"
    >
      <div className="agpt-embed__panel-header">
        <strong>Chats</strong>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close chat sessions"
        >
          ×
        </button>
      </div>
      <button type="button" className="agpt-embed__new-chat" onClick={onCreate}>
        + New chat
      </button>
      <div className="agpt-embed__session-list">
        {sessions.map((session) => (
          <button
            key={session.id}
            type="button"
            data-slot="session-item"
            data-state={session.id === selectedID ? "active" : "idle"}
            aria-current={session.id === selectedID ? "page" : undefined}
            onClick={() => onSelect(session.id)}
          >
            <span>{session.title || "New conversation"}</span>
            <time dateTime={session.updatedAt}>
              {relativeDate(session.updatedAt)}
            </time>
          </button>
        ))}
      </div>
      <p className="agpt-embed__capabilities">
        {blockCount} AutoGPT {blockCount === 1 ? "block" : "blocks"} enabled
      </p>
    </nav>
  );
}

interface ArtifactPanelProps {
  artifacts: EmbedArtifact[];
  isOpen: boolean;
  onToggle: () => void;
  onDownload: (artifact: EmbedArtifact) => void;
}

function ArtifactPanel({
  artifacts,
  isOpen,
  onToggle,
  onDownload,
}: ArtifactPanelProps) {
  return (
    <aside
      className="agpt-embed__artifacts"
      data-slot="artifact-list"
      data-state={isOpen ? "open" : "closed"}
      aria-label="Session artifacts"
    >
      <button
        className="agpt-embed__artifact-toggle"
        type="button"
        aria-expanded={isOpen}
        aria-label={`Artifacts (${artifacts.length})`}
        onClick={onToggle}
      >
        Artifacts <span>{artifacts.length}</span>
      </button>
      <div className="agpt-embed__artifact-content">
        {artifacts.length === 0 ? (
          <p>Files created by AutoPilot in this chat will appear here.</p>
        ) : (
          artifacts.map((artifact) => (
            <article key={artifact.id} data-slot="artifact-card">
              <div>
                <strong>{artifact.name}</strong>
                <span>
                  {artifact.mimeType || "File"} ·{" "}
                  {formatBytes(artifact.sizeBytes)}
                </span>
              </div>
              <button type="button" onClick={() => onDownload(artifact)}>
                Download
              </button>
            </article>
          ))
        )}
      </div>
    </aside>
  );
}

function EmptyState() {
  return (
    <div className="agpt-embed__empty">
      <strong>What would you like AutoPilot to do?</strong>
      <p>
        Review arrivals, investigate exceptions, run an enabled block, or create
        a document.
      </p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="agpt-embed__empty" role="status">
      <strong>Loading your secure workspace…</strong>
      <p>Restoring conversations and session artifacts.</p>
    </div>
  );
}

function themeStyle(theme?: AutoGPTEmbeddedChatTheme): CSSProperties {
  if (!theme) return {};
  const values: Record<string, string | undefined> = {
    "--agpt-embed-background": theme.background,
    "--agpt-embed-foreground": theme.foreground,
    "--agpt-embed-surface": theme.surface,
    "--agpt-embed-surface-muted": theme.surfaceMuted,
    "--agpt-embed-accent": theme.accent,
    "--agpt-embed-accent-foreground": theme.accentForeground,
    "--agpt-embed-border": theme.border,
    "--agpt-embed-danger": theme.danger,
    "--agpt-embed-radius": theme.radius,
    "--agpt-embed-font": theme.fontFamily,
    "--agpt-embed-shadow": theme.shadow,
  };
  return Object.fromEntries(
    Object.entries(values).filter((entry): entry is [string, string] =>
      Boolean(entry[1]),
    ),
  ) as CSSProperties;
}

function newSessionSummary(id: string, createdAt: string): EmbedSessionSummary {
  return {
    id,
    title: null,
    createdAt,
    updatedAt: createdAt,
    chatStatus: "idle",
  };
}

function partKey(part: UIMessage["parts"][number], index: number): string {
  const value = part as unknown as Record<string, unknown>;
  return typeof value.toolCallId === "string"
    ? value.toolCallId
    : `${part.type}-${index}`;
}

function humanize(value: string): string {
  return value.replace(/^query_/, "").replaceAll("_", " ");
}

function toolState(value: unknown): string {
  if (value === "output-available") return "Complete";
  if (value === "output-error") return "Failed";
  if (value === "input-streaming") return "Preparing";
  return "Running";
}

function prettyJSON(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function relativeDate(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return "";
  const minutes = Math.max(0, Math.round((Date.now() - timestamp) / 60_000));
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.round(hours / 24)}d`;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Unable to load embedded chat";
}
