import { normalizeSameOriginApiBaseURL, type AccessTokenProvider } from "./api";

export interface EmbedSessionSummary {
  id: string;
  title: string | null;
  createdAt: string;
  updatedAt: string;
  chatStatus: string;
}

export interface EmbedRawMessage {
  id?: string | null;
  role: string;
  content?: string | null;
  name?: string | null;
  tool_call_id?: string | null;
  tool_calls?: unknown[] | null;
  sequence?: number | null;
}

export interface EmbedSessionDetail extends EmbedSessionSummary {
  messages: EmbedRawMessage[];
  hasMoreMessages: boolean;
  oldestSequence: number | null;
  capabilities: string[];
}

export interface EmbedArtifact {
  id: string;
  name: string;
  path: string;
  mimeType: string;
  sizeBytes: number;
  createdAt: string;
}

export async function listEmbedSessions(
  apiBaseURL: string,
  getAccessToken: AccessTokenProvider,
): Promise<EmbedSessionSummary[]> {
  const response = await authorizedFetch(
    apiBaseURL,
    "/api/embed/v1/sessions",
    getAccessToken,
  );
  const body = (await response.json()) as {
    sessions?: Array<Record<string, unknown>>;
  };
  return (body.sessions ?? []).map(toSessionSummary);
}

export async function getEmbedSession(
  apiBaseURL: string,
  sessionID: string,
  getAccessToken: AccessTokenProvider,
): Promise<EmbedSessionDetail> {
  const response = await authorizedFetch(
    apiBaseURL,
    `/api/embed/v1/sessions/${encodeURIComponent(sessionID)}`,
    getAccessToken,
  );
  const body = (await response.json()) as Record<string, unknown>;
  return {
    ...toSessionSummary(body),
    messages: Array.isArray(body.messages)
      ? (body.messages as EmbedRawMessage[])
      : [],
    hasMoreMessages: body.has_more_messages === true,
    oldestSequence:
      typeof body.oldest_sequence === "number" ? body.oldest_sequence : null,
    capabilities: Array.isArray(body.capabilities)
      ? body.capabilities.filter(
          (capability): capability is string => typeof capability === "string",
        )
      : [],
  };
}

export async function listEmbedArtifacts(
  apiBaseURL: string,
  sessionID: string,
  getAccessToken: AccessTokenProvider,
): Promise<EmbedArtifact[]> {
  const response = await authorizedFetch(
    apiBaseURL,
    `/api/embed/v1/sessions/${encodeURIComponent(sessionID)}/artifacts`,
    getAccessToken,
  );
  const body = (await response.json()) as {
    artifacts?: Array<Record<string, unknown>>;
  };
  return (body.artifacts ?? []).map((artifact) => ({
    id: stringValue(artifact.id),
    name: stringValue(artifact.name),
    path: stringValue(artifact.path),
    mimeType: stringValue(artifact.mime_type),
    sizeBytes:
      typeof artifact.size_bytes === "number" ? artifact.size_bytes : 0,
    createdAt: stringValue(artifact.created_at),
  }));
}

export async function downloadEmbedArtifact(
  apiBaseURL: string,
  sessionID: string,
  artifact: EmbedArtifact,
  getAccessToken: AccessTokenProvider,
): Promise<void> {
  const response = await authorizedFetch(
    apiBaseURL,
    `/api/embed/v1/sessions/${encodeURIComponent(sessionID)}/artifacts/${encodeURIComponent(artifact.id)}/download`,
    getAccessToken,
  );
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = artifact.name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function authorizedFetch(
  apiBaseURL: string,
  path: string,
  getAccessToken: AccessTokenProvider,
): Promise<Response> {
  const baseURL = normalizeSameOriginApiBaseURL(apiBaseURL);
  const token = await getAccessToken();
  const response = await fetch(baseURL + path, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(`Embedded API request failed (${response.status})`);
  }
  return response;
}

function toSessionSummary(value: Record<string, unknown>): EmbedSessionSummary {
  return {
    id: stringValue(value.id),
    title: typeof value.title === "string" ? value.title : null,
    createdAt: stringValue(value.created_at),
    updatedAt: stringValue(value.updated_at),
    chatStatus: stringValue(value.chat_status, "idle"),
  };
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}
