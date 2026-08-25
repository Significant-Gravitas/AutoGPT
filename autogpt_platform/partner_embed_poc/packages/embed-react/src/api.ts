export interface EmbedSession {
  id: string;
  createdAt: string;
}

export type AccessTokenProvider = () => Promise<string>;

export function normalizeSameOriginApiBaseURL(apiBaseURL: string): string {
  const normalized = apiBaseURL.trim().replace(/\/+$/, "");
  if (!normalized) return "";

  const location = globalThis.location;
  if (!location || location.origin === "null") {
    if (normalized.startsWith("/") && !normalized.startsWith("//")) {
      return normalized;
    }
    throw new Error("Embedded chat API must use the host page origin");
  }

  const resolved = new URL(normalized, location.href);
  if (resolved.origin !== location.origin) {
    throw new Error("Embedded chat API must use the host page origin");
  }
  return normalized.startsWith("/") ? normalized : resolved.href;
}

export async function createEmbedSession(
  apiBaseURL: string,
  getAccessToken: AccessTokenProvider,
): Promise<EmbedSession> {
  const baseURL = normalizeSameOriginApiBaseURL(apiBaseURL);
  const token = await getAccessToken();
  const response = await fetch(`${baseURL}/api/embed/v1/sessions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(`Unable to create embedded chat (${response.status})`);
  }
  const body: unknown = await response.json();
  if (!isSessionResponse(body)) {
    throw new Error("Embedded chat returned an invalid session");
  }
  return { id: body.id, createdAt: body.created_at };
}

function isSessionResponse(
  value: unknown,
): value is { id: string; created_at: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    typeof value.id === "string" &&
    "created_at" in value &&
    typeof value.created_at === "string"
  );
}
