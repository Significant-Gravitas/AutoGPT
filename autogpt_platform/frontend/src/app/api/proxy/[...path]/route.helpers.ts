const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SAFE_MEDIA_PATH_COMPONENT_RE = /^(?!\.{1,2}$)[A-Za-z0-9_.-]+$/;
const SINGLE_BYTE_RANGE_RE =
  /^bytes=(?:([0-9]{1,15})-([0-9]{0,15})|-([0-9]{1,15}))$/;
// Private media up to this size is buffered before it is forwarded, below
// Vercel's 4.5 MB limit for buffered bodies, and each video range is capped to
// it so every video response stays buffered.
export const PRIVATE_MEDIA_RANGE_CHUNK_BYTES = 4 * 1024 * 1024;

export function isPrivateStoreMediaRequest(path: string[]): boolean {
  return (
    path.length === 7 &&
    path[0] === "api" &&
    path[1] === "store" &&
    path[2] === "submissions" &&
    path[3] === "media" &&
    SAFE_MEDIA_PATH_COMPONENT_RE.test(path[4]) &&
    (path[5] === "images" || path[5] === "videos") &&
    SAFE_MEDIA_PATH_COMPONENT_RE.test(path[6])
  );
}

export function isPrivateStoreVideoRequest(path: string[]): boolean {
  return isPrivateStoreMediaRequest(path) && path[5] === "videos";
}

export function getSafePrivateMediaRange(value: string | null): string | null {
  if (!value) return null;
  const match = SINGLE_BYTE_RANGE_RE.exec(value.trim());
  if (!match) return null;
  const [, startText, endText, suffixText] = match;
  if (suffixText !== undefined) {
    const suffix = Math.min(
      Number(suffixText),
      PRIVATE_MEDIA_RANGE_CHUNK_BYTES,
    );
    return `bytes=-${suffix}`;
  }
  const start = Number(startText);
  const chunkEnd = start + PRIVATE_MEDIA_RANGE_CHUNK_BYTES - 1;
  const end = endText ? Math.min(Number(endText), chunkEnd) : chunkEnd;
  return `bytes=${start}-${end}`;
}

export function shouldBufferPrivateMedia(headers: Headers): boolean {
  const header = headers.get("content-length");
  if (!header) return false;
  const length = Number(header);
  return (
    Number.isFinite(length) &&
    length >= 0 &&
    length <= PRIVATE_MEDIA_RANGE_CHUNK_BYTES
  );
}

export function isWorkspaceDownloadRequest(path: string[]): boolean {
  // api/workspace/files/{id}/download
  if (
    path.length === 5 &&
    path[0] === "api" &&
    path[1] === "workspace" &&
    path[2] === "files" &&
    UUID_RE.test(path[3]) &&
    path[4] === "download"
  ) {
    return true;
  }

  // api/public/shared/{token}/files/{id}/download
  if (
    path.length === 7 &&
    path[0] === "api" &&
    path[1] === "public" &&
    path[2] === "shared" &&
    UUID_RE.test(path[3]) &&
    path[4] === "files" &&
    UUID_RE.test(path[5]) &&
    path[6] === "download"
  ) {
    return true;
  }

  // api/public/shared/chats/{token}/files/{id}/download
  if (
    path.length === 8 &&
    path[0] === "api" &&
    path[1] === "public" &&
    path[2] === "shared" &&
    path[3] === "chats" &&
    UUID_RE.test(path[4]) &&
    path[5] === "files" &&
    UUID_RE.test(path[6]) &&
    path[7] === "download"
  ) {
    return true;
  }

  return false;
}

export function getSafeDownloadContentDisposition(
  contentDisposition: string | null,
): string {
  if (!contentDisposition) return "attachment";

  const parametersStart = contentDisposition.indexOf(";");
  return parametersStart === -1
    ? "attachment"
    : `attachment${contentDisposition.slice(parametersStart)}`;
}

export function buildSafeWorkspaceDownloadHeaders(
  contentType: string | null,
  contentDisposition: string | null,
  contentLength: number,
): Record<string, string> {
  return {
    "Content-Type": contentType || "application/octet-stream",
    "Content-Length": String(contentLength),
    "Content-Disposition":
      getSafeDownloadContentDisposition(contentDisposition),
    "Content-Security-Policy": "sandbox",
    "X-Content-Type-Options": "nosniff",
  };
}

export function isRedirectStatus(status: number): boolean {
  return [301, 302, 303, 307, 308].includes(status);
}

export function isTransientWorkspaceDownloadStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function fetchWorkspaceDownloadOnce(
  backendUrl: string,
  headers: Record<string, string>,
): Promise<Response> {
  const backendResponse = await fetch(backendUrl, {
    method: "GET",
    headers,
    redirect: "manual",
  });

  if (!isRedirectStatus(backendResponse.status)) {
    return backendResponse;
  }

  const location = backendResponse.headers.get("Location");
  if (!location) return backendResponse;

  return await fetch(location, {
    method: "GET",
    redirect: "follow",
  });
}

export async function fetchWorkspaceDownloadWithRetry(
  backendUrl: string,
  headers: Record<string, string>,
  maxRetries: number,
  retryDelayMs: number,
): Promise<Response> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetchWorkspaceDownloadOnce(backendUrl, headers);
      if (
        response.ok ||
        !isTransientWorkspaceDownloadStatus(response.status) ||
        attempt === maxRetries
      ) {
        return response;
      }
    } catch (error) {
      if (attempt === maxRetries) throw error;
    }

    await sleep(retryDelayMs);
  }

  throw new Error("Workspace download failed after retries");
}

export function getWorkspaceDownloadErrorMessage(body: unknown): string | null {
  if (typeof body === "string") {
    const trimmed = body.trim();
    return trimmed || null;
  }

  if (!body || typeof body !== "object") return null;

  if (
    "detail" in body &&
    typeof body.detail === "string" &&
    body.detail.trim().length > 0
  ) {
    return body.detail.trim();
  }

  if (
    "error" in body &&
    typeof body.error === "string" &&
    body.error.trim().length > 0
  ) {
    return body.error.trim();
  }

  if (
    "detail" in body &&
    body.detail &&
    typeof body.detail === "object" &&
    "message" in body.detail &&
    typeof body.detail.message === "string" &&
    body.detail.message.trim().length > 0
  ) {
    return body.detail.message.trim();
  }

  return null;
}

// A backend that stays silent this long AFTER receiving the full request is
// stalled — legitimate work answers with headers well within this, while
// legitimately slow transfers (big uploads) spend their time in the upload
// phase, which this timeout deliberately excludes.
export const RESPONSE_START_TIMEOUT_MS = 30_000;
export const CODEX_LOGIN_RESPONSE_START_TIMEOUT_MS = 120_000;

export function getResponseStartTimeoutMs(
  path: string[],
  method: string,
): number {
  const isCodexCredentialControl =
    path.length >= 5 &&
    path.slice(0, 4).join("/") === "api/integrations/codex/credentials" &&
    ((method === "DELETE" && path.length === 5) ||
      (method === "GET" &&
        path.length === 6 &&
        (path[5] === "account" || path[5] === "rate-limits")));
  if (
    (method === "GET" && path.join("/") === "api/integrations/codex/login") ||
    isCodexCredentialControl
  ) {
    return CODEX_LOGIN_RESPONSE_START_TIMEOUT_MS;
  }
  return RESPONSE_START_TIMEOUT_MS;
}

export function watchResponseStart(
  requestBody: ReadableStream | null,
  timeoutMs: number = RESPONSE_START_TIMEOUT_MS,
) {
  const abort = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;

  function arm() {
    timer = setTimeout(
      () =>
        abort.abort(
          new DOMException(
            "Backend sent no response within " +
              `${timeoutMs}ms of receiving the request`,
            "TimeoutError",
          ),
        ),
      timeoutMs,
    );
  }

  // With a body, the clock starts only once the upload has been fully read
  // (TransformStream flush); without one there is no upload phase, so it
  // starts immediately.
  let body: ReadableStream | undefined;
  if (requestBody) {
    body = requestBody.pipeThrough(new TransformStream({ flush: arm }));
  } else {
    arm();
  }

  return {
    body,
    signal: abort.signal,
    // Call when the backend starts responding (or errors): from that point
    // only the overall fetch ceiling applies.
    clear: () => clearTimeout(timer),
  };
}
