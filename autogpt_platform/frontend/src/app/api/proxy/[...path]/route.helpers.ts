const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

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

  return false;
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
