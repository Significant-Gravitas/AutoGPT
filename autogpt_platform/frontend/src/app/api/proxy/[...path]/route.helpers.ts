export function isWorkspaceDownloadRequest(path: string[]): boolean {
  return (
    path.length == 5 &&
    path[0] === "api" &&
    path[1] === "workspace" &&
    path[2] === "files" &&
    path[path.length - 1] === "download"
  );
}

export function isRedirectStatus(status: number): boolean {
  return [301, 302, 303, 307, 308].includes(status);
}

export function isTransientWorkspaceDownloadStatus(status: number): boolean {
  return status === 408 || status === 429 || status >= 500;
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
