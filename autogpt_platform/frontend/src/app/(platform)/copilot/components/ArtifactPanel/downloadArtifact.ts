import type { ArtifactRef } from "../../store";

const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 500;

function isTransientError(status: number): boolean {
  return status >= 500 || status === 408 || status === 429;
}

class DownloadError extends Error {}

async function fetchWithRetry(url: string, retries: number): Promise<Response> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url);
      if (res.ok) return res;
      if (!isTransientError(res.status) || attempt === retries) {
        throw new DownloadError(`Download failed: ${res.status}`);
      }
    } catch (error) {
      if (error instanceof DownloadError) throw error;
      if (attempt === retries) throw error;
    }
    await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
  }
  throw new Error("Unreachable");
}

/**
 * Trigger a file download from an artifact URL.
 *
 * Uses fetch+blob instead of a bare `<a download>` because the browser
 * ignores the `download` attribute on cross-origin responses (GCS signed
 * URLs), and some browsers require the anchor to be attached to the DOM
 * before `.click()` fires the download.
 *
 * Retries up to {@link MAX_RETRIES} times on transient server errors (5xx,
 * 408, 429) to handle intermittent proxy/GCS failures.
 */
export function downloadArtifact(artifact: ArtifactRef): Promise<void> {
  // Replace path separators, Windows-reserved chars, control chars, and
  // parent-dir sequences so the browser-assigned filename is safe to write
  // anywhere on the user's filesystem.
  const collapsedDots = artifact.title.replace(/\.\./g, "");
  const hasVisibleName = collapsedDots.replace(/^\.+/, "").length > 0;
  const safeName = artifact.title
    .replace(/\.\./g, "_")
    .replace(/[\\/:*?"<>|\x00-\x1f]/g, "_")
    .replace(/^\.+/, "");

  return fetchWithRetry(artifact.sourceUrl, MAX_RETRIES)
    .then((res) => res.blob())
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = safeName && hasVisibleName ? safeName : "download";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
}
