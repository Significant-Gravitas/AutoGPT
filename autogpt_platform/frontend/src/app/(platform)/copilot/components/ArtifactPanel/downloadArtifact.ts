import type { ArtifactRef } from "../../store";

/**
 * Trigger a file download from an artifact URL.
 *
 * Uses fetch+blob instead of a bare `<a download>` because the browser
 * ignores the `download` attribute on cross-origin responses (GCS signed
 * URLs), and some browsers require the anchor to be attached to the DOM
 * before `.click()` fires the download.
 */
export function downloadArtifact(artifact: ArtifactRef): Promise<void> {
  // Replace path separators, Windows-reserved chars, control chars, and
  // parent-dir sequences so the browser-assigned filename is safe to write
  // anywhere on the user's filesystem.
  const safeName =
    artifact.title
      .replace(/\.\./g, "_")
      .replace(/[\\/:*?"<>|\x00-\x1f]/g, "_")
      .replace(/^\.+/, "") || "download";
  return fetch(artifact.sourceUrl)
    .then((res) => {
      if (!res.ok) throw new Error(`Download failed: ${res.status}`);
      return res.blob();
    })
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = safeName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
}
