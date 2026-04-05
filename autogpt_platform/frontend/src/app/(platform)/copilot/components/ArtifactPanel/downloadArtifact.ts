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
  const safeName =
    artifact.title.replace(/[\\/:*?"<>|\x00-\x1f]/g, "_") || "download";
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
