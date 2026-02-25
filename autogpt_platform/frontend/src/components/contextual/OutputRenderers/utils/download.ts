import { OutputRenderer, OutputMetadata } from "../types";

export interface DownloadItem {
  value: any;
  metadata?: OutputMetadata;
  renderer: OutputRenderer;
}

export async function downloadOutputs(items: DownloadItem[]) {
  const concatenableTexts: string[] = [];
  const nonConcatenableDownloads: Array<{ blob: Blob; filename: string }> = [];

  for (const item of items) {
    if (item.renderer.isConcatenable(item.value, item.metadata)) {
      const copyContent = item.renderer.getCopyContent(
        item.value,
        item.metadata,
      );
      if (copyContent) {
        // Extract text from CopyContent
        let text: string;
        if (typeof copyContent.data === "string") {
          text = copyContent.data;
        } else if (copyContent.fallbackText) {
          text = copyContent.fallbackText;
        } else {
          continue;
        }
        concatenableTexts.push(text);
      }
    } else {
      const downloadContent = item.renderer.getDownloadContent(
        item.value,
        item.metadata,
      );
      if (downloadContent) {
        if (typeof downloadContent.data === "string") {
          if (downloadContent.data.startsWith("http")) {
            const link = document.createElement("a");
            link.href = downloadContent.data;
            link.download = downloadContent.filename;
            link.click();
          }
        } else {
          nonConcatenableDownloads.push({
            blob: downloadContent.data as Blob,
            filename: downloadContent.filename,
          });
        }
      }
    }
  }

  if (concatenableTexts.length > 0) {
    const combinedText = concatenableTexts.join("\n\n---\n\n");
    const blob = new Blob([combinedText], { type: "text/plain" });
    downloadBlob(blob, "combined_output.txt");
  }

  for (const download of nonConcatenableDownloads) {
    downloadBlob(download.blob, download.filename);
  }
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
