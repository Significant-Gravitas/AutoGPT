import JSZip from "jszip";
import { OutputRenderer, OutputMetadata } from "../types";

export interface DownloadItem {
  value: any;
  metadata?: OutputMetadata;
  renderer: OutputRenderer;
}

interface FetchResult {
  blob: Blob | null;
  failedUrl?: string;
  failedFilename?: string;
}

async function fetchFileAsBlob(
  url: string,
  filename: string,
): Promise<FetchResult> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      console.error(`Failed to fetch ${url}: ${response.status}`);
      return { blob: null, failedUrl: url, failedFilename: filename };
    }
    return { blob: await response.blob() };
  } catch (error) {
    console.error(`Error fetching ${url}:`, error);
    return { blob: null, failedUrl: url, failedFilename: filename };
  }
}

function getUniqueFilename(filename: string, usedNames: Set<string>): string {
  if (!usedNames.has(filename)) {
    usedNames.add(filename);
    return filename;
  }

  const dotIndex = filename.lastIndexOf(".");
  const baseName = dotIndex > 0 ? filename.slice(0, dotIndex) : filename;
  const extension = dotIndex > 0 ? filename.slice(dotIndex) : "";

  let counter = 1;
  let newName = `${baseName}_${counter}${extension}`;
  while (usedNames.has(newName)) {
    counter++;
    newName = `${baseName}_${counter}${extension}`;
  }
  usedNames.add(newName);
  return newName;
}

export async function downloadOutputs(items: DownloadItem[]) {
  const zip = new JSZip();
  const usedFilenames = new Set<string>();
  let hasFiles = false;

  const concatenableTexts: string[] = [];
  const failedDownloads: Array<{ url: string; filename: string }> = [];

  for (const item of items) {
    if (item.renderer.isConcatenable(item.value, item.metadata)) {
      const copyContent = item.renderer.getCopyContent(
        item.value,
        item.metadata,
      );
      if (copyContent) {
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
        let blob: Blob | null = null;
        const filename = downloadContent.filename;

        if (typeof downloadContent.data === "string") {
          if (downloadContent.data.startsWith("http")) {
            const result = await fetchFileAsBlob(
              downloadContent.data,
              filename,
            );
            blob = result.blob;
            if (!blob && result.failedUrl && result.failedFilename) {
              failedDownloads.push({
                url: result.failedUrl,
                filename: result.failedFilename,
              });
            }
          }
        } else {
          blob = downloadContent.data as Blob;
        }

        if (blob) {
          const uniqueFilename = getUniqueFilename(filename, usedFilenames);
          zip.file(uniqueFilename, blob);
          hasFiles = true;
        }
      }
    }
  }

  if (concatenableTexts.length > 0) {
    const combinedText = concatenableTexts.join("\n\n---\n\n");
    const filename = getUniqueFilename("combined_output.txt", usedFilenames);
    zip.file(filename, combinedText);
    hasFiles = true;
  }

  if (hasFiles) {
    const zipBlob = await zip.generateAsync({ type: "blob" });
    downloadBlob(zipBlob, "outputs.zip");
  }

  for (const failed of failedDownloads) {
    downloadViaAnchor(failed.url, failed.filename);
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

function downloadViaAnchor(url: string, filename: string) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
