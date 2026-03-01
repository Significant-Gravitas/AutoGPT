import JSZip from "jszip";
import { OutputRenderer, OutputMetadata } from "../types";

export interface DownloadItem {
  value: any;
  metadata?: OutputMetadata;
  renderer: OutputRenderer;
}

/** Maximum individual file size for zip inclusion (50 MB) */
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024;

async function fetchFileAsBlob(url: string): Promise<Blob | null> {
  try {
    const response = await fetch(url, { mode: "cors" });
    if (!response.ok) {
      console.error(`Failed to fetch ${url}: ${response.status}`);
      return null;
    }
    const blob = await response.blob();
    if (blob.size > MAX_FILE_SIZE_BYTES) {
      console.warn(
        `Skipping ${url}: file too large (${(blob.size / 1024 / 1024).toFixed(1)} MB, limit ${MAX_FILE_SIZE_BYTES / 1024 / 1024} MB)`,
      );
      return null;
    }
    return blob;
  } catch (error) {
    // CORS or network error — fall back to adding a link reference
    // instead of silently dropping the file
    console.warn(
      `Could not fetch ${url} (likely CORS). Adding as link reference.`,
    );
    return null;
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
  const unfetchableUrls: string[] = [];

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
        let sourceUrl: string | null = null;

        if (typeof downloadContent.data === "string") {
          if (downloadContent.data.startsWith("http")) {
            sourceUrl = downloadContent.data;
            blob = await fetchFileAsBlob(downloadContent.data);
          }
        } else {
          blob = downloadContent.data as Blob;
        }

        if (blob) {
          const uniqueFilename = getUniqueFilename(filename, usedFilenames);
          zip.file(uniqueFilename, blob);
          hasFiles = true;
        } else if (sourceUrl) {
          // File couldn't be fetched (CORS, too large, etc.) — track for reference
          unfetchableUrls.push(sourceUrl);
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

  // Include references to files that couldn't be fetched (CORS, size limit, etc.)
  if (unfetchableUrls.length > 0) {
    const linksContent = unfetchableUrls
      .map((url, i) => `${i + 1}. ${url}`)
      .join("\n");
    zip.file(
      "unfetched_files.txt",
      `The following files could not be included in the zip (CORS restriction or size limit).\nYou can download them directly from these URLs:\n\n${linksContent}\n`,
    );
    hasFiles = true;
  }

  if (hasFiles) {
    const zipBlob = await zip.generateAsync({ type: "blob" });
    downloadBlob(zipBlob, "outputs.zip");
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
