import type {
  OutputRenderer,
  OutputMetadata,
} from "@/components/contextual/OutputRenderers/types";

export interface DownloadItem {
  value: unknown;
  metadata?: OutputMetadata;
  renderer: OutputRenderer;
}

/** Maximum individual file size for zip inclusion (50 MB) */
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024;

/** Maximum total zip content size before generation (200 MB) */
const MAX_TOTAL_SIZE_BYTES = 200 * 1024 * 1024;

/** Maximum concurrent file fetches */
const FETCH_CONCURRENCY = 5;

async function fetchFileAsBlob(url: string): Promise<Blob | null> {
  try {
    const response = await fetch(url, { mode: "cors" });
    if (!response.ok) {
      console.error(`Failed to fetch ${url}: ${response.status}`);
      return null;
    }
    const contentLength = Number(response.headers.get("content-length") ?? "0");
    if (contentLength > MAX_FILE_SIZE_BYTES) {
      console.warn(
        `Skipping ${url}: file too large (${(contentLength / 1024 / 1024).toFixed(1)} MB, limit ${MAX_FILE_SIZE_BYTES / 1024 / 1024} MB)`,
      );
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
  } catch (_error) {
    console.warn(
      `Could not fetch ${url} (likely CORS). Adding as link reference.`,
    );
    return null;
  }
}

/** Strip path traversal components and unsafe characters from a filename. */
export function sanitizeFilename(filename: string): string {
  const sanitized = filename
    .replace(/[/\\]/g, "_")
    .replace(/^\.+/, "")
    .replace(/\.\./g, "_");
  return sanitized || "file";
}

export function getUniqueFilename(
  filename: string,
  usedNames: Set<string>,
): string {
  const safe = sanitizeFilename(filename);
  if (!usedNames.has(safe)) {
    usedNames.add(safe);
    return safe;
  }

  const dotIndex = safe.lastIndexOf(".");
  const baseName = dotIndex > 0 ? safe.slice(0, dotIndex) : safe;
  const extension = dotIndex > 0 ? safe.slice(dotIndex) : "";

  let counter = 1;
  let newName = `${baseName}_${counter}${extension}`;
  while (usedNames.has(newName)) {
    counter++;
    newName = `${baseName}_${counter}${extension}`;
  }
  usedNames.add(newName);
  return newName;
}

async function fetchInParallel<T>(
  tasks: (() => Promise<T>)[],
  concurrency: number,
): Promise<T[]> {
  const results: T[] = [];
  let index = 0;

  async function worker() {
    while (index < tasks.length) {
      const i = index++;
      results[i] = await tasks[i]();
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(concurrency, tasks.length) }, () => worker()),
  );
  return results;
}

type FetchResult = {
  blob: Blob | null;
  filename: string;
  sourceUrl: string | null;
};

export async function downloadOutputs(items: DownloadItem[]) {
  if (items.length === 0) return;

  const { default: JSZip } = await import("jszip");
  const zip = new JSZip();
  const usedFilenames = new Set<string>();
  let hasFiles = false;
  let totalSize = 0;

  const concatenableTexts: string[] = [];
  const unfetchableUrls: string[] = [];

  const fileItems: Array<{
    downloadContent: { data: unknown; filename: string };
  }> = [];

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
        fileItems.push({ downloadContent });
      }
    }
  }

  const fetchTasks = fileItems.map(
    ({ downloadContent }) =>
      async (): Promise<FetchResult> => {
        let blob: Blob | null = null;
        const filename = downloadContent.filename;
        let sourceUrl: string | null = null;

        if (typeof downloadContent.data === "string") {
          if (
            downloadContent.data.startsWith("http://") ||
            downloadContent.data.startsWith("https://") ||
            downloadContent.data.startsWith("/")
          ) {
            sourceUrl = downloadContent.data;
            blob = await fetchFileAsBlob(downloadContent.data);
          } else if (downloadContent.data.startsWith("data:")) {
            try {
              const dataBlob = await fetch(downloadContent.data).then((r) =>
                r.blob(),
              );
              if (dataBlob.size <= MAX_FILE_SIZE_BYTES) {
                blob = dataBlob;
              } else {
                console.warn(
                  `Skipping data URL: too large (${(dataBlob.size / 1024 / 1024).toFixed(1)} MB)`,
                );
              }
            } catch (_error) {
              console.warn(
                `Failed to process data URL for ${filename}: malformed or unsupported format`,
              );
            }
          } else {
            console.warn(
              `Skipping unsupported URL format: ${downloadContent.data.slice(0, 50)}...`,
            );
          }
        } else {
          const rawBlob = downloadContent.data as Blob;
          if (rawBlob.size <= MAX_FILE_SIZE_BYTES) {
            blob = rawBlob;
          } else {
            console.warn(
              `Skipping ${filename}: blob too large (${(rawBlob.size / 1024 / 1024).toFixed(1)} MB)`,
            );
          }
        }

        return { blob, filename, sourceUrl };
      },
  );

  const results = await fetchInParallel(fetchTasks, FETCH_CONCURRENCY);

  for (const { blob, filename, sourceUrl } of results) {
    if (blob) {
      if (totalSize + blob.size > MAX_TOTAL_SIZE_BYTES) {
        console.warn(
          `Skipping ${filename}: would exceed total zip size limit (${MAX_TOTAL_SIZE_BYTES / 1024 / 1024} MB)`,
        );
        if (sourceUrl) unfetchableUrls.push(sourceUrl);
        continue;
      }
      const uniqueFilename = getUniqueFilename(filename, usedFilenames);
      zip.file(uniqueFilename, blob);
      totalSize += blob.size;
      hasFiles = true;
    } else if (sourceUrl) {
      unfetchableUrls.push(sourceUrl);
    }
  }

  if (concatenableTexts.length > 0) {
    const combinedText = concatenableTexts.join("\n\n---\n\n");
    const textSize = new Blob([combinedText]).size;
    if (totalSize + textSize <= MAX_TOTAL_SIZE_BYTES) {
      const filename = getUniqueFilename("combined_output.txt", usedFilenames);
      zip.file(filename, combinedText);
      totalSize += textSize;
      hasFiles = true;
    }
  }

  if (unfetchableUrls.length > 0) {
    const linksContent = unfetchableUrls
      .map((url, i) => `${i + 1}. ${url}`)
      .join("\n");
    const manifest = `The following files could not be included in the zip (CORS restriction or size limit).\nYou can download them directly from these URLs:\n\n${linksContent}\n`;
    const manifestSize = new Blob([manifest]).size;
    if (totalSize + manifestSize <= MAX_TOTAL_SIZE_BYTES) {
      const manifestFilename = getUniqueFilename(
        "unfetched_files.txt",
        usedFilenames,
      );
      zip.file(manifestFilename, manifest);
      totalSize += manifestSize;
      hasFiles = true;
    }
  }

  if (!hasFiles) return;

  // Single-file shortcut: download directly instead of wrapping in a zip
  if (
    zip.files &&
    Object.keys(zip.files).length === 1 &&
    unfetchableUrls.length === 0
  ) {
    const onlyFilename = Object.keys(zip.files)[0];
    const entry = zip.files[onlyFilename];
    const content = await entry.async("blob");
    downloadBlob(content, onlyFilename);
    return;
  }

  const zipBlob = await zip.generateAsync({ type: "blob" });
  downloadBlob(zipBlob, "outputs.zip");
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
