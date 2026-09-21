// Saving a fetched blob to disk is the same anchor dance everywhere, so it
// lives here rather than beside whichever feature downloaded something first.
export function downloadFile(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // `click()` only queues the download; revoking in the same tick can leave
  // Safari and Firefox fetching a URL that no longer resolves, which saves a
  // zero-byte file with no error. Same deferral the admin export helpers use.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

const ENCODED_FILENAME = /filename\*\s*=\s*UTF-8''([^;]+)/i;
const QUOTED_FILENAME = /filename\s*=\s*"([^"]*)"/i;
const BARE_FILENAME = /filename\s*=\s*([^;]+)/i;

/** The name the server picked for an attachment, or ours when it picked none.
 *  RFC 5987's `filename*` wins where both forms are present: it is the one
 *  that survives a non-ascii name. */
export function filenameFromContentDisposition(
  headers: Headers,
  fallback: string,
): string {
  const header = headers.get("content-disposition");
  if (!header) return fallback;

  const encoded = ENCODED_FILENAME.exec(header);
  if (encoded) {
    try {
      const decoded = decodeURIComponent(encoded[1].trim());
      if (decoded) return decoded;
    } catch {
      // A malformed escape falls through to the plain `filename` beside it.
    }
  }

  const plain = QUOTED_FILENAME.exec(header) ?? BARE_FILENAME.exec(header);
  return plain?.[1].trim() || fallback;
}
