import { CopyContent } from "../types";

export function isClipboardTypeSupported(mimeType: string): boolean {
  // ClipboardItem.supports() is the proper way to check
  if ("ClipboardItem" in window && "supports" in ClipboardItem) {
    return ClipboardItem.supports(mimeType);
  }

  // Fallback for browsers that don't support the supports() method
  // These are generally supported
  const fallbackSupported = ["text/plain", "text/html", "image/png"];

  return fallbackSupported.includes(mimeType);
}

export function getSupportedClipboardType(
  preferredTypes: string[],
): string | null {
  for (const type of preferredTypes) {
    if (isClipboardTypeSupported(type)) {
      return type;
    }
  }
  return null;
}

export async function copyToClipboard(copyContent: CopyContent): Promise<void> {
  try {
    // Determine the best supported MIME type
    const supportedTypes = [
      copyContent.mimeType,
      ...(copyContent.alternativeMimeTypes || []),
    ];
    const bestType = getSupportedClipboardType(supportedTypes);

    if (!bestType) {
      // No supported type found, use fallback text if available
      if (copyContent.fallbackText) {
        await navigator.clipboard.writeText(copyContent.fallbackText);
        return;
      }
      throw new Error(
        `None of the MIME types are supported: ${supportedTypes.join(", ")}`,
      );
    }

    // Get the data (resolve if it's a function)
    let data = copyContent.data;
    if (typeof data === "function") {
      data = await data();
    }

    // If data is already a Blob, use it directly
    if (data instanceof Blob) {
      // If we need a different MIME type than the blob has, recreate it
      if (bestType !== data.type && bestType !== copyContent.mimeType) {
        data = new Blob([data], { type: bestType });
      }

      await navigator.clipboard.write([
        new ClipboardItem({
          [bestType]: data,
        }),
      ]);
      return;
    }

    // If data is a string
    if (typeof data === "string") {
      // For plain text, use the simpler writeText API
      if (bestType === "text/plain") {
        await navigator.clipboard.writeText(data);
        return;
      }

      // For other text formats (HTML, JSON, etc.), create a blob
      const blob = new Blob([data], { type: bestType });
      await navigator.clipboard.write([
        new ClipboardItem({
          [bestType]: blob,
        }),
      ]);
    }
  } catch (_error) {
    // If rich copy fails and we have fallback text, try that
    if (copyContent.fallbackText) {
      try {
        await navigator.clipboard.writeText(copyContent.fallbackText);
        return;
      } catch {
        // Even fallback failed
      }
    }
    throw _error;
  }
}

export async function fetchAndCopyImage(imageUrl: string): Promise<void> {
  try {
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error("Failed to fetch image");

    const blob = await response.blob();
    const mimeType = blob.type || "image/png";

    await navigator.clipboard.write([
      new ClipboardItem({
        [mimeType]: blob,
      }),
    ]);
  } catch (_error) {
    // If fetching fails (e.g., CORS), fall back to copying the URL
    await navigator.clipboard.writeText(imageUrl);
  }
}
