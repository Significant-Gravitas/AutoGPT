export interface NativeDownloadBridge {
  postMessage(message: string): void;
  onmessage?: ((event: { data: unknown }) => void) | null;
}

declare global {
  interface Window {
    AutoGPTDownloads?: NativeDownloadBridge;
  }
}

export const NATIVE_DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024;
export const NATIVE_DOWNLOAD_CHUNK_BYTES = 48 * 1024;
export const NATIVE_DOWNLOAD_PICKER_TIMEOUT_MS = 130000;
export const NATIVE_DOWNLOAD_FINISH_TIMEOUT_MS = 130000;
export const NATIVE_DOWNLOAD_STEP_TIMEOUT_MS = 30000;

export function readNativeDownloadReply(data: unknown) {
  if (typeof data !== "string" || data.length > 16384) return null;
  try {
    const value: unknown = JSON.parse(data);
    if (!value || typeof value !== "object") return null;
    if (!("id" in value) || typeof value.id !== "string") return null;
    if (!("type" in value) || typeof value.type !== "string") return null;
    return {
      id: value.id,
      type: value.type,
      index:
        "index" in value && typeof value.index === "number"
          ? value.index
          : undefined,
      message:
        "message" in value && typeof value.message === "string"
          ? value.message.slice(0, 500)
          : undefined,
    };
  } catch {
    return null;
  }
}

export function nativeDownloadMimeType(mimeType: string) {
  const bare = mimeType.split(";")[0].trim();
  return bare.length <= 128 &&
    /^[a-z0-9!#$&^_.+-]+\/[a-z0-9!#$&^_.+-]+$/i.test(bare)
    ? bare
    : "application/octet-stream";
}

export function nativeDownloadFilename(filename: string) {
  let result = "";
  for (const character of filename.trim()) {
    const code = character.charCodeAt(0);
    const safe =
      character === "/" ||
      character === "\\" ||
      code < 32 ||
      (code >= 127 && code <= 159)
        ? "_"
        : character;
    if (result.length + safe.length > 240) break;
    result += safe;
  }
  return !result || result === "." || result === ".." ? "download" : result;
}
