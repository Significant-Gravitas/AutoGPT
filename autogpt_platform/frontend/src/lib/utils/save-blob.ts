import { NativeDownloadChannel } from "./native-download-channel";
import {
  NATIVE_DOWNLOAD_CHUNK_BYTES,
  NATIVE_DOWNLOAD_FINISH_TIMEOUT_MS,
  NATIVE_DOWNLOAD_MAX_BYTES,
  NATIVE_DOWNLOAD_PICKER_TIMEOUT_MS,
  NATIVE_DOWNLOAD_STEP_TIMEOUT_MS,
  nativeDownloadFilename,
  nativeDownloadMimeType,
} from "./native-download-protocol";

let nativeDownloadActive = false;

function saveBrowserBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  try {
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
  } finally {
    link.remove();
    URL.revokeObjectURL(url);
  }
}

function downloadID() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

async function sendBlob(channel: NativeDownloadChannel, blob: Blob) {
  for (
    let offset = 0, index = 0;
    offset < blob.size;
    offset += NATIVE_DOWNLOAD_CHUNK_BYTES, index++
  ) {
    const bytes = new Uint8Array(
      await blob
        .slice(offset, offset + NATIVE_DOWNLOAD_CHUNK_BYTES)
        .arrayBuffer(),
    );
    const data = btoa(String.fromCharCode(...bytes));
    await channel.request(
      { type: "chunk", index, data },
      "ack",
      NATIVE_DOWNLOAD_STEP_TIMEOUT_MS,
    );
  }
  await channel.request(
    { type: "finish" },
    "complete",
    NATIVE_DOWNLOAD_FINISH_TIMEOUT_MS,
  );
}

export async function saveBlob(
  blob: Blob,
  filename: string,
  options: { signal?: AbortSignal } = {},
) {
  if (options.signal?.aborted)
    throw new DOMException("Download cancelled.", "AbortError");
  const bridge = window.top === window ? window.AutoGPTDownloads : undefined;
  if (!bridge || typeof bridge.postMessage !== "function") {
    saveBrowserBlob(blob, filename);
    return;
  }
  if (blob.size > NATIVE_DOWNLOAD_MAX_BYTES) {
    throw new Error(
      "Files larger than 50 MiB must be downloaded in your browser.",
    );
  }
  if (nativeDownloadActive)
    throw new Error("A download is already in progress.");
  nativeDownloadActive = true;
  let channel: NativeDownloadChannel | undefined;
  try {
    channel = new NativeDownloadChannel(bridge, downloadID(), options.signal);
    await channel.request(
      {
        type: "start",
        filename: nativeDownloadFilename(filename),
        mimeType: nativeDownloadMimeType(blob.type),
        size: blob.size,
      },
      "ready",
      NATIVE_DOWNLOAD_PICKER_TIMEOUT_MS,
    );
    await sendBlob(channel, blob);
  } catch (error) {
    channel?.cancel();
    throw error;
  } finally {
    channel?.close();
    nativeDownloadActive = false;
  }
}
