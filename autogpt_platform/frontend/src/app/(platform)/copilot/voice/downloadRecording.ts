/**
 * Handing the recording back is the last resort when transcription keeps
 * failing: the user has already said the thing, and re-saying it is the one
 * cost a retry cannot refund.
 */

const EXTENSIONS: Record<string, string> = {
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/wave": "wav",
};

/** `audio/webm;codecs=opus` is what MediaRecorder actually reports. */
function baseType(blob: Blob): string {
  return blob.type.split(";")[0].trim().toLowerCase();
}

export function recordingExtension(blob: Blob): string {
  return EXTENSIONS[baseType(blob)] ?? "webm";
}

export function recordingFileName(blob: Blob, at: Date = new Date()): string {
  const stamp = at.toISOString().slice(0, 19).replace(/[:T]/g, "-");
  return `voice-recording-${stamp}.${recordingExtension(blob)}`;
}

export function downloadRecording(blob: Blob, at?: Date): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = recordingFileName(blob, at);
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Immediate revocation cancels the download in Safari; one tick is enough
  // for every browser to have taken the blob.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
