import {
  getMeta,
  getParts,
  isIndexedDBAvailable,
  type RecordingMeta,
  type RecordingPart,
} from "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/recordingStore";
import { getDownloadBrainDumpRecordingUrl } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { AppEnv, environment } from "@/services/environment";

// Processing budget the brain-dump feature is held to for a 60-90s dump.
export const TRANSCRIBE_BUDGET_MS = 5000;
export const EXTRACT_BUDGET_MS = 5000;
export const BUDGET_DUMP_LENGTH = "60-90s dump";

export const STATUS_POLL_MS = 1000;
export const SNAPSHOT_POLL_MS = 1000;

// Server truncates the finalize preview; the full transcript never leaves
// the backend.
export const TRANSCRIPT_PREVIEW_CHARS = 280;

export interface DebugPart {
  partIndex: number;
  bytes: number;
  savedAt: number;
  uploaded: boolean;
}

export interface RecordingSnapshot {
  supported: boolean;
  meta: RecordingMeta | null;
  parts: DebugPart[];
  totalBytes: number;
  pendingUploads: number;
  readAt: number | null;
  error: string | null;
}

export const EMPTY_SNAPSHOT: RecordingSnapshot = {
  supported: true,
  meta: null,
  parts: [],
  totalBytes: 0,
  pendingUploads: 0,
  readAt: null,
  error: null,
};

export function isProductionEnvironment() {
  return environment.getAppEnv() === AppEnv.PROD;
}

export function recordingDownloadHref() {
  return `/api/proxy${getDownloadBrainDumpRecordingUrl()}`;
}

export function describeError(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

export async function readRecordingSnapshot(): Promise<RecordingSnapshot> {
  if (!isIndexedDBAvailable()) {
    return { ...EMPTY_SNAPSHOT, supported: false, readAt: Date.now() };
  }
  try {
    const meta = await getMeta();
    if (!meta) return { ...EMPTY_SNAPSHOT, readAt: Date.now() };

    const stored = await getParts(meta.recordingId);
    const parts = stored.map(toDebugPart);
    return {
      supported: true,
      meta,
      parts,
      totalBytes: parts.reduce((total, part) => total + part.bytes, 0),
      pendingUploads: parts.filter((part) => !part.uploaded).length,
      readAt: Date.now(),
      error: null,
    };
  } catch (error) {
    return {
      ...EMPTY_SNAPSHOT,
      readAt: Date.now(),
      error: describeError(error),
    };
  }
}

function toDebugPart(part: RecordingPart): DebugPart {
  return {
    partIndex: part.partIndex,
    bytes: part.blob.size,
    savedAt: part.savedAt,
    uploaded: part.uploaded,
  };
}

const BYTE_UNITS = ["B", "KB", "MB", "GB"];

export function formatBytes(bytes: number) {
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < BYTE_UNITS.length - 1) {
    value = value / 1024;
    unit += 1;
  }
  const rounded = unit === 0 ? value : Number(value.toFixed(1));
  return `${rounded} ${BYTE_UNITS[unit]}`;
}

export function formatClock(epochMs: number | null | undefined) {
  if (epochMs === null || epochMs === undefined) return "—";
  return new Date(epochMs).toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatMs(durationMs: number | null) {
  if (durationMs === null) return "—";
  if (durationMs < 1000) return `${Math.round(durationMs)} ms`;
  return `${(durationMs / 1000).toFixed(2)} s`;
}

export function formatSeconds(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "—";
  return `${seconds.toFixed(1)} s`;
}

export function formatValue(value: string | boolean | null | undefined) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return value === "" ? "—" : value;
}
