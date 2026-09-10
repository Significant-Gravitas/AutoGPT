import type { BrainDumpStatus } from "@/app/api/__generated__/models/brainDumpStatus";
import type { IntroPath } from "@/services/onboarding/brain-dump-handoff";
import {
  EXTRACT_BUDGET_MS,
  TRANSCRIBE_BUDGET_MS,
  type RecordingSnapshot,
} from "./helpers";

export type StatusSeenAt = Partial<Record<BrainDumpStatus, number>>;

export interface WaterfallStage {
  id: string;
  label: string;
  durationMs: number | null;
  budgetMs: number | null;
  source: string;
}

interface Args {
  snapshot: RecordingSnapshot;
  seenAt: StatusSeenAt;
  introPath: IntroPath | null;
}

export function buildWaterfall({ snapshot, seenAt, introPath }: Args) {
  return [
    recordStage(snapshot),
    uploadStage(snapshot),
    transcribeStage(seenAt),
    extractStage(seenAt),
    introStage(introPath),
  ];
}

export function isOverBudget(stage: WaterfallStage) {
  return stage.durationMs !== null && stage.budgetMs !== null
    ? stage.durationMs > stage.budgetMs
    : false;
}

function recordStage(snapshot: RecordingSnapshot): WaterfallStage {
  const firstSavedAt = snapshot.parts.at(0)?.savedAt ?? null;
  const lastSavedAt = snapshot.parts.at(-1)?.savedAt ?? null;
  const startedAt = snapshot.meta?.startedAt ?? firstSavedAt;
  const durationMs =
    startedAt !== null && lastSavedAt !== null ? lastSavedAt - startedAt : null;

  return {
    id: "record",
    label: "Record",
    durationMs,
    budgetMs: null,
    source:
      durationMs === null
        ? "Unmeasured — no recording currently in IndexedDB."
        : "Measured from IndexedDB: meta.startedAt → savedAt of the last part.",
  };
}

function uploadStage(snapshot: RecordingSnapshot): WaterfallStage {
  return {
    id: "upload",
    label: "Upload",
    durationMs: null,
    budgetMs: null,
    source:
      "Unmeasured — RecordingPart records `uploaded` as a boolean with no " +
      "`uploadedAt`, so per-part upload latency cannot be derived. " +
      `${snapshot.pendingUploads} of ${snapshot.parts.length} part(s) are still not marked uploaded.`,
  };
}

function transcribeStage(seenAt: StatusSeenAt): WaterfallStage {
  const startedAt = firstSeen(seenAt, ["recording_uploaded", "transcribing"]);
  const endedAt = firstSeen(seenAt, ["transcribed", "extracting", "completed"]);
  return {
    id: "transcribe",
    label: "Transcribe",
    durationMs: span(startedAt, endedAt),
    budgetMs: TRANSCRIBE_BUDGET_MS,
    source: transitionSource(
      startedAt,
      endedAt,
      "recording_uploaded/transcribing",
      "transcribed",
    ),
  };
}

function extractStage(seenAt: StatusSeenAt): WaterfallStage {
  const startedAt = firstSeen(seenAt, ["transcribed", "extracting"]);
  const endedAt = firstSeen(seenAt, ["completed"]);
  return {
    id: "extract",
    label: "Extract",
    durationMs: span(startedAt, endedAt),
    budgetMs: EXTRACT_BUDGET_MS,
    source: transitionSource(
      startedAt,
      endedAt,
      "transcribed/extracting",
      "completed",
    ),
  };
}

function introStage(introPath: IntroPath | null): WaterfallStage {
  return {
    id: "intro",
    label: "Intro message",
    durationMs: null,
    budgetMs: null,
    source: introPath
      ? `Unmeasured — the intro turn is composed on the copilot home after handoff. Pending handoff path: ${introPath}.`
      : "Unmeasured — no intro handoff in sessionStorage (autogpt:onboarding-intro-path), and no endpoint reports intro latency.",
  };
}

function firstSeen(seenAt: StatusSeenAt, statuses: BrainDumpStatus[]) {
  const times = statuses
    .map((status) => seenAt[status])
    .filter((time): time is number => time !== undefined);
  return times.length > 0 ? Math.min(...times) : null;
}

function span(startedAt: number | null, endedAt: number | null) {
  return startedAt !== null && endedAt !== null && endedAt >= startedAt
    ? endedAt - startedAt
    : null;
}

function transitionSource(
  startedAt: number | null,
  endedAt: number | null,
  from: string,
  to: string,
) {
  if (startedAt !== null && endedAt !== null) {
    return `Measured from polled status transitions: first "${from}" → first "${to}" (1s poll resolution).`;
  }
  return `Unmeasured — this page did not observe both the "${from}" and "${to}" statuses while polling. Keep it open across a finalize to capture them.`;
}
