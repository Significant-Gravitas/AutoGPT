import { describe, expect, it } from "vitest";
import {
  EMPTY_SNAPSHOT,
  EXTRACT_BUDGET_MS,
  TRANSCRIBE_BUDGET_MS,
  type DebugPart,
  type RecordingSnapshot,
} from "../helpers";
import { buildWaterfall, isOverBudget, type StatusSeenAt } from "../waterfall";

function snapshotWith(
  parts: DebugPart[],
  meta: RecordingSnapshot["meta"] = null,
): RecordingSnapshot {
  return {
    ...EMPTY_SNAPSHOT,
    meta,
    parts,
    totalBytes: parts.reduce((total, part) => total + part.bytes, 0),
    pendingUploads: parts.filter((part) => !part.uploaded).length,
  };
}

function part(partIndex: number, savedAt: number, uploaded = false): DebugPart {
  return { partIndex, bytes: 10, savedAt, uploaded };
}

function meta(startedAt: number): RecordingSnapshot["meta"] {
  return {
    recordingId: "rec-1",
    mimeType: "audio/webm",
    startedAt,
    durationSecs: 60,
    finalized: false,
  };
}

function build(args: {
  snapshot?: RecordingSnapshot;
  seenAt?: StatusSeenAt;
  introPath?: "A" | "B" | null;
}) {
  const stages = buildWaterfall({
    snapshot: args.snapshot ?? EMPTY_SNAPSHOT,
    seenAt: args.seenAt ?? {},
    introPath: args.introPath ?? null,
  });
  return {
    stages,
    byId: (id: string) => {
      const stage = stages.find((candidate) => candidate.id === id);
      if (!stage) throw new Error(`no stage ${id}`);
      return stage;
    },
  };
}

describe("buildWaterfall", () => {
  it("returns the five pipeline stages in pipeline order", () => {
    const { stages } = build({});

    expect(stages.map((stage) => stage.id)).toEqual([
      "record",
      "upload",
      "transcribe",
      "extract",
      "intro",
    ]);
    expect(stages.map((stage) => stage.label)).toEqual([
      "Record",
      "Upload",
      "Transcribe",
      "Extract",
      "Intro message",
    ]);
  });

  it("only holds transcribe and extract to a budget", () => {
    const { byId } = build({});

    expect(byId("transcribe").budgetMs).toBe(TRANSCRIBE_BUDGET_MS);
    expect(byId("extract").budgetMs).toBe(EXTRACT_BUDGET_MS);
    expect(byId("record").budgetMs).toBeNull();
    expect(byId("upload").budgetMs).toBeNull();
    expect(byId("intro").budgetMs).toBeNull();
  });
});

describe("record stage", () => {
  it("measures meta.startedAt to the last part's savedAt", () => {
    const { byId } = build({
      snapshot: snapshotWith(
        [part(0, 1500), part(1, 2500), part(2, 4000)],
        meta(1000),
      ),
    });

    expect(byId("record").durationMs).toBe(3000);
    expect(byId("record").source).toContain("Measured from IndexedDB");
  });

  it("falls back to the first part when there is no meta row", () => {
    const { byId } = build({
      snapshot: snapshotWith([part(0, 1500), part(1, 4000)]),
    });

    expect(byId("record").durationMs).toBe(2500);
  });

  it("is unmeasured when nothing is stored", () => {
    const { byId } = build({ snapshot: snapshotWith([], meta(1000)) });

    expect(byId("record").durationMs).toBeNull();
    expect(byId("record").source).toContain("Unmeasured");
  });
});

describe("upload stage", () => {
  it("stays unmeasured and reports how much of the queue is undrained", () => {
    const { byId } = build({
      snapshot: snapshotWith([part(0, 1, true), part(1, 2), part(2, 3)]),
    });

    expect(byId("upload").durationMs).toBeNull();
    expect(byId("upload").source).toContain("2 of 3 part(s)");
  });
});

describe("transcribe stage", () => {
  it("spans the first upload/transcribing sighting to the first transcribed-or-later one", () => {
    const { byId } = build({
      seenAt: {
        recording_uploaded: 1000,
        transcribing: 1200,
        transcribed: 4000,
        completed: 9000,
      },
    });

    expect(byId("transcribe").durationMs).toBe(3000);
    expect(byId("transcribe").source).toContain(
      "Measured from polled status transitions",
    );
  });

  it("uses the earliest sighting on each side when a status was missed", () => {
    // The poll never caught `recording_uploaded` or `transcribed`; the
    // remaining statuses still bracket the phase.
    const { byId } = build({
      seenAt: { transcribing: 2000, completed: 6000, extracting: 5000 },
    });

    expect(byId("transcribe").durationMs).toBe(3000);
  });

  it("is unmeasured when only one side was observed", () => {
    const { byId } = build({ seenAt: { transcribing: 2000 } });

    expect(byId("transcribe").durationMs).toBeNull();
    expect(byId("transcribe").source).toContain("Unmeasured");
    expect(byId("transcribe").source).toContain(
      "Keep it open across a finalize",
    );
  });
});

describe("extract stage", () => {
  it("spans the first transcribed/extracting sighting to completed", () => {
    const { byId } = build({
      seenAt: { transcribed: 4000, extracting: 4500, completed: 9000 },
    });

    expect(byId("extract").durationMs).toBe(5000);
  });

  it("is unmeasured when the run never reached completed", () => {
    const { byId } = build({ seenAt: { transcribed: 4000, failed: 5000 } });

    expect(byId("extract").durationMs).toBeNull();
  });

  it("refuses a negative span when timestamps arrive out of order", () => {
    // A clock jump backwards must not be reported as a duration.
    const { byId } = build({ seenAt: { transcribed: 9000, completed: 4000 } });

    expect(byId("extract").durationMs).toBeNull();
  });

  it("reports a zero-length span when both statuses land on the same poll", () => {
    const { byId } = build({ seenAt: { transcribed: 4000, completed: 4000 } });

    expect(byId("extract").durationMs).toBe(0);
  });
});

describe("intro stage", () => {
  it("names the pending handoff path when one is queued", () => {
    const { byId } = build({ introPath: "B" });

    expect(byId("intro").durationMs).toBeNull();
    expect(byId("intro").source).toContain("Pending handoff path: B");
  });

  it("points at the sessionStorage key when there is no handoff", () => {
    const { byId } = build({ introPath: null });

    expect(byId("intro").source).toContain("autogpt:onboarding-intro-path");
  });
});

describe("isOverBudget", () => {
  it("is true only when a measured duration exceeds a set budget", () => {
    const stage = {
      id: "transcribe",
      label: "Transcribe",
      source: "",
      budgetMs: 5000,
      durationMs: 5001,
    };

    expect(isOverBudget(stage)).toBe(true);
    expect(isOverBudget({ ...stage, durationMs: 5000 })).toBe(false);
    expect(isOverBudget({ ...stage, durationMs: 4999 })).toBe(false);
    expect(isOverBudget({ ...stage, durationMs: null })).toBe(false);
    expect(isOverBudget({ ...stage, budgetMs: null })).toBe(false);
  });
});
