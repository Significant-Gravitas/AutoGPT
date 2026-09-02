import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

interface FakeMeta {
  recordingId: string;
  mimeType: string;
  startedAt: number;
  durationSecs: number;
  finalized: boolean;
}

interface FakePart {
  id: string;
  recordingId: string;
  partIndex: number;
  blob: Blob;
  savedAt: number;
  uploaded: boolean;
}

// IndexedDB does not exist in happy-dom, so the recording store is replaced
// with an in-memory double. `readRecordingSnapshot` itself runs for real.
const { store } = vi.hoisted(() => ({
  store: {
    available: true,
    meta: null as FakeMeta | null,
    parts: [] as FakePart[],
    getPartsError: null as unknown,
  },
}));

vi.mock(
  "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/recordingStore",
  () => ({
    isIndexedDBAvailable: () => store.available,
    getMeta: async () => store.meta,
    getParts: async (recordingId: string) => {
      if (store.getPartsError) throw store.getPartsError;
      return store.parts.filter((part) => part.recordingId === recordingId);
    },
  }),
);

import {
  describeError,
  formatBytes,
  formatClock,
  formatMs,
  formatSeconds,
  formatValue,
  isProductionEnvironment,
  readRecordingSnapshot,
  recordingDownloadHref,
} from "../helpers";

function part(overrides: Partial<FakePart> = {}): FakePart {
  const partIndex = overrides.partIndex ?? 0;
  return {
    id: `rec-1:${partIndex}`,
    recordingId: "rec-1",
    partIndex,
    blob: new Blob(["x"]),
    savedAt: 1000,
    uploaded: false,
    ...overrides,
  };
}

beforeEach(() => {
  store.available = true;
  store.meta = null;
  store.parts = [];
  store.getPartsError = null;
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.useRealTimers();
});

describe("isProductionEnvironment", () => {
  it("is true only for the prod app env", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");
    expect(isProductionEnvironment()).toBe(true);

    // The env var is spelled both ways across our deploys.
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "production");
    expect(isProductionEnvironment()).toBe(true);

    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "dev");
    expect(isProductionEnvironment()).toBe(false);

    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
    expect(isProductionEnvironment()).toBe(false);

    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "");
    expect(isProductionEnvironment()).toBe(false);
  });
});

describe("recordingDownloadHref", () => {
  it("routes the generated recording URL through the proxy", () => {
    expect(recordingDownloadHref()).toBe(
      "/api/proxy/api/onboarding/brain-dump/recording",
    );
  });
});

describe("describeError", () => {
  it("uses the message for Errors and stringifies anything else", () => {
    expect(describeError(new Error("boom"))).toBe("boom");
    expect(describeError(new TypeError("bad type"))).toBe("bad type");
    expect(describeError("plain string")).toBe("plain string");
    expect(describeError(404)).toBe("404");
    expect(describeError(null)).toBe("null");
    expect(describeError(undefined)).toBe("undefined");
  });
});

describe("readRecordingSnapshot", () => {
  it("reports unsupported without touching the store when IndexedDB is missing", async () => {
    store.available = false;
    store.meta = {
      recordingId: "rec-1",
      mimeType: "audio/webm",
      startedAt: 1,
      durationSecs: 2,
      finalized: false,
    };

    const snapshot = await readRecordingSnapshot();

    expect(snapshot.supported).toBe(false);
    // The meta above must not leak through — nothing was read.
    expect(snapshot.meta).toBeNull();
    expect(snapshot.parts).toEqual([]);
    expect(snapshot.error).toBeNull();
    expect(snapshot.readAt).not.toBeNull();
  });

  it("returns an empty snapshot when no meta row is stored", async () => {
    store.parts = [part()];

    const snapshot = await readRecordingSnapshot();

    expect(snapshot.meta).toBeNull();
    expect(snapshot.parts).toEqual([]);
    expect(snapshot.totalBytes).toBe(0);
    expect(snapshot.pendingUploads).toBe(0);
    expect(snapshot.supported).toBe(true);
    expect(snapshot.readAt).not.toBeNull();
  });

  it("summarises the stored parts of the current recording", async () => {
    store.meta = {
      recordingId: "rec-1",
      mimeType: "audio/webm",
      startedAt: 500,
      durationSecs: 91.5,
      finalized: false,
    };
    store.parts = [
      part({
        partIndex: 0,
        blob: new Blob(["aaa"]),
        savedAt: 700,
        uploaded: true,
      }),
      part({ partIndex: 1, blob: new Blob(["bbbbb"]), savedAt: 900 }),
      part({ partIndex: 2, blob: new Blob(["cc"]), savedAt: 1100 }),
      // Belongs to a different take; must not be counted.
      part({
        partIndex: 0,
        recordingId: "rec-2",
        id: "rec-2:0",
        blob: new Blob(["zzzzzzzzzz"]),
      }),
    ];

    const snapshot = await readRecordingSnapshot();

    expect(snapshot.meta).toEqual(store.meta);
    expect(snapshot.parts).toEqual([
      { partIndex: 0, bytes: 3, savedAt: 700, uploaded: true },
      { partIndex: 1, bytes: 5, savedAt: 900, uploaded: false },
      { partIndex: 2, bytes: 2, savedAt: 1100, uploaded: false },
    ]);
    expect(snapshot.totalBytes).toBe(10);
    expect(snapshot.pendingUploads).toBe(2);
    expect(snapshot.error).toBeNull();
  });

  it("surfaces a read failure instead of throwing", async () => {
    store.meta = {
      recordingId: "rec-1",
      mimeType: "audio/webm",
      startedAt: 1,
      durationSecs: 2,
      finalized: false,
    };
    store.getPartsError = new Error("QuotaExceededError");

    const snapshot = await readRecordingSnapshot();

    expect(snapshot.error).toBe("QuotaExceededError");
    expect(snapshot.meta).toBeNull();
    expect(snapshot.parts).toEqual([]);
    expect(snapshot.readAt).not.toBeNull();
  });

  it("stamps readAt with the clock at read time", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2024-03-01T10:00:00.000Z"));

    const snapshot = await readRecordingSnapshot();

    expect(snapshot.readAt).toBe(Date.parse("2024-03-01T10:00:00.000Z"));
  });
});

describe("formatBytes", () => {
  it("climbs units only once the value reaches 1024", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1023)).toBe("1023 B");
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1024 * 1024)).toBe("1 MB");
    expect(formatBytes(1024 * 1024 * 1024)).toBe("1 GB");
  });

  it("stops at GB rather than inventing a unit", () => {
    expect(formatBytes(1024 ** 4)).toBe("1024 GB");
  });
});

describe("formatClock", () => {
  it("renders a 24-hour local clock and an em dash for nothing", () => {
    expect(formatClock(null)).toBe("—");
    expect(formatClock(undefined)).toBe("—");
    // Built from local-time parts so the assertion holds in any timezone.
    const localAfternoon = new Date(2024, 2, 1, 13, 5, 9).getTime();
    expect(formatClock(localAfternoon)).toMatch(/^13.05.09$/);
  });
});

describe("formatMs", () => {
  it("switches from whole milliseconds to two-decimal seconds at 1s", () => {
    expect(formatMs(null)).toBe("—");
    expect(formatMs(0)).toBe("0 ms");
    expect(formatMs(12.4)).toBe("12 ms");
    expect(formatMs(999)).toBe("999 ms");
    expect(formatMs(1000)).toBe("1.00 s");
    expect(formatMs(5432)).toBe("5.43 s");
  });
});

describe("formatSeconds", () => {
  it("renders one decimal place, or an em dash for nothing", () => {
    expect(formatSeconds(null)).toBe("—");
    expect(formatSeconds(undefined)).toBe("—");
    expect(formatSeconds(0)).toBe("0.0 s");
    expect(formatSeconds(95.44)).toBe("95.4 s");
  });
});

describe("formatValue", () => {
  it("maps booleans to yes/no and treats empty string as absent", () => {
    expect(formatValue(null)).toBe("—");
    expect(formatValue(undefined)).toBe("—");
    expect(formatValue("")).toBe("—");
    expect(formatValue(true)).toBe("yes");
    expect(formatValue(false)).toBe("no");
    expect(formatValue("transcribing")).toBe("transcribing");
  });
});
