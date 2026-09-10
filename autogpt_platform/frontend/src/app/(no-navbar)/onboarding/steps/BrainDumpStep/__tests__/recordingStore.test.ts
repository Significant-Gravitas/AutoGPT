import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clearRecording,
  getMeta,
  getMetaById,
  getParts,
  isIndexedDBAvailable,
  markPartUploaded,
  partId,
  saveMeta,
  savePart,
  type RecordingMeta,
  type RecordingPart,
} from "../recordingStore";

// happy-dom has no IndexedDB and `fake-indexeddb` is not a devDependency
// here, so the whole factory is faked in-memory. It models the parts the
// store actually leans on: events fire asynchronously, a transaction
// commits *after* its requests succeed, an upgrade can be blocked by
// another connection, and connections have to be closed by hand.
const DB_NAME = "autogpt-onboarding-brain-dump";

interface FakeStoreData {
  keyPath: string;
  records: Map<unknown, Record<string, unknown>>;
  indexes: Map<string, string>;
}

interface FakeDatabaseData {
  version: number;
  stores: Map<string, FakeStoreData>;
}

class FakeRequest {
  result: unknown = undefined;
  error: unknown = null;
  onsuccess: (() => void) | null = null;
  onerror: (() => void) | null = null;
}

class FakeOpenRequest extends FakeRequest {
  onupgradeneeded: (() => void) | null = null;
  onblocked: (() => void) | null = null;
}

class FakeStoreHandle {
  constructor(
    private data: FakeStoreData,
    private transaction: FakeTransaction | null,
    private factory: FakeIndexedDB,
  ) {}

  createIndex(name: string, keyPath: string) {
    this.data.indexes.set(name, keyPath);
  }

  put(value: Record<string, unknown>) {
    return this.run(() => {
      if (this.factory.failWrites) throw new Error("QuotaExceededError");
      const key = value[this.data.keyPath];
      this.data.records.set(key, value);
      return key;
    });
  }

  get(key: unknown) {
    return this.run(() => this.data.records.get(key));
  }

  getAll() {
    return this.run(() => [...this.data.records.values()]);
  }

  delete(key: unknown) {
    return this.run(() => {
      this.data.records.delete(key);
      return undefined;
    });
  }

  index(name: string) {
    const keyPath = this.data.indexes.get(name);
    if (!keyPath) throw new Error(`no index ${name}`);
    return {
      getAll: (value: unknown) =>
        this.run(() =>
          [...this.data.records.values()].filter(
            (record) => record[keyPath] === value,
          ),
        ),
    };
  }

  private run(execute: () => unknown) {
    const request = new FakeRequest();
    if (!this.transaction) throw new Error("no transaction");
    this.transaction.push(request, execute);
    return request;
  }
}

class FakeTransaction {
  error: unknown = null;
  oncomplete: (() => void) | null = null;
  onabort: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private operations: (() => void)[] = [];
  private aborted = false;

  constructor(
    private data: FakeDatabaseData,
    private factory: FakeIndexedDB,
  ) {
    queueMicrotask(() => this.settle());
  }

  objectStore(name: string) {
    const store = this.data.stores.get(name);
    if (!store) throw new Error(`no store ${name}`);
    return new FakeStoreHandle(store, this, this.factory);
  }

  push(request: FakeRequest, execute: () => unknown) {
    this.operations.push(() => {
      try {
        request.result = execute();
        request.onsuccess?.();
      } catch (error) {
        request.error = error;
        this.error = error;
        this.aborted = true;
        request.onerror?.();
      }
    });
  }

  private settle() {
    for (const operation of this.operations) {
      if (this.aborted) break;
      operation();
    }
    // A commit can fail after every request reported success — the case
    // the store is careful to resolve on `oncomplete` for.
    if (!this.aborted && this.factory.failCommit) {
      this.error = new Error("commit failed");
      this.aborted = true;
    }
    if (this.aborted) this.onabort?.();
    else this.oncomplete?.();
  }
}

class FakeConnection {
  constructor(
    private data: FakeDatabaseData,
    private factory: FakeIndexedDB,
  ) {}

  get objectStoreNames() {
    const names = [...this.data.stores.keys()];
    return { contains: (name: string) => names.includes(name) };
  }

  createObjectStore(name: string, options: { keyPath: string }) {
    const store: FakeStoreData = {
      keyPath: options.keyPath,
      records: new Map(),
      indexes: new Map(),
    };
    this.data.stores.set(name, store);
    return new FakeStoreHandle(store, null, this.factory);
  }

  deleteObjectStore(name: string) {
    this.data.stores.delete(name);
  }

  transaction(_storeName: string, _mode: string) {
    return new FakeTransaction(this.data, this.factory);
  }

  close() {
    this.factory.openConnections.delete(this);
  }
}

class FakeIndexedDB {
  readonly openConnections = new Set<FakeConnection>();
  failWrites = false;
  failCommit = false;
  // Stands in for another tab holding an older version open.
  blockUpgrades = false;
  private databases = new Map<string, FakeDatabaseData>();
  private parked: (() => void)[] = [];

  open(name: string, version: number) {
    const request = new FakeOpenRequest();
    queueMicrotask(() => {
      const existing = this.databases.get(name);
      if (this.blockUpgrades && (!existing || existing.version < version)) {
        this.parked.push(() => this.finishOpen(request, name, version));
        request.onblocked?.();
        return;
      }
      this.finishOpen(request, name, version);
    });
    return request;
  }

  // The blocking tab went away: parked upgrades now run to completion,
  // handing a connection to a caller that may already have given up.
  releaseBlockedUpgrades() {
    const parked = this.parked;
    this.parked = [];
    this.blockUpgrades = false;
    parked.forEach((resume) => resume());
  }

  private finishOpen(request: FakeOpenRequest, name: string, version: number) {
    let data = this.databases.get(name);
    const needsUpgrade = !data || data.version < version;
    if (!data) {
      data = { version, stores: new Map() };
      this.databases.set(name, data);
    }
    const connection = new FakeConnection(data, this);
    request.result = connection;
    if (needsUpgrade) {
      data.version = version;
      request.onupgradeneeded?.();
    }
    this.openConnections.add(connection);
    request.onsuccess?.();
  }
}

let fakeIndexedDB: FakeIndexedDB;

function part(overrides: Partial<RecordingPart> = {}): RecordingPart {
  const recordingId = overrides.recordingId ?? "rec-1";
  const partIndex = overrides.partIndex ?? 0;
  return {
    id: partId(recordingId, partIndex),
    recordingId,
    partIndex,
    blob: new Blob(["chunk"], { type: "audio/webm" }),
    savedAt: 1_000,
    uploaded: false,
    ...overrides,
  };
}

function meta(overrides: Partial<RecordingMeta> = {}): RecordingMeta {
  return {
    recordingId: "rec-1",
    mimeType: "audio/webm",
    startedAt: 1_000,
    durationSecs: 0,
    finalized: false,
    ...overrides,
  };
}

// Builds the v1 shape: a `meta` store keyed by a single "current" row.
function seedLegacyDatabase(legacyPart: RecordingPart) {
  return new Promise<void>((resolve, reject) => {
    const request = fakeIndexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result as FakeConnection;
      const parts = db.createObjectStore("parts", { keyPath: "id" });
      parts.createIndex("recordingId", "recordingId");
      db.createObjectStore("meta", { keyPath: "key" });
    };
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result as FakeConnection;
      const transaction = db.transaction("parts", "readwrite");
      transaction
        .objectStore("parts")
        .put(legacyPart as unknown as Record<string, unknown>);
      transaction.objectStore("meta").put({
        key: "current",
        recordingId: "rec-legacy",
        finalized: false,
        startedAt: 1,
      });
      transaction.oncomplete = () => {
        db.close();
        resolve();
      };
      transaction.onabort = () => reject(transaction.error);
    };
  });
}

describe("recordingStore", () => {
  beforeEach(() => {
    fakeIndexedDB = new FakeIndexedDB();
    vi.stubGlobal("indexedDB", fakeIndexedDB);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns parts in recording order however they were written", async () => {
    await savePart(part({ partIndex: 2 }));
    await savePart(part({ partIndex: 0 }));
    await savePart(part({ partIndex: 1 }));
    await savePart(part({ recordingId: "rec-2", partIndex: 0 }));

    const parts = await getParts("rec-1");

    expect(parts.map((p) => p.partIndex)).toEqual([0, 1, 2]);
    // Scoped by index, so a second tab's take never bleeds in.
    expect(parts.every((p) => p.recordingId === "rec-1")).toBe(true);
  });

  it("marks a stored part uploaded", async () => {
    await savePart(part({ partIndex: 0 }));

    await markPartUploaded(partId("rec-1", 0));

    const [stored] = await getParts("rec-1");
    expect(stored.uploaded).toBe(true);
    expect(stored.blob).toBeInstanceOf(Blob);
  });

  // The upload queue calls this for parts it may have replayed from a
  // previous session — writing a row back for an id that is gone would
  // resurrect a part with no blob.
  it("ignores an unknown id rather than writing a phantom part", async () => {
    await savePart(part({ partIndex: 0 }));

    await markPartUploaded(partId("rec-1", 7));

    expect(await getParts("rec-1")).toHaveLength(1);
  });

  it("offers the newest unfinalized take for recovery", async () => {
    await saveMeta(meta({ recordingId: "old", startedAt: 100 }));
    await saveMeta(meta({ recordingId: "new", startedAt: 300 }));

    expect((await getMeta())?.recordingId).toBe("new");
  });

  // Two tabs: one finished its dump, the other is still going. The
  // unfinished one is the only one worth offering back, even though the
  // finished one is newer.
  it("prefers an older unfinalized take over a newer finalized one", async () => {
    await saveMeta(meta({ recordingId: "live", startedAt: 100 }));
    await saveMeta(
      meta({ recordingId: "done", startedAt: 300, finalized: true }),
    );

    expect((await getMeta())?.recordingId).toBe("live");
  });

  it("falls back to the newest take when every take is finalized", async () => {
    await saveMeta(
      meta({ recordingId: "first", startedAt: 100, finalized: true }),
    );
    await saveMeta(
      meta({ recordingId: "second", startedAt: 300, finalized: true }),
    );

    expect((await getMeta())?.recordingId).toBe("second");
  });

  it("returns null when nothing was ever recorded", async () => {
    expect(await getMeta()).toBeNull();
    expect(await getMetaById("rec-1")).toBeNull();
  });

  // Finalize marks the take it just submitted, not "the newest" — which
  // in a second tab is somebody else's recording.
  it("looks a take up by id without regard to which is newest", async () => {
    await saveMeta(meta({ recordingId: "mine", startedAt: 100 }));
    await saveMeta(meta({ recordingId: "theirs", startedAt: 300 }));

    const mine = await getMetaById("mine");
    expect(mine?.recordingId).toBe("mine");
    expect(mine?.finalized).toBe(false);

    await saveMeta({ ...mine!, finalized: true });
    expect((await getMetaById("mine"))?.finalized).toBe(true);
    // The other tab's take is untouched, so it can still be recovered.
    expect((await getMetaById("theirs"))?.finalized).toBe(false);
  });

  it("clears only the recording it was asked to clear", async () => {
    await savePart(part({ recordingId: "done", partIndex: 0 }));
    await savePart(part({ recordingId: "done", partIndex: 1 }));
    await savePart(part({ recordingId: "other", partIndex: 0 }));
    await saveMeta(meta({ recordingId: "done" }));
    await saveMeta(meta({ recordingId: "other" }));

    await clearRecording("done");

    expect(await getParts("done")).toEqual([]);
    expect(await getMetaById("done")).toBeNull();
    expect(await getParts("other")).toHaveLength(1);
    expect((await getMetaById("other"))?.recordingId).toBe("other");
  });

  it("closes every connection it opens", async () => {
    await savePart(part({ partIndex: 0 }));
    await saveMeta(meta());
    await getParts("rec-1");
    await getMeta();
    await clearRecording("rec-1");

    expect(fakeIndexedDB.openConnections.size).toBe(0);
  });

  it("rejects instead of hanging when IndexedDB is missing", async () => {
    vi.unstubAllGlobals();
    expect(isIndexedDBAvailable()).toBe(false);

    await expect(savePart(part())).rejects.toBeInstanceOf(Error);
  });

  it("rejects and closes the connection when a write aborts", async () => {
    fakeIndexedDB.failWrites = true;

    await expect(savePart(part())).rejects.toThrow("QuotaExceededError");
    expect(fakeIndexedDB.openConnections.size).toBe(0);
  });

  // The zero-loss guarantee: reporting a chunk as persisted on request
  // success would call a write durable that the commit then threw away.
  it("rejects when the commit fails even though the write succeeded", async () => {
    fakeIndexedDB.failCommit = true;

    await expect(savePart(part())).rejects.toThrow("commit failed");
    expect(fakeIndexedDB.openConnections.size).toBe(0);
  });

  it("rejects rather than hanging when another tab blocks the upgrade", async () => {
    fakeIndexedDB.blockUpgrades = true;

    await expect(savePart(part())).rejects.toThrow(
      "IndexedDB upgrade blocked by another tab",
    );
  });

  // The blocking tab can close after we have given up, handing us a
  // connection nobody holds. Leaving it open would block the next upgrade.
  it("closes a connection that arrives after the blocked open gave up", async () => {
    fakeIndexedDB.blockUpgrades = true;
    await expect(savePart(part())).rejects.toThrow(/blocked/);

    fakeIndexedDB.releaseBlockedUpgrades();
    await Promise.resolve();

    expect(fakeIndexedDB.openConnections.size).toBe(0);
    // And the store is usable again once the blocker is gone.
    await savePart(part({ partIndex: 0 }));
    expect(await getParts("rec-1")).toHaveLength(1);
  });

  // v2 re-keys the meta store by recordingId. The parts are the
  // irreplaceable half and live in a store the upgrade does not touch.
  it("keeps the parts and drops the single-row meta store on upgrade", async () => {
    const legacyPart = part({ recordingId: "rec-legacy", partIndex: 0 });
    await seedLegacyDatabase(legacyPart);

    const parts = await getParts("rec-legacy");
    expect(parts.map((p) => p.id)).toEqual([legacyPart.id]);
    // The v1 "current" row is gone rather than migrated, so recovery does
    // not offer a take keyed by something the new code cannot address.
    expect(await getMeta()).toBeNull();

    await saveMeta(meta({ recordingId: "rec-legacy" }));
    expect((await getMetaById("rec-legacy"))?.recordingId).toBe("rec-legacy");
  });

  it("keys a part by recording and index", () => {
    expect(partId("rec-1", 3)).toBe("rec-1:3");
  });
});
