// Local-first storage for brain-dump audio. Every MediaRecorder chunk lands
// here BEFORE it is offered to the network, and the store is only cleared
// once the server has reported the dump completed. That ordering is the
// whole zero-loss guarantee: a crash, a refresh, a dead tunnel or a closed
// laptop can cost an upload, never a recording.
//
// Hand-rolled rather than pulling in `idb`: the surface used here is four
// object-store calls, and the wizard should not carry a dependency for it.

const DB_NAME = "autogpt-onboarding-brain-dump";
// v2 re-keys the meta store by recordingId. v1 kept a single "current"
// row, so two tabs recording at once clobbered each other's metadata and
// recovery could only ever find the newer take.
const DB_VERSION = 2;
const PARTS_STORE = "parts";
const META_STORE = "meta";

export interface RecordingPart {
  id: string;
  recordingId: string;
  partIndex: number;
  blob: Blob;
  savedAt: number;
  uploaded: boolean;
}

export interface RecordingMeta {
  recordingId: string;
  mimeType: string;
  startedAt: number;
  durationSecs: number;
  finalized: boolean;
}

export function isIndexedDBAvailable(): boolean {
  return typeof window !== "undefined" && "indexedDB" in window;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = function handleUpgrade() {
      const db = request.result;
      if (!db.objectStoreNames.contains(PARTS_STORE)) {
        const store = db.createObjectStore(PARTS_STORE, { keyPath: "id" });
        store.createIndex("recordingId", "recordingId");
      }
      // Recreate rather than migrate: the only thing lost is which take
      // was in flight, and the parts (the irreplaceable bit) live in a
      // different store and are untouched.
      if (db.objectStoreNames.contains(META_STORE)) {
        db.deleteObjectStore(META_STORE);
      }
      db.createObjectStore(META_STORE, { keyPath: "recordingId" });
    };
    // Another tab still holding v1 open blocks the upgrade, and without
    // this the promise never settles — the recorder would hang before it
    // ever reached the mic. Failing here instead falls back to
    // upload-only, which is degraded but alive.
    let blocked = false;
    request.onsuccess = () => {
      // The other tab can close after we have already given up, at which
      // point the upgrade completes and hands us a connection nobody is
      // holding. Closing it keeps it from blocking the *next* upgrade.
      if (blocked) {
        request.result.close();
        return;
      }
      resolve(request.result);
    };
    request.onerror = () => reject(request.error);
    request.onblocked = () => {
      blocked = true;
      reject(new Error("IndexedDB upgrade blocked by another tab"));
    };
  });
}

// Safari in private mode, hardened browser profiles and the test
// environment all lack a usable IndexedDB. Losing local persistence
// degrades the zero-loss guarantee to "the upload queue is the only
// backup" — worth continuing on, not worth crashing the step for.
class IndexedDBUnavailableError extends Error {}

function runTransaction<T>(
  storeName: string,
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  if (!isIndexedDBAvailable()) {
    return Promise.reject(new IndexedDBUnavailableError());
  }
  return openDB().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const transaction = db.transaction(storeName, mode);
        const request = run(transaction.objectStore(storeName));
        let result: T;
        request.onsuccess = () => {
          result = request.result;
        };
        // Settled on the transaction, not the request. `onsuccess` fires
        // before the transaction commits, so resolving there would report
        // a chunk as persisted that a later abort (a quota error on a big
        // blob, say) silently throws away — precisely the thing the
        // zero-loss guarantee at the top of this file promises not to do.
        transaction.oncomplete = () => {
          db.close();
          resolve(result);
        };
        // Closing on every exit, not just the happy one: a leaked
        // connection also blocks the next version upgrade.
        transaction.onabort = () => {
          db.close();
          reject(transaction.error ?? request.error);
        };
        transaction.onerror = () => {
          db.close();
          reject(transaction.error ?? request.error);
        };
      }),
  );
}

export async function savePart(part: RecordingPart) {
  await runTransaction(PARTS_STORE, "readwrite", (store) => store.put(part));
}

export async function markPartUploaded(id: string) {
  const existing = await runTransaction<RecordingPart | undefined>(
    PARTS_STORE,
    "readonly",
    (store) => store.get(id),
  );
  if (!existing) return;
  await savePart({ ...existing, uploaded: true });
}

export async function getParts(recordingId: string) {
  const all = await runTransaction<RecordingPart[]>(
    PARTS_STORE,
    "readonly",
    (store) => store.index("recordingId").getAll(recordingId),
  );
  return all.sort((a, b) => a.partIndex - b.partIndex);
}

export async function saveMeta(meta: RecordingMeta) {
  await runTransaction(META_STORE, "readwrite", (store) => store.put(meta));
}

// The take a returning user should be offered. Two tabs can each have a
// recording in flight, so prefer the newest *unfinalized* one — a
// finished take is not something to recover from.
export async function getMeta() {
  const all = await runTransaction<RecordingMeta[]>(
    META_STORE,
    "readonly",
    (store) => store.getAll(),
  );
  const unfinalized = all.filter((meta) => !meta.finalized);
  const candidates = unfinalized.length > 0 ? unfinalized : all;
  return candidates.sort((a, b) => b.startedAt - a.startedAt)[0] ?? null;
}

// The finalize path knows exactly which take it just submitted, and must
// say so: `getMeta()` answers "what should we offer to recover", which in
// a second tab is a different, newer take. Marking that one finalized
// would strand the take that actually completed.
export async function getMetaById(recordingId: string) {
  const meta = await runTransaction<RecordingMeta | undefined>(
    META_STORE,
    "readonly",
    (store) => store.get(recordingId),
  );
  return meta ?? null;
}

// Called only after the server reports `completed` — an unfinalized
// recording must survive every other kind of exit.
export async function clearRecording(recordingId: string) {
  const parts = await getParts(recordingId);
  await Promise.all(
    parts.map((part) =>
      runTransaction(PARTS_STORE, "readwrite", (store) =>
        store.delete(part.id),
      ),
    ),
  );
  await runTransaction(META_STORE, "readwrite", (store) =>
    store.delete(recordingId),
  );
}

export function partId(recordingId: string, partIndex: number) {
  return `${recordingId}:${partIndex}`;
}
