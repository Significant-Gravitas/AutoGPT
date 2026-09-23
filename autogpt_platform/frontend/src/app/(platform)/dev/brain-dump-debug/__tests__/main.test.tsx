import { getGetBrainDumpStatusMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
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

// happy-dom has no IndexedDB, so the recording store is an in-memory
// double. Everything above it — the snapshot hook, the panels — is real.
const { store, flagStatus } = vi.hoisted(() => ({
  store: {
    available: true,
    meta: null as FakeMeta | null,
    parts: [] as FakePart[],
  },
  flagStatus: { enabled: true as boolean, ready: true },
}));

vi.mock(
  "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/recordingStore",
  () => ({
    isIndexedDBAvailable: () => store.available,
    getMeta: async () => store.meta,
    getParts: async (recordingId: string) =>
      store.parts.filter((part) => part.recordingId === recordingId),
  }),
);

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "onboarding-brain-dump"
        ? flagStatus
        : actual.useFlagStatus(flag as never),
  };
});

const notFoundMock = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/dev/brain-dump-debug",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

import BrainDumpDebugPage from "../page";

let statusCalls: number;

function panel(title: string) {
  const section = screen.getByText(title).closest("section");
  if (!section) throw new Error(`no panel titled ${title}`);
  return within(section);
}

function renderPage() {
  try {
    render(<BrainDumpDebugPage />);
  } catch {
    // The mocked notFound() throws the way Next's does; the assertions
    // below are what the test actually cares about.
  }
}

beforeEach(() => {
  statusCalls = 0;
  store.available = true;
  store.meta = null;
  store.parts = [];
  flagStatus.enabled = true;
  flagStatus.ready = true;
  notFoundMock.mockClear();
  window.sessionStorage.clear();
  server.use(
    getGetBrainDumpStatusMockHandler200(() => {
      statusCalls += 1;
      return { status: "transcribing", input_mode: "voice", has_audio: true };
    }),
  );
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("brain dump debug page — gating", () => {
  it("404s in production even with the flag on", () => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "prod");

    renderPage();

    expect(notFoundMock).toHaveBeenCalled();
    expect(screen.queryByText("Brain dump debug")).toBeNull();
    expect(statusCalls).toBe(0);
  });

  it("404s when the onboarding-brain-dump flag is off", () => {
    flagStatus.enabled = false;

    renderPage();

    expect(notFoundMock).toHaveBeenCalled();
    expect(screen.queryByText("Brain dump debug")).toBeNull();
    expect(statusCalls).toBe(0);
  });

  it("waits with skeletons instead of 404ing while the flag is unresolved", async () => {
    // Short-circuiting to notFound() before LaunchDarkly answers would 404
    // users who actually have the flag on.
    flagStatus.ready = false;
    flagStatus.enabled = false;

    const { container } = render(<BrainDumpDebugPage />);

    expect(notFoundMock).not.toHaveBeenCalled();
    expect(container.querySelectorAll(".animate-pulse")).toHaveLength(3);
    expect(screen.queryByText("Brain dump debug")).toBeNull();
    // The status query stays disabled until the page is known to be
    // available, so nothing is polled while the flag is in flight.
    expect(statusCalls).toBe(0);
  });

  it("renders every panel once the flag resolves on outside production", async () => {
    renderPage();

    expect(await screen.findByText("Brain dump debug")).toBeDefined();
    expect(screen.getByText("dev only")).toBeDefined();
    expect(screen.getByText("IndexedDB recording state")).toBeDefined();
    expect(screen.getByText("Upload queue")).toBeDefined();
    expect(screen.getByText("Server status")).toBeDefined();
    expect(screen.getByText("Transcript & extraction JSON")).toBeDefined();
    expect(screen.getByText("Timing waterfall")).toBeDefined();
    expect(notFoundMock).not.toHaveBeenCalled();
  });
});

describe("brain dump debug page — recording state", () => {
  it("shows the stored meta row, the parts table and the byte totals", async () => {
    store.meta = {
      recordingId: "rec-42",
      mimeType: "audio/webm;codecs=opus",
      startedAt: new Date(2024, 2, 1, 13, 5, 9).getTime(),
      durationSecs: 91.5,
      finalized: false,
    };
    store.parts = [
      {
        id: "rec-42:0",
        recordingId: "rec-42",
        partIndex: 0,
        blob: new Blob(["a".repeat(2048)]),
        savedAt: 1,
        uploaded: true,
      },
      {
        id: "rec-42:1",
        recordingId: "rec-42",
        partIndex: 1,
        blob: new Blob(["b".repeat(1024)]),
        savedAt: 2,
        uploaded: false,
      },
    ];

    renderPage();

    const recording = panel("IndexedDB recording state");
    expect(await recording.findByText("rec-42")).toBeDefined();
    expect(recording.getByText("audio/webm;codecs=opus")).toBeDefined();
    expect(recording.getByText("91.5 s")).toBeDefined();
    expect(recording.getByText(/^13.05.09$/)).toBeDefined();
    expect(recording.getByText("2 parts")).toBeDefined();
    expect(recording.getByText("3 KB total")).toBeDefined();

    const rows = recording.getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText("2 KB")).toBeDefined();
    expect(within(rows[0]).getByText("yes")).toBeDefined();
    expect(within(rows[1]).getByText("1 KB")).toBeDefined();
    expect(within(rows[1]).getByText("no")).toBeDefined();

    expect(recording.getByRole("button", { name: /Refresh/ })).toBeDefined();
  });

  it("explains an empty store instead of rendering an empty table", async () => {
    renderPage();

    const recording = panel("IndexedDB recording state");
    expect(await recording.findByText(/No meta row stored/)).toBeDefined();
    expect(recording.getByText("0 parts")).toBeDefined();
    expect(recording.getByText("0 B total")).toBeDefined();
    expect(recording.queryByRole("table")).toBeNull();
  });

  it("says so when the browser has no IndexedDB", async () => {
    store.available = false;

    renderPage();

    const recording = panel("IndexedDB recording state");
    expect(
      await recording.findByText("IndexedDB is not available in this browser."),
    ).toBeDefined();
  });
});

describe("brain dump debug page — upload queue", () => {
  it("counts the parts that have not been marked uploaded", async () => {
    store.meta = {
      recordingId: "rec-42",
      mimeType: "audio/webm",
      startedAt: 1,
      durationSecs: 60,
      finalized: false,
    };
    store.parts = [
      {
        id: "rec-42:0",
        recordingId: "rec-42",
        partIndex: 0,
        blob: new Blob(["a".repeat(100)]),
        savedAt: 1,
        uploaded: true,
      },
      {
        id: "rec-42:1",
        recordingId: "rec-42",
        partIndex: 1,
        blob: new Blob(["b".repeat(250)]),
        savedAt: 2,
        uploaded: false,
      },
      {
        id: "rec-42:2",
        recordingId: "rec-42",
        partIndex: 2,
        blob: new Blob(["c".repeat(250)]),
        savedAt: 3,
        uploaded: false,
      },
    ];

    renderPage();

    const queue = panel("Upload queue");
    expect(await queue.findByText("2 pending")).toBeDefined();
    // pending count, uploaded count, pending bytes
    expect(queue.getByText("2")).toBeDefined();
    expect(queue.getByText("1")).toBeDefined();
    expect(queue.getByText("500 B")).toBeDefined();
  });

  it("reports the queue as drained when every part is uploaded", async () => {
    store.meta = {
      recordingId: "rec-42",
      mimeType: "audio/webm",
      startedAt: 1,
      durationSecs: 60,
      finalized: true,
    };
    store.parts = [
      {
        id: "rec-42:0",
        recordingId: "rec-42",
        partIndex: 0,
        blob: new Blob(["a"]),
        savedAt: 1,
        uploaded: true,
      },
    ];

    renderPage();

    const queue = panel("Upload queue");
    expect(await queue.findByText("drained")).toBeDefined();
    expect(queue.getByText("0 B")).toBeDefined();
  });
});

describe("brain dump debug page — transcript", () => {
  it("keeps the 'not exposed by any endpoint' warning and offers no preview", async () => {
    renderPage();

    const transcript = panel("Transcript & extraction JSON");
    expect(
      await transcript.findByText("Not exposed by any current endpoint"),
    ).toBeDefined();
    expect(transcript.getByText(/None captured/)).toBeDefined();
  });
});
