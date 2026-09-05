import {
  getFinalizeBrainDumpMockHandler200,
  getGetBrainDumpStatusMockHandler200,
  getGetBrainDumpStatusMockHandler401,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import type { DumpStatusResponse } from "@/app/api/__generated__/models/dumpStatusResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const FINALIZE_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/finalize";

interface FakeMeta {
  recordingId: string;
  mimeType: string;
  startedAt: number;
  durationSecs: number;
  finalized: boolean;
}

const { store, flagStatus } = vi.hoisted(() => ({
  store: {
    meta: null as FakeMeta | null,
  },
  flagStatus: { enabled: true, ready: true },
}));

vi.mock(
  "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/recordingStore",
  () => ({
    isIndexedDBAvailable: () => true,
    getMeta: async () => store.meta,
    getParts: async () => [],
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
    throw new Error("NEXT_NOT_FOUND");
  },
}));

import BrainDumpDebugPage from "../page";

const STORED_META: FakeMeta = {
  recordingId: "rec-42",
  mimeType: "audio/webm;codecs=opus",
  startedAt: 1_700_000_000_000,
  durationSecs: 91.5,
  finalized: false,
};

function panel(title: string) {
  const section = screen.getByText(title).closest("section");
  if (!section) throw new Error(`no panel titled ${title}`);
  return within(section);
}

function serveStatus(...responses: DumpStatusResponse[]) {
  let call = 0;
  server.use(
    getGetBrainDumpStatusMockHandler200(() => {
      const response = responses[Math.min(call, responses.length - 1)];
      call += 1;
      return response;
    }),
  );
}

// Records the finalize request bodies so the debug call can be checked
// against the meta row it claims to replay.
function captureFinalize(response: {
  status: "completed" | "failed";
  input_mode: "voice" | "typed" | "skipped";
  transcript_preview?: string;
  error_code?: string;
}) {
  const bodies: unknown[] = [];
  server.use(
    getFinalizeBrainDumpMockHandler200(async (info) => {
      bodies.push(await info.request.json());
      return response;
    }),
  );
  return bodies;
}

beforeEach(() => {
  store.meta = null;
  window.sessionStorage.clear();
  serveStatus({ status: "transcribing", input_mode: "voice", has_audio: true });
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("server status panel", () => {
  it("renders the polled status fields", async () => {
    serveStatus({
      status: "extracting",
      input_mode: "typed",
      has_audio: false,
      error_code: null,
    });

    render(<BrainDumpDebugPage />);

    const status = panel("Server status");
    expect(await status.findByText("extracting")).toBeDefined();
    expect(status.getByText("typed")).toBeDefined();
    // has_audio false renders as "no", error_code null as an em dash.
    expect(status.getByText("no")).toBeDefined();
    expect(status.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("explains a failing status request rather than showing a blank panel", async () => {
    server.use(getGetBrainDumpStatusMockHandler401());

    render(<BrainDumpDebugPage />);

    const status = panel("Server status");
    expect(
      await status.findByText(/the endpoint 404s when ONBOARDING_BRAIN_DUMP/),
    ).toBeDefined();
  });

  it("links the download at the proxied recording endpoint", async () => {
    render(<BrainDumpDebugPage />);

    const link = await screen.findByRole("link", {
      name: /Download server-side recording/,
    });
    expect(link.getAttribute("href")).toBe(
      "/api/proxy/api/onboarding/brain-dump/recording",
    );
  });
});

describe("debug finalize", () => {
  it("cannot run without a stored recording", async () => {
    render(<BrainDumpDebugPage />);

    const button = await screen.findByRole("button", {
      name: "Run finalize on the stored recording",
    });
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });

  it("replays the stored meta row and shows the response it got back", async () => {
    store.meta = STORED_META;
    const bodies = captureFinalize({
      status: "completed",
      input_mode: "voice",
      transcript_preview: "I spend every Monday rebuilding the same report.",
    });

    render(<BrainDumpDebugPage />);

    const button = await screen.findByRole("button", {
      name: "Run finalize on the stored recording",
    });
    await vi.waitFor(() =>
      expect((button as HTMLButtonElement).disabled).toBe(false),
    );
    await userEvent.click(button);

    expect(await screen.findByText(/^round-trip \d/)).toBeDefined();
    expect(bodies).toEqual([
      {
        recording_id: "rec-42",
        input_mode: "voice",
        duration_secs: 91.5,
        mime_type: "audio/webm;codecs=opus",
      },
    ]);

    // The finalize response, not the polled status, which is "transcribing".
    const status = panel("Server status");
    expect(status.getByText("completed")).toBeDefined();

    const transcript = panel("Transcript & extraction JSON");
    expect(
      transcript.getByText("I spend every Monday rebuilding the same report."),
    ).toBeDefined();
    expect(transcript.queryByText(/None captured/)).toBeNull();

    // The round-trip is also surfaced as the whole post-Done budget.
    expect(
      panel("Timing waterfall").getByText(/^finalize round-trip \d/),
    ).toBeDefined();
  });

  it("surfaces the server's message when finalize fails", async () => {
    store.meta = STORED_META;
    server.use(
      http.post(FINALIZE_URL, () =>
        HttpResponse.json(
          { detail: "recording already consumed" },
          {
            status: 409,
          },
        ),
      ),
    );

    render(<BrainDumpDebugPage />);

    const button = await screen.findByRole("button", {
      name: "Run finalize on the stored recording",
    });
    await vi.waitFor(() =>
      expect((button as HTMLButtonElement).disabled).toBe(false),
    );
    await userEvent.click(button);

    expect(
      await screen.findByText(/Finalize failed: recording already consumed/),
    ).toBeDefined();
    // A failed call must not leave a stale preview behind.
    expect(
      panel("Transcript & extraction JSON").getByText(/None captured/),
    ).toBeDefined();
  });
});

describe("timing waterfall", () => {
  it("measures transcribe once the poll has seen both ends of the phase", async () => {
    serveStatus(
      { status: "transcribing", input_mode: "voice" },
      { status: "completed", input_mode: "voice" },
    );

    render(<BrainDumpDebugPage />);

    const timing = panel("Timing waterfall");
    // Transcribe and extract both start out unobserved.
    expect(timing.getAllByText(/this page did not observe both/)).toHaveLength(
      2,
    );

    // The second poll lands a second later and closes the transcribe phase.
    expect(
      await timing.findByText(
        /first "recording_uploaded\/transcribing" . first "transcribed"/,
        {},
        { timeout: 4000 },
      ),
    ).toBeDefined();
    expect(timing.getByText("measured")).toBeDefined();
    // Extract never saw `transcribed`, so it stays unmeasured.
    expect(timing.getAllByText(/this page did not observe both/)).toHaveLength(
      1,
    );
  });

  it("keeps stages the page cannot time labelled unmeasured", async () => {
    render(<BrainDumpDebugPage />);

    const timing = panel("Timing waterfall");
    expect(
      await timing.findByText(/no recording currently in IndexedDB/),
    ).toBeDefined();
    expect(
      timing.getByText(/0 of 0 part\(s\) are still not marked uploaded/),
    ).toBeDefined();
    expect(timing.getAllByText("unmeasured")).toHaveLength(5);
    expect(timing.getAllByText("no budget")).toHaveLength(3);
  });
});

describe("intro handoff", () => {
  it("shows the pending intro path from sessionStorage in the header and the waterfall", async () => {
    window.sessionStorage.setItem("autogpt:onboarding-intro-path", "B");

    render(<BrainDumpDebugPage />);

    expect(await screen.findByText("intro path B")).toBeDefined();
    expect(
      panel("Timing waterfall").getByText(/Pending handoff path: B/),
    ).toBeDefined();
  });

  it("points at the sessionStorage key when no handoff is queued", async () => {
    render(<BrainDumpDebugPage />);

    const timing = panel("Timing waterfall");
    expect(
      await timing.findByText(/autogpt:onboarding-intro-path/),
    ).toBeDefined();
    expect(screen.queryByText(/^intro path/)).toBeNull();
  });
});
