import * as Sentry from "@sentry/nextjs";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import { getTurnRuntime, resetTurnRuntimes } from "../turnRuntime";
import {
  compareShadowWithSession,
  createShadowFetch,
  getShadowLog,
  resetShadows,
  setStreamShadowRate,
} from "../turnShadow";
import { fakeBackend, type FakeBackend } from "./fakeBackend";
import { loadRecordedTurn } from "./recordedTurns";

// The real SDK, so each report is normalised as Sentry stores it: arrays nested
// past `normalizeDepth` (3) arrive as "[Array]".
const events: Sentry.ErrorEvent[] = [];
const toolTurn = loadRecordedTurn("baseline-tool-turn");
const SECOND_TEXT = "It says Example Domain.";

beforeAll(() => {
  Sentry.init({
    dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
    beforeSend(event) {
      events.push(event);
      return null;
    },
  });
});

beforeEach(() => {
  events.length = 0;
});

describe("a drift report reaching Sentry", () => {
  let backend: FakeBackend;

  beforeEach(() => {
    vi.useFakeTimers();
    backend = fakeBackend(toolTurn);
    getTurnRuntime("session-1", {
      baseUrl: () => "http://backend.test",
      headers: async () => ({}),
      fetch: backend.fetch,
      fetchSession: async () => backend.view(),
      toast: vi.fn(),
    });
  });

  afterEach(() => {
    resetTurnRuntimes();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("names the fields of a turn whose saved rows differ from the stream", async () => {
    void getTurnRuntime("session-1").send({ text: "hi" }, undefined);
    await vi.advanceTimersByTimeAsync(0);
    backend.publish(toolTurn.sse.length - 1);
    await vi.advanceTimersByTimeAsync(100);
    const rows = toolTurn.rows.map((row) =>
      row.content === SECOND_TEXT ? { ...row, content: "It says Other." } : row,
    );
    vi.spyOn(backend, "view").mockImplementation(() => ({
      messages: rows.map((row, sequence) => ({ ...row, sequence })),
      has_more_messages: false,
      active_stream: null,
    }));
    backend.publish();
    await vi.advanceTimersByTimeAsync(1_000);

    await vi.waitFor(() =>
      expect(driftExtras("finish")).toEqual([
        expect.objectContaining({
          sessionId: "session-1",
          diffs: ["2: content"],
        }),
      ]),
    );
  });

  it("names the field a refused send failed validation on, without its input", async () => {
    backend.respond(() => ({
      status: 422,
      body: {
        message: "Invalid data for POST /api/chat/sessions/session-1/stream",
        detail: [
          {
            type: "missing",
            loc: ["body", "message"],
            msg: "Field required",
            input: { text: "the user's words" },
          },
        ],
      },
    }));
    void getTurnRuntime("session-1").send({ text: "hi" }, undefined);
    await vi.advanceTimersByTimeAsync(1_000);

    await vi.waitFor(() =>
      expect(driftExtras("refused")).toEqual([
        expect.objectContaining({
          status: 422,
          body: expect.objectContaining({
            detail: ["body.message: Field required"],
          }),
        }),
      ]),
    );
    expect(JSON.stringify(events)).not.toContain("the user's words");
  });
});

describe("a shadow drift report reaching Sentry", () => {
  beforeEach(() => {
    resetShadows();
    setStreamShadowRate(1);
  });

  it("names the fields of a turn whose saved rows differ from the stream", async () => {
    const body = toolTurn.sse.join("") + "data: [DONE]\n\n";
    const shadowFetch = createShadowFetch("s1", async () => new Response(body));
    await (await shadowFetch("http://x/stream", { method: "POST" })).text();
    await vi.waitFor(() => expect(getShadowLog("s1")?.status).toBe("finished"));
    compareShadowWithSession("s1", {
      messages: toolTurn.rows.map((row) =>
        row.content === SECOND_TEXT ? { ...row, content: "Other." } : row,
      ),
      active_stream: null,
    });

    await vi.waitFor(() =>
      expect(driftExtras("finish")).toEqual([
        expect.objectContaining({ diffs: ["2: content"] }),
      ]),
    );
  });
});

function driftExtras(kind: string) {
  return events
    .filter((event) => event.tags?.stream_drift === kind)
    .map((event) => event.extra);
}
