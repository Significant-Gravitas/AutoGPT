import * as Sentry from "@sentry/nextjs";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { readSseFrames } from "../sseClient";
import {
  compareShadowWithSession,
  getShadowLog,
  observeShadowEnd,
  observeShadowEntry,
  resetShadows,
  setStreamShadowRate,
} from "../turnShadow";
import { loadRecordedTurn } from "./recordedTurns";

vi.mock("@sentry/nextjs", () => ({ captureMessage: vi.fn() }));

const SESSION = "s1";
const turn = loadRecordedTurn("baseline-tool-turn");
const body = turn.sse.join("") + "data: [DONE]\n\n";
// The turn up to the first text block's second delta, as a server cut leaves it.
const cutBody = turn.sse.slice(0, 7).join("");

beforeEach(() => {
  resetShadows();
  setStreamShadowRate(1);
  vi.mocked(Sentry.captureMessage).mockClear();
});

/** One connection's worth of SSE, as the turn stream hands it to the shadow. */
async function feed(sse: string, { session = SESSION, byClient = false } = {}) {
  let turnId: string | null = null;
  await readSseFrames(new Response(sse).body!, (frame) => {
    if (frame.kind !== "entry") return;
    turnId = frame.entry.turn;
    observeShadowEntry(session, frame.entry);
  });
  observeShadowEnd(session, turnId, byClient);
}

async function stream(sse: string) {
  await feed(sse);
  await vi.waitFor(() =>
    expect(getShadowLog(SESSION)?.status).toBe("finished"),
  );
}

function driftKinds() {
  return vi
    .mocked(Sentry.captureMessage)
    .mock.calls.map(
      ([, hint]) =>
        (hint as { tags: { stream_drift: string } }).tags.stream_drift,
    );
}

describe("the stream shadow", () => {
  it("reports nothing for a turn that matches", async () => {
    await stream(body);
    compareShadowWithSession(SESSION, {
      messages: turn.rows,
      active_stream: null,
    });
    // The digest check is async; give it the time the tampered case needs.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(getShadowLog(SESSION)?.rows).toHaveLength(3);
    expect(driftKinds()).toEqual([]);
  });

  it("reports a checkpoint whose digest the folded rows do not reproduce", async () => {
    const tampered = body.replace(/"digest": "[0-9a-f]+"/, '"digest": "0000"');
    expect(tampered).not.toBe(body);
    await stream(tampered);
    await vi.waitFor(() => expect(driftKinds()).toEqual(["checkpoint"]));
  });

  it("reports a finished turn whose persisted rows lack what streamed", async () => {
    await stream(body);
    const withoutReply = turn.rows.filter((row) => row.sequence !== 3);
    compareShadowWithSession(SESSION, {
      messages: withoutReply,
      active_stream: null,
    });
    expect(driftKinds()).toEqual(["finish"]);
  });

  it("does not spend the finish check on a view whose window misses the turn", async () => {
    await stream(body);
    const afterTurnStart = turn.rows.filter((row) => (row.sequence ?? 0) > 1);
    compareShadowWithSession(SESSION, {
      messages: afterTurnStart,
      active_stream: null,
      has_more_messages: true,
    });
    expect(driftKinds()).toEqual([]);
    const withoutReply = turn.rows.filter((row) => row.sequence !== 3);
    compareShadowWithSession(SESSION, {
      messages: withoutReply,
      active_stream: null,
    });
    expect(driftKinds()).toEqual(["finish"]);
  });

  it("keeps a finished turn's check when the next turn starts before the view arrives", async () => {
    await stream(body);
    const next = body.replaceAll("drift-turn:", "next-turn:");
    await feed(next);
    await vi.waitFor(() =>
      expect(getShadowLog(SESSION)?.turnId).toBe("next-turn"),
    );
    const withoutReply = turn.rows.filter((row) => row.sequence !== 3);
    compareShadowWithSession(SESSION, {
      messages: withoutReply,
      active_stream: { turn_id: "next-turn" },
    });
    expect(driftKinds()).toEqual(["finish"]);
  });

  it("keeps a shadow for the most recent sessions only", async () => {
    for (const id of ["s-1", "s-2", "s-3", "s-4", "s-5", "s-6"]) {
      await feed(body, { session: id });
      await vi.waitFor(() => expect(getShadowLog(id)?.status).toBe("finished"));
    }
    expect(getShadowLog("s-1")).toBeNull();
    expect(getShadowLog("s-6")?.status).toBe("finished");
  });

  it("leaves a session the flag does not sample untouched", async () => {
    setStreamShadowRate(0);
    await feed(body);
    expect(getShadowLog(SESSION)).toBeNull();
  });

  it("reports a turn the server finished while its stream ended before finish", async () => {
    await feed(cutBody);
    expect(getShadowLog(SESSION)?.rows).toHaveLength(1);
    compareShadowWithSession(SESSION, {
      messages: turn.rows,
      active_stream: { turn_id: "drift-turn" },
    });
    expect(driftKinds()).toEqual([]);
    compareShadowWithSession(SESSION, {
      messages: turn.rows,
      active_stream: null,
    });
    expect(driftKinds()).toEqual(["abandoned"]);
  });

  it("reports a turn the client stopped as stopped", async () => {
    await feed(cutBody, { byClient: true });
    compareShadowWithSession(SESSION, { messages: [], active_stream: null });
    expect(driftKinds()).toEqual(["stopped"]);
  });

  it("checks a cut turn as finished once a resume delivers its end", async () => {
    await feed(cutBody);
    await stream(body);
    compareShadowWithSession(SESSION, {
      messages: turn.rows,
      active_stream: null,
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(driftKinds()).toEqual([]);
  });

  it("waits while the session view still runs the turn", async () => {
    await stream(body);
    compareShadowWithSession(SESSION, {
      messages: [],
      active_stream: { turn_id: "drift-turn" },
    });
    expect(driftKinds()).toEqual([]);
  });

  it("reports nothing for a replay that opens past the turn's start", async () => {
    const tail = turn.sse.slice(
      turn.sse.findIndex((f) => f.includes("tool-output-available")),
    );
    await feed(tail.join(""));
    await vi.waitFor(() =>
      expect(getShadowLog(SESSION)?.status).toBe("finished"),
    );
    compareShadowWithSession(SESSION, {
      messages: turn.rows,
      active_stream: null,
    });
    expect(getShadowLog(SESSION)?.protocolErrors.length).toBeGreaterThan(0);
    expect(driftKinds()).toEqual([]);
  });
});
