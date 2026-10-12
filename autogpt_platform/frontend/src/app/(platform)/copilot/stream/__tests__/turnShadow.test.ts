import * as Sentry from "@sentry/nextjs";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  compareShadowWithSession,
  createShadowFetch,
  getShadowLog,
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

async function stream(sse: string) {
  const shadowFetch = createShadowFetch(SESSION, async () => new Response(sse));
  const response = await shadowFetch("http://x/stream", { method: "POST" });
  const text = await response.text();
  await vi.waitFor(() =>
    expect(getShadowLog(SESSION)?.status).toBe("finished"),
  );
  return text;
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
  it("hands the chat the stream untouched and reports nothing for a turn that matches", async () => {
    expect(await stream(body)).toBe(body);
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
    const nextTurn = createShadowFetch(SESSION, async () => new Response(next));
    await (await nextTurn("http://x/stream", {})).text();
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
      const shadowFetch = createShadowFetch(id, async () => new Response(body));
      await (await shadowFetch("http://x/stream", {})).text();
      await vi.waitFor(() => expect(getShadowLog(id)?.status).toBe("finished"));
    }
    expect(getShadowLog("s-1")).toBeNull();
    expect(getShadowLog("s-6")?.status).toBe("finished");
  });

  it("leaves a session the flag does not sample untouched", async () => {
    setStreamShadowRate(0);
    const original = new Response(body);
    const shadowFetch = createShadowFetch(SESSION, async () => original);
    expect(await shadowFetch("http://x/stream", {})).toBe(original);
    expect(getShadowLog(SESSION)).toBeNull();
  });

  it("reports a turn the server finished while its stream ended before finish", async () => {
    const cut = createShadowFetch(SESSION, async () => new Response(cutBody));
    await (await cut("http://x/stream", {})).text();
    await vi.waitFor(() => expect(getShadowLog(SESSION)?.rows).toHaveLength(1));
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
    const encoder = new TextEncoder();
    const source = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(cutBody));
      },
    });
    const abort = new AbortController();
    const shadowFetch = createShadowFetch(
      SESSION,
      async () => new Response(source),
    );
    await shadowFetch("http://x/stream", { signal: abort.signal });
    await vi.waitFor(() => expect(getShadowLog(SESSION)?.rows).toHaveLength(1));
    abort.abort();
    await vi.waitFor(() => {
      compareShadowWithSession(SESSION, { messages: [], active_stream: null });
      expect(driftKinds()).toEqual(["stopped"]);
    });
  });

  it("checks a cut turn as finished once a resume delivers its end", async () => {
    const cut = createShadowFetch(SESSION, async () => new Response(cutBody));
    await (await cut("http://x/stream", {})).text();
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
    const shadowFetch = createShadowFetch(
      SESSION,
      async () => new Response(tail.join("")),
    );
    await (await shadowFetch("http://x/stream", {})).text();
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

  it("stops reading when the chat cancels its branch, so the connection closes", async () => {
    const cancelled = vi.fn();
    const source = new ReadableStream<Uint8Array>({
      pull: () => new Promise(() => {}),
      cancel: cancelled,
    });
    const shadowFetch = createShadowFetch(
      SESSION,
      async () => new Response(source),
    );
    const response = await shadowFetch("http://x/stream", {});
    await response.body!.cancel();
    await vi.waitFor(() => expect(cancelled).toHaveBeenCalled());
  });
});
