import type { UIMessage } from "ai";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";

import { createTailRenderer } from "../renderTail";
import {
  ANOMALY_TOAST,
  LIVENESS_MS,
  ROTATE_AFTER_MS,
  TurnRuntime,
  type RuntimeDeps,
  type RuntimeSnapshot,
} from "../turnRuntime";
import { fakeBackend, type FakeBackend } from "./fakeBackend";
import { loadRecordedTurn } from "./recordedTurns";

const toolTurn = loadRecordedTurn("baseline-tool-turn");
const PROMPT = String(toolTurn.rows[0].content);
// 9-0 ends the first text block, 12-0 opens the tool call, 20-0 is the checkpoint.
const FIRST_BLOCK_DONE = 9;
const TOOL_CALL_OPEN = 12;
const FIRST_TEXT = "Let me fetch that page.";
const SECOND_TEXT = "It says Example Domain.";

let backend: FakeBackend;
let runtime: TurnRuntime;
let toast: Mock<RuntimeDeps["toast"]>;
let report: Mock<RuntimeDeps["report"]>;

beforeEach(() => {
  vi.useFakeTimers();
  setVisibility("visible");
  backend = fakeBackend(toolTurn);
  toast = vi.fn<RuntimeDeps["toast"]>();
  report = vi.fn<RuntimeDeps["report"]>();
  runtime = new TurnRuntime("session-1", {
    baseUrl: () => "http://backend.test",
    headers: async () => ({}),
    fetch: backend.fetch,
    fetchSession: async () => backend.view(),
    toast,
    report,
  });
});

afterEach(() => {
  runtime.dispose();
  vi.restoreAllMocks();
  vi.useRealTimers();
  setVisibility("visible");
});

describe("§1.7 — every disconnect cause is expected but one", () => {
  it("a tab hidden when its stream drops resumes on return, silently", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    setVisibility("hidden");
    backend.connections[0].fail();
    await advance(60_000);
    expect(backend.connections).toHaveLength(1);

    setVisibility("visible");
    await advance(0);
    expect(backend.connections).toHaveLength(2);
    expect(backend.connections[1].url.searchParams.get("after")).toBe("9-0");
    backend.publish();
    await advance(1_000);

    expect(snapshot().phase).toBe("finished");
    expect(textsOf(render())).toEqual([PROMPT, `${FIRST_TEXT}${SECOND_TEXT}`]);
    expect(toast).not.toHaveBeenCalled();
  });

  it("a tab shown again over a live stream opens nothing", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    setVisibility("hidden");
    await advance(25_000);
    setVisibility("visible");
    await advance(0);
    expect(backend.connections).toHaveLength(1);
    expect(toast).not.toHaveBeenCalled();
  });

  it("the load balancer's cut, landing before the rotation, resumes at once", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.connections[0].cut();
    await advance(0);
    expect(backend.connections).toHaveLength(2);
    expect(backend.connections[1].url.searchParams.get("after")).toBe("9-0");
    expect(snapshot().notice).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });

  it("a trimmed or expired stream is caught up from the rows, without a toast", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.publish();
    backend.respond(() => ({ status: 410, body: { reason: "expired" } }));
    backend.connections[0].fail();
    await advance(1_000);
    expect(snapshot().phase).toBe("finished");
    expect(toast).not.toHaveBeenCalled();
  });

  it("two minutes of model silence with heartbeats is not a disconnect", async () => {
    await sendAndPublish(TOOL_CALL_OPEN);
    await advance(120_000);
    expect(backend.connections).toHaveLength(1);
    expect(snapshot().phase).toBe("live");
    expect(snapshot().notice).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });

  it("a turn that ended while disconnected ends on its stored finish, and is reconciled", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.respond(() => "network-error");
    backend.connections[0].fail();
    backend.publish();
    backend.respond(() => "stream");
    await advance(3_000);

    expect(snapshot().phase).toBe("finished");
    const turn = turnSegment();
    expect(turn.reconciled).toBe(true);
    expect(turn.log.rows.map((r) => r.sequence)).toEqual([1, 2, 3]);
    expect(toast).not.toHaveBeenCalled();
  });

  it("a stream the user stopped is never resumed", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    runtime.stop();
    await advance(60_000);
    expect(backend.connections).toHaveLength(1);
    expect(backend.connections[0].aborted).toBe(true);
    expect(snapshot().stopped).toBe(true);
    expect(textsOf(render()).at(-1)).toContain("Operation cancelled");
    expect(toast).not.toHaveBeenCalled();
  });

  it("a connection silent for 30 s is replaced without a toast", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.connections[0].silence();
    await advance(LIVENESS_MS + 5_000);
    expect(backend.connections[0].aborted).toBe(true);
    expect(backend.connections).toHaveLength(2);
    expect(toast).not.toHaveBeenCalled();
  });

  it("failures while visible and online back off, show an inline notice, and toast once after five", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.respond(() => "network-error");
    backend.connections[0].fail();
    await advance(0);
    expect(snapshot().notice).toBeNull();

    // Attempts after the first drop: 1, 2, 4 and 8 s apart.
    await advance(1_000);
    expect(snapshot().notice).toBe("reconnecting");
    await advance(2_000 + 4_000 + 8_000);
    expect(toast).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith(ANOMALY_TOAST);

    // It keeps trying: every 30 s after the backoff runs out.
    await advance(16_000 + 30_000 + 30_000);
    expect(toast).toHaveBeenCalledTimes(1);
    // The next attempt connects; its first frame clears the notice.
    backend.respond(() => "stream");
    await advance(30_000 + 15_000);
    expect(snapshot().notice).toBeNull();
    expect(snapshot().phase).toBe("live");
  });

  it("offline waits for the network, with a notice and no toast", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    const online = vi.spyOn(navigator, "onLine", "get").mockReturnValue(false);
    backend.connections[0].fail();
    await advance(60_000);
    expect(backend.connections).toHaveLength(1);
    expect(snapshot().notice).toBe("offline");

    online.mockReturnValue(true);
    window.dispatchEvent(new Event("online"));
    await advance(0);
    expect(backend.connections).toHaveLength(2);
    expect(snapshot().notice).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });

  it("an executor that died ends the turn inline as interrupted, never with the connection toast", async () => {
    await sendAndPublish(TOOL_CALL_OPEN);
    backend.respond(() => ({ status: 410, body: { reason: "expired" } }));
    backend.connections[0].fail();
    await advance(1_000);

    expect(snapshot().phase).toBe("finished");
    const messages = render();
    expect(textsOf(messages).at(-1)).toContain("Response was interrupted");
    expect(toolStates(messages)).toEqual(["output-error"]);
    expect(toast).not.toHaveBeenCalled();
  });
});

describe("one connection slot", () => {
  it("W1: two resumes inside the window open one connection and draw one bubble", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    setVisibility("hidden");
    backend.connections[0].fail();
    await advance(10_000);

    // The tab returns and the network event lands in the same tick.
    setVisibility("visible");
    window.dispatchEvent(new Event("online"));
    await advance(0);
    expect(backend.connections).toHaveLength(2);

    backend.publish();
    await advance(1_000);
    const assistant = render().filter((m) => m.role === "assistant");
    expect(assistant).toHaveLength(1);
    expect(textsOf(assistant)).toEqual([`${FIRST_TEXT}${SECOND_TEXT}`]);
  });
});

describe("rotation", () => {
  it("opens the second connection before closing the first, and applies no entry twice", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    await advance(ROTATE_AFTER_MS + 5_000 - 1);
    const [first] = backend.connections;
    expect(backend.connections).toHaveLength(2);
    expect(first.closed).toBe(false);

    const second = backend.connections[1];
    expect(second.url.searchParams.get("after")).toBe("9-0");
    backend.publish();
    await advance(1_000);

    expect(first.aborted).toBe(true);
    expect(second.sent.length).toBeGreaterThan(0);
    expect(textsOf(render())).toEqual([PROMPT, `${FIRST_TEXT}${SECOND_TEXT}`]);
    expect(turnSegment().log.protocolErrors).toEqual([]);
    expect(toast).not.toHaveBeenCalled();
  });
});

describe("resync", () => {
  it("a trimmed stream seeds from the rows at its checkpoint, then tails from it", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    // The turn completes while the client is away, and its stream is
    // trimmed to the last checkpoint.
    backend.connections[0].fail();
    backend.publish();
    const checkpoint = backend.lastCheckpoint();
    expect(checkpoint?.entry_id).toBe("20-0");
    backend.respond(({ url }) =>
      url.searchParams.get("after") === "9-0"
        ? { status: 409, body: { reason: "trimmed", checkpoint } }
        : "stream",
    );
    await advance(2_000);

    const tail = backend.connections.at(-1)!;
    expect(tail.url.searchParams.get("after")).toBe("20-0");
    expect(snapshot().phase).toBe("finished");
    expect(turnSegment().log.rows.map((r) => r.content)).toEqual(
      toolTurn.rows.slice(1).map((r) => r.content),
    );
    expect(toast).not.toHaveBeenCalled();
  });

  it("an expired stream takes the persisted rows as the whole truth", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.publish();
    backend.respond(() => ({ status: 410, body: { reason: "expired" } }));
    backend.connections[0].fail();
    await advance(1_000);

    expect(turnSegment().ended).toBe(true);
    expect(textsOf(render())).toEqual([PROMPT, `${FIRST_TEXT}${SECOND_TEXT}`]);
    expect(textsOf(render()).join()).not.toContain("interrupted");
  });

  it("a fresh mount seeds the running turn from its checkpoint and tails from there", async () => {
    backend.beginRunning();
    backend.publish(20);
    runtime.observe(backend.view());
    await advance(0);
    expect(backend.connections[0].url.searchParams.get("after")).toBe("20-0");
    expect(snapshot().ownedFrom).toBe(1);
    backend.publish();
    await advance(1_000);
    expect(textsOf(render())).toEqual([`${FIRST_TEXT}${SECOND_TEXT}`]);
  });
});

describe("the end-of-turn reconcile", () => {
  it("adopts the rows' sequences and keeps every message object", async () => {
    await sendAndPublish(toolTurn.sse.length - 1);
    const before = render();
    backend.publish();
    await advance(1_000);

    const turn = turnSegment();
    expect(turn.reconciled).toBe(true);
    expect(turn.log.rows.map((r) => r.sequence)).toEqual([1, 2, 3]);
    const after = render();
    expect(after.map((m) => m.id)).toEqual(before.map((m) => m.id));
    expect(report).not.toHaveBeenCalled();
  });

  it("replaces a row that differs under its own key, and reports it", async () => {
    await sendAndPublish(toolTurn.sse.length - 1);
    const before = render();
    const rows = toolTurn.rows.map((row) =>
      row.content === SECOND_TEXT ? { ...row, content: "It says Other." } : row,
    );
    backend.publish();
    vi.spyOn(backend, "view").mockImplementation(() => ({
      messages: rows.map((row, sequence) => ({ ...row, sequence })),
      has_more_messages: false,
      active_stream: null,
    }));
    await advance(1_000);

    expect(render().map((m) => m.id)).toEqual(before.map((m) => m.id));
    expect(textsOf(render())).toEqual([PROMPT, `${FIRST_TEXT}It says Other.`]);
    expect(report).toHaveBeenCalledWith(
      "finish",
      expect.objectContaining({ diffs: [{ index: 2, fields: ["content"] }] }),
    );
  });
});

async function sendAndPublish(count: number) {
  void runtime.send({ text: PROMPT }, undefined);
  await advance(0);
  backend.publish(count);
  await advance(100);
}

async function advance(ms: number) {
  await vi.advanceTimersByTimeAsync(ms);
}

function snapshot(): RuntimeSnapshot {
  return runtime.getSnapshot();
}

function turnSegment() {
  const turn = snapshot().segments.find((s) => s.kind === "turn");
  if (turn?.kind !== "turn") throw new Error("no turn");
  return turn;
}

const renderTail = createTailRenderer("session-1");
function render() {
  return renderTail(snapshot()).messages;
}

function textsOf(messages: UIMessage[]) {
  return messages.map((m) =>
    m.parts.map((p) => (p.type === "text" ? p.text : "")).join(""),
  );
}

function toolStates(messages: UIMessage[]) {
  return messages
    .flatMap((m) => m.parts)
    .filter((p) => p.type.startsWith("tool-"))
    .map((p) => (p as { state: string }).state);
}

function setVisibility(state: "hidden" | "visible") {
  Object.defineProperty(document, "visibilityState", {
    value: state,
    configurable: true,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}
