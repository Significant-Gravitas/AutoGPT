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

import { CopilotAuthError } from "../../helpers";
import { createTailRenderer } from "../renderTail";
import {
  ANOMALY_TOAST,
  ChatUnavailableError,
  LIVENESS_MS,
  MessageTooLongError,
  ROTATE_AFTER_MS,
  SendRefusedError,
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
let tokenFailures = 0;

beforeEach(() => {
  vi.useFakeTimers();
  setVisibility("visible");
  backend = fakeBackend(toolTurn);
  toast = vi.fn<RuntimeDeps["toast"]>();
  report = vi.fn<RuntimeDeps["report"]>();
  tokenFailures = 0;
  runtime = new TurnRuntime("session-1", {
    baseUrl: () => "http://backend.test",
    headers: async () => {
      if (tokenFailures > 0) {
        tokenFailures -= 1;
        throw new CopilotAuthError();
      }
      return {};
    },
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

  it("a stop before the POST names its turn never attaches to that turn", async () => {
    void runtime.send({ text: PROMPT }, undefined);
    await advance(0);
    runtime.stop();
    // The server runs the turn on for a moment, and the view says so.
    backend.publish(FIRST_BLOCK_DONE);
    runtime.observe(backend.view());
    await advance(60_000);
    expect(backend.connections).toHaveLength(1);
    expect(snapshot().stopped).toBe(true);
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

  it("a resume that yields only the end marker while the view shows no active turn ends the turn, inline", async () => {
    await sendAndPublish(toolTurn.sse.length - 1);
    backend.stopWithoutFinish();
    await advance(60_000);

    expect(backend.connections.map((c) => c.method)).toEqual(["POST", "GET"]);
    expect(snapshot().phase).toBe("finished");
    expect(turnSegment().reconciled).toBe(true);
    expect(textsOf(render())).toEqual([PROMPT, `${FIRST_TEXT}${SECOND_TEXT}`]);
    expect(textsOf(render()).join()).not.toContain("interrupted");
    expect(toast).not.toHaveBeenCalled();
  });

  it("a resume that yields only the end marker while the view still runs the turn backs off", async () => {
    await sendAndPublish(toolTurn.sse.length - 1);
    runtime.bind({}, async () => ({
      ...backend.view(),
      active_stream: { turn_id: backend.turnId },
    }));
    backend.stopWithoutFinish();
    await advance(60_000);

    // At once after the POST's close, then 1, 2, 4 and 8 s after each 5 s read.
    expect(backend.connections.filter((c) => c.method === "GET")).toHaveLength(
      5,
    );
    expect(snapshot().notice).toBe("reconnecting");
  });
});

describe("recovery", () => {
  it("a resync in flight keeps every other trigger from reconnecting", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.connections[0].fail();
    backend.publish();
    const checkpoint = backend.lastCheckpoint();
    backend.respond(({ url }) =>
      url.searchParams.get("after") === "9-0"
        ? { status: 409, body: { reason: "trimmed", checkpoint } }
        : "stream",
    );
    let release!: () => void;
    const held = new Promise<void>((resolve) => (release = resolve));
    runtime.bind({}, async () => {
      await held;
      return backend.view();
    });
    await advance(1_000);
    const before = backend.requests.length;

    // The tab returns and the network event lands while the view is fetched.
    window.dispatchEvent(new Event("online"));
    setVisibility("hidden");
    setVisibility("visible");
    await advance(1_000);
    expect(backend.requests).toHaveLength(before);

    release();
    await advance(1_000);
    expect(backend.requests.at(-1)?.url.searchParams.get("after")).toBe("20-0");
    expect(turnSegment().frozen).toBe(false);
    expect(snapshot().phase).toBe("finished");
  });

  it("a send lost while hidden is found on return", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    setVisibility("hidden");
    void runtime.send({ text: PROMPT }, undefined);
    await advance(60_000);
    expect(snapshot().phase).toBe("idle");

    // The backend started the turn anyway.
    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    setVisibility("visible");
    await advance(1_000);
    expect(backend.connections.at(-1)?.url.searchParams.get("turn")).toBe(
      backend.turnId,
    );
    expect(textsOf(render())).toEqual([PROMPT, FIRST_TEXT]);
  });

  it("a send cut while hidden whose turn finished meanwhile shows the answer, not an error", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    setVisibility("hidden");
    void runtime.send({ text: PROMPT }, undefined);
    await advance(60_000);

    // The backend ran the turn to its end while the tab slept.
    backend.beginRunning();
    backend.publish();
    setVisibility("visible");
    await advance(10_000);
    expect(snapshot().error).toBeNull();
    expect(textsOf(render()).join("")).toBe(
      `${PROMPT}${FIRST_TEXT}${SECOND_TEXT}`,
    );
  });

  it("a send silent while the tab slept is given up on return, not at the next tick", async () => {
    backend.respond(({ method }) => (method === "POST" ? "silent" : "stream"));
    setVisibility("hidden");
    void runtime.send({ text: PROMPT }, undefined);
    await advance(60_000);
    backend.beginRunning();
    backend.publish();

    const views = vi.spyOn(backend, "view");
    setVisibility("visible");
    await advance(0);
    expect(views).toHaveBeenCalledTimes(1);
    expect(textsOf(render()).join("")).toBe(
      `${PROMPT}${FIRST_TEXT}${SECOND_TEXT}`,
    );
  });

  it("a send that never reached the server still reports the failure", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    setVisibility("hidden");
    void runtime.send({ text: PROMPT }, undefined);
    await advance(60_000);
    setVisibility("visible");
    await advance(10_000);
    expect(snapshot().error?.message).toBe("Failed to fetch");
  });

  it("a send lost while a later send lands still reports its own failure", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    void runtime.send({ text: "Did this one arrive?" }, undefined);
    await advance(0);

    // The user sends again while the lost one is being looked for.
    backend.respond(() => "stream");
    await sendAndPublish(toolTurn.sse.length);
    await advance(10_000);
    expect(snapshot().error?.message).toBe("Failed to fetch");
  });

  it("a lost resend of the same prompt takes no earlier prompt's row", async () => {
    await sendAndPublish(toolTurn.sse.length);
    await advance(2_000);
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);

    expect(textsOf(render())).toEqual([
      PROMPT,
      `${FIRST_TEXT}${SECOND_TEXT}`,
      PROMPT,
    ]);
  });

  it("a send the backend queued behind the running cap is followed into its turn", async () => {
    backend.queueNextTurn();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    expect(snapshot().phase).toBe("idle");

    // Another of the user's turns ends, and the backend starts this one.
    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    await advance(15_000);
    expect(backend.connections.at(-1)?.url.searchParams.get("turn")).toBe(
      backend.turnId,
    );
    expect(snapshot().phase).toBe("live");
    expect(textsOf(render())).toEqual([PROMPT, FIRST_TEXT]);
  });

  it("a queued turn marked running before its stream exists is still followed into it", async () => {
    backend.queueNextTurn();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    backend.claimQueued();
    await advance(30_000);

    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    await advance(15_000);
    expect(snapshot().phase).toBe("live");
    expect(textsOf(render())).toEqual([PROMPT, FIRST_TEXT]);
  });

  it("gives up on a claimed queued turn whose stream never appears, and reports it", async () => {
    backend.queueNextTurn();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    backend.claimQueued();
    await advance(3 * 60_000);

    expect(report).toHaveBeenCalledWith(
      "queue_start_stalled",
      expect.anything(),
    );
    const views = vi.spyOn(backend, "view");
    await advance(60_000);
    expect(views).not.toHaveBeenCalled();
  });

  it("does not follow a queued turn the user stopped", async () => {
    backend.queueNextTurn();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    runtime.stop();
    // The page's own refetch still shows the session queued.
    const queuedView = backend.view();
    runtime.observe(queuedView);

    const views = vi.spyOn(backend, "view");
    await advance(60_000);
    expect(views).not.toHaveBeenCalled();
  });

  it("reads a queued session one request at a time", async () => {
    backend.queueNextTurn();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    let inFlight = 0;
    let most = 0;
    runtime.bind({}, async () => {
      most = Math.max(most, ++inFlight);
      await new Promise((resolve) => setTimeout(resolve, 25_000));
      inFlight -= 1;
      return backend.view();
    });

    await advance(120_000);
    expect(most).toBe(1);
  });

  it("a send queued behind another tab's running turn is not drawn above that turn", async () => {
    // Another tab's turn runs; this tab has not read the session since.
    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    backend.respond(({ method }) =>
      method === "POST" ? "follow-up-queued" : "stream",
    );
    void runtime.send({ text: "And what about example.org?" }, undefined);
    await advance(2_000);

    expect(backend.connections.at(-1)?.url.searchParams.get("turn")).toBe(
      backend.turnId,
    );
    // The other tab's prompt (row 0) stays history above the followed turn.
    expect(snapshot().ownedFrom).toBe(1);
    expect(textsOf(render())).toEqual([FIRST_TEXT]);
    expect(snapshot().error).toBeNull();
  });

  it("stops reading the session once its queued turn is cancelled", async () => {
    backend.queueNextTurn();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    backend.cancelQueued();
    await advance(10_000);

    const views = vi.spyOn(backend, "view");
    await advance(60_000);
    expect(views).not.toHaveBeenCalled();
  });
});

describe("after a lost send, what the user is told follows the server", () => {
  it("shows a lost send's answer without a lingering notice when the first reads on return fail", async () => {
    loseSendWhileHidden();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(60_000);
    backend.beginRunning();
    backend.publish();
    viewsFailFirst(2);
    setVisibility("visible");
    await advance(30_000);

    expect(snapshot().notice).toBeNull();
    expect(snapshot().error).toBeNull();
    expect(textsOf(render()).join("")).toBe(
      `${PROMPT}${FIRST_TEXT}${SECOND_TEXT}`,
    );
  });

  it("still reports a send that never reached the server when the first reads on return fail", async () => {
    loseSendWhileHidden();
    void runtime.send({ text: PROMPT }, undefined);
    await advance(60_000);
    viewsFailFirst(2);
    setVisibility("visible");
    await advance(30_000);

    expect(snapshot().error?.message).toBe("Failed to fetch");
    expect(snapshot().notice).toBeNull();
  });

  it("reports no error when another read attaches the lost send's turn first", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    let release!: () => void;
    const held = new Promise<void>((resolve) => (release = resolve));
    runtime.bind({}, async () => {
      await held;
      return backend.view();
    });
    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    void runtime.send({ text: PROMPT }, undefined);
    await advance(0);
    // The page's own read lands while the recovery's first read is in flight.
    runtime.observe(backend.view());
    release();
    await advance(10_000);
    expect(snapshot().error).toBeNull();
    expect(snapshot().phase).toBe("live");

    backend.publish();
    await advance(5_000);
    expect(snapshot().error).toBeNull();
    expect(snapshot().phase).toBe("finished");
  });

  it("clears a failed send's error once a turn the server started is followed", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    expect(snapshot().error?.message).toBe("Failed to fetch");

    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    runtime.followBackendTurn();
    await advance(2_000);
    expect(snapshot().phase).toBe("live");
    expect(snapshot().error).toBeNull();
  });

  it("stops reading the session once the lost send's prompt has landed and nothing runs", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    backend.beginRunning();
    backend.publish();
    const views = vi.spyOn(backend, "view");
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);

    expect(views).toHaveBeenCalledTimes(1);
    expect(snapshot().error).toBeNull();
  });

  it("keeps reading while the session runs a landed prompt whose turn has not registered yet", async () => {
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    // The slot is claimed and the prompt written before the turn's stream exists.
    let reads = 0;
    runtime.bind({}, async () => {
      reads += 1;
      const view = backend.view();
      return reads <= 2
        ? { ...view, active_stream: null, chat_status: "running" }
        : view;
    });
    void runtime.send({ text: PROMPT }, undefined);
    await advance(5_000);

    expect(backend.connections.at(-1)?.url.searchParams.get("turn")).toBe(
      backend.turnId,
    );
    expect(snapshot().phase).toBe("live");
  });

  it("shows the offline notice while a send waits for the network", async () => {
    const online = vi.spyOn(navigator, "onLine", "get").mockReturnValue(false);
    backend.respond(({ method }) =>
      method === "POST" ? "network-error" : "stream",
    );
    void runtime.send({ text: PROMPT }, undefined);
    await advance(10_000);
    expect(snapshot().notice).toBe("offline");
    expect(snapshot().error).toBeNull();

    online.mockReturnValue(true);
    window.dispatchEvent(new Event("online"));
    await advance(10_000);
    expect(snapshot().notice).toBeNull();
    expect(snapshot().error?.message).toBe("Failed to fetch");
  });
});

describe("refusals no retry can change", () => {
  it("a send while signed out reports the sign-in error after one retry, and stops", async () => {
    const onError = vi.fn();
    signOut(() => true, onError);
    void runtime.send({ text: PROMPT }, undefined);
    await advance(120_000);

    expect(snapshot().error).toBeInstanceOf(CopilotAuthError);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(snapshot().notice).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });

  it("a send whose token fails once goes out a second later", async () => {
    tokenFailsFirst(1);
    void runtime.send({ text: PROMPT }, undefined);
    await advance(500);
    expect(backend.requests.filter((r) => r.method === "POST")).toHaveLength(0);

    await advance(1_000);
    backend.publish(FIRST_BLOCK_DONE);
    await advance(100);
    expect(backend.requests.filter((r) => r.method === "POST")).toHaveLength(1);
    expect(snapshot().error).toBeNull();
    expect(snapshot().phase).toBe("live");
  });

  it("a resume whose token fails once resumes", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    tokenFailsFirst(1);
    backend.connections[0].fail();
    await advance(2_000);

    expect(snapshot().error).toBeNull();
    expect(backend.connections.at(-1)?.url.searchParams.get("turn")).toBe(
      backend.turnId,
    );
  });

  it("a second send while a send waits out its token retry opens no second request", async () => {
    tokenFailsFirst(1);
    void runtime.send({ text: PROMPT }, undefined);
    await advance(500);
    void runtime.send({ text: "And example.org?" }, undefined);
    await advance(1_000);
    backend.publish(FIRST_BLOCK_DONE);
    await advance(100);

    expect(backend.requests.filter((r) => r.method === "POST")).toHaveLength(1);
    expect(backend.connections.some((c) => c.aborted)).toBe(false);
    expect(snapshot().error).toBeNull();
    expect(snapshot().phase).toBe("live");
  });

  it("a stop while a send waits out its token retry sends nothing after", async () => {
    tokenFailsFirst(1);
    void runtime.send({ text: PROMPT }, undefined);
    await advance(500);
    runtime.stop();
    await advance(5_000);

    expect(backend.requests.filter((r) => r.method === "POST")).toHaveLength(0);
    expect(snapshot().error).toBeNull();
  });

  it("signed out mid-turn ends the turn with the sign-in error, without the connection toast", async () => {
    let signedOut = false;
    signOut(() => signedOut);
    await sendAndPublish(FIRST_BLOCK_DONE);
    signedOut = true;
    backend.connections[0].fail();
    await advance(180_000);

    expect(snapshot().error).toBeInstanceOf(CopilotAuthError);
    expect(snapshot().phase).toBe("failed");
    expect(snapshot().notice).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });

  it("a resume refused twice with 401 ends the turn with the sign-in error", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.respond(() => ({ status: 401, body: { detail: "Unauthorized" } }));
    backend.connections[0].fail();
    await advance(180_000);

    expect(backend.requests.filter((r) => r.method === "GET")).toHaveLength(2);
    expect(snapshot().error).toBeInstanceOf(CopilotAuthError);
    expect(snapshot().phase).toBe("failed");
    expect(toast).not.toHaveBeenCalled();
  });

  it("a resume refused once with 401 resumes on a fresh token", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    let refusals = 0;
    backend.respond(() =>
      refusals++ === 0
        ? { status: 401, body: { detail: "Unauthorized" } }
        : "stream",
    );
    backend.connections[0].fail();
    await advance(1_000);
    backend.publish();
    await advance(1_000);

    expect(backend.requests.filter((r) => r.method === "GET")).toHaveLength(2);
    expect(snapshot().phase).toBe("finished");
    expect(snapshot().error).toBeNull();
  });

  it("a resume refused 403 twice ends with the refusal, not the sign-in error", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.respond(() => ({ status: 403, body: { detail: "Forbidden" } }));
    backend.connections[0].fail();
    await advance(180_000);

    expect(backend.requests.filter((r) => r.method === "GET")).toHaveLength(2);
    expect(snapshot().error).toEqual(
      new SendRefusedError('{"detail":"Forbidden"}', 403),
    );
    expect(snapshot().error).not.toBeInstanceOf(CopilotAuthError);
    expect(report).toHaveBeenCalledWith(
      "refused",
      expect.objectContaining({ status: 403 }),
    );
    expect(toast).not.toHaveBeenCalled();
  });

  it("reports a send refused with a 4xx the app never provokes, with its status", async () => {
    backend.respond(() => ({
      status: 422,
      body: { detail: "Message too long" },
    }));
    void runtime.send({ text: PROMPT }, undefined);
    await advance(1_000);

    expect(report).toHaveBeenCalledWith("refused", {
      status: 422,
      body: { detail: "Message too long" },
    });
    expect(snapshot().error).toEqual(
      new SendRefusedError('{"detail":"Message too long"}', 422),
    );
  });

  it("a message over the server's length limit is its own refusal, not reported", async () => {
    // FastAPI's 422 for StreamChatRequest.message over max_length=64_000.
    backend.respond(() => ({
      status: 422,
      body: {
        detail: [
          {
            type: "string_too_long",
            loc: ["body", "message"],
            msg: "String should have at most 64000 characters",
          },
        ],
      },
    }));
    void runtime.send({ text: PROMPT }, undefined);
    await advance(1_000);

    expect(snapshot().error).toBeInstanceOf(MessageTooLongError);
    expect(report).not.toHaveBeenCalled();
  });

  it.each([401, 402, 404, 429])(
    "does not report a send refused %i, which the user's state explains",
    async (status) => {
      backend.respond(() => ({ status, body: { detail: "refused" } }));
      void runtime.send({ text: PROMPT }, undefined);
      await advance(1_000);

      expect(report).not.toHaveBeenCalled();
    },
  );

  it("a resume refused 404 ends the turn: the chat is gone", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    backend.respond(() => ({ status: 404, body: { detail: "Not found" } }));
    backend.connections[0].fail();
    await advance(180_000);

    expect(backend.requests.filter((r) => r.method === "GET")).toHaveLength(1);
    expect(snapshot().error).toBeInstanceOf(ChatUnavailableError);
    expect(toast).not.toHaveBeenCalled();
  });

  it("a send refused by a proxy's HTML page reads as plain language", async () => {
    backend.respond(() => ({
      status: 502,
      body: "<html><head><title>502 Server Error</title></head></html>",
    }));
    void runtime.send({ text: PROMPT }, undefined);
    await advance(1_000);

    expect(snapshot().error).toEqual(
      new SendRefusedError("Couldn't reach the server (HTTP 502).", 502),
    );
    expect(snapshot().error).toBeInstanceOf(SendRefusedError);
  });
});

describe("a return to the tab", () => {
  it("after more than 30 s away, follows a turn started elsewhere meanwhile", async () => {
    runtime.observe(backend.view());
    setVisibility("hidden");
    await advance(60_000);
    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    setVisibility("visible");
    await advance(1_000);

    expect(backend.connections.at(-1)?.url.searchParams.get("turn")).toBe(
      backend.turnId,
    );
    expect(snapshot().phase).toBe("live");
  });

  it("when the first read on return fails, reads again and follows the turn", async () => {
    runtime.observe(backend.view());
    setVisibility("hidden");
    await advance(60_000);
    backend.beginRunning();
    backend.publish(FIRST_BLOCK_DONE);
    let reads = 0;
    runtime.bind({}, async () => (reads++ === 0 ? null : backend.view()));
    setVisibility("visible");
    await advance(3_000);

    expect(reads).toBe(2);
    expect(snapshot().phase).toBe("live");
  });

  it("stops re-reading after three failed retries", async () => {
    runtime.observe(backend.view());
    setVisibility("hidden");
    await advance(60_000);
    let reads = 0;
    runtime.bind({}, async () => {
      reads += 1;
      return null;
    });
    setVisibility("visible");
    await advance(120_000);

    expect(reads).toBe(4);
  });

  it("within 30 s, reads nothing", async () => {
    runtime.observe(backend.view());
    setVisibility("hidden");
    await advance(10_000);
    const views = vi.spyOn(backend, "view");
    setVisibility("visible");
    await advance(1_000);
    expect(views).not.toHaveBeenCalled();
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
    // Auth headers take a moment, so the entry published as the rotation
    // starts reaches the first connection before the second request leaves.
    runtime.dispose();
    runtime = new TurnRuntime("session-1", {
      baseUrl: () => "http://backend.test",
      headers: () =>
        new Promise((resolve) => setTimeout(() => resolve({}), 50)),
      fetch: backend.fetch,
      fetchSession: async () => backend.view(),
      toast,
      report,
    });
    void runtime.send({ text: PROMPT }, undefined);
    await advance(100);
    backend.publish(6);
    await advance(100);

    // The rotation tick fires at 25 minutes; its request waits on headers.
    await advance(ROTATE_AFTER_MS - 200 + 10);
    expect(runtime.getSnapshot().phase).toBe("live");
    backend.publish(1);
    await advance(100);
    const [first, second] = backend.connections;
    expect(backend.connections).toHaveLength(2);
    expect(second.url.searchParams.get("after")).toBe("6-0");
    // Both carried the entry; the first applied it and then was retired.
    expect(first.sent.at(-1)).toBe("7-0");
    expect(second.sent).toEqual(["7-0"]);
    expect(first.aborted).toBe(true);
    // Checked here: the end-of-turn reconcile would heal a doubled row.
    expect(turnSegment().log.rows.map((r) => r.content)).toEqual([
      "Let me fetch that ",
    ]);

    backend.publish();
    await advance(1_000);
    expect(textsOf(render())).toEqual([PROMPT, `${FIRST_TEXT}${SECOND_TEXT}`]);
    expect(turnSegment().log.protocolErrors).toEqual([]);
    expect(report).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
  });

  it("a connection lost past the rotation age resumes at once", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    // Every rotation fails to open, so the first connection lives to the
    // load balancer's cut.
    backend.respond(({ method }) =>
      method === "GET" ? "network-error" : "stream",
    );
    await advance(ROTATE_AFTER_MS + 60_000);
    backend.respond(() => "stream");
    backend.connections[0].fail();
    await advance(0);
    expect(backend.connections).toHaveLength(2);
    expect(backend.connections[1].url.searchParams.get("after")).toBe("9-0");
    expect(toast).not.toHaveBeenCalled();
  });

  it("keeps the first connection open until the second delivers a frame", async () => {
    await sendAndPublish(FIRST_BLOCK_DONE);
    await advance(ROTATE_AFTER_MS + 5_000 - 1);
    const [first, second] = backend.connections;
    expect(second.url.searchParams.get("after")).toBe("9-0");
    expect(first.closed).toBe(false);
    await advance(10_000);
    expect(first.aborted).toBe(true);
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

function loseSendWhileHidden() {
  backend.respond(({ method }) =>
    method === "POST" ? "network-error" : "stream",
  );
  setVisibility("hidden");
}

function viewsFailFirst(count: number) {
  let failed = 0;
  runtime.bind({}, async () => (failed++ < count ? null : backend.view()));
}

// Signed out: no token for any request, and the session read is refused too.
function signOut(isSignedOut: () => boolean, onError = vi.fn()) {
  runtime.dispose();
  runtime = new TurnRuntime("session-1", {
    baseUrl: () => "http://backend.test",
    headers: async () => {
      if (isSignedOut()) throw new CopilotAuthError();
      return {};
    },
    fetch: backend.fetch,
    fetchSession: async () => backend.view(),
    toast,
    report,
  });
  runtime.bind({ onError }, async () =>
    isSignedOut() ? null : backend.view(),
  );
}

// The token action fails the next `count` times, then hands out tokens again.
function tokenFailsFirst(count: number) {
  tokenFailures = count;
}

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
