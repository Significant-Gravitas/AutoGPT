import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CONTINUATION_PROBES, SendInFlightError } from "../../turnRuntime";
import { FaultServer } from "./faultServer";
import { Tab, type Screen } from "./tab";

// A case the runtime still gets wrong runs as `it.fails`; fixing it fails the
// case here until its entry goes.
const KNOWN_WRONG: Record<string, string> = {
  F1: "a server-started turn that registers after the probes is not followed",
  F3: "an idle visible tab is not told of a turn another device starts",
};
const CASES = new Set<string>();

const P1 = "What does example.com say?";
const P2 = "And what about example.org?";
const P3 = "Then compare the two.";
const ANSWER = "assistant: Let me fetch that page.It says Example Domain.";
const FIRST_BLOCK = "assistant: Let me fetch that page.";
const SIGN_IN_AGAIN = "Authentication failed — please sign in again.";
const UNCONFIRMED =
  "The connection stalled before the server confirmed your message.";
const KICKOFF = {
  kind: "expert_kickoff",
  expertId: "3f2b8c1e-7d4a-4e9b-8c2f-1a6d5e9b0c47",
};

const auth = vi.hoisted(() => ({ signedIn: true, tokenDown: false }));
vi.mock("@/lib/auth/actions", () => ({
  getWebSocketToken: async () =>
    auth.tokenDown
      ? { token: null, error: "Token service unavailable", retryable: true }
      : auth.signedIn
        ? { token: "token" }
        : { token: null, error: "No session" },
}));

let server: FaultServer;
let tab: Tab;
const tabs: Tab[] = [];

beforeEach(() => {
  vi.useFakeTimers();
  setVisibility("visible");
  auth.signedIn = true;
  auth.tokenDown = false;
  server = new FaultServer();
  tab = openTab();
});

afterEach(() => {
  // Each lost send's check ends in a verdict, a stop or a give-up, or is
  // still held: one dropped on the way is never reported.
  const dropped = tabs.flatMap((t) => t.runtime.droppedChecks());
  tabs.splice(0).forEach((t) => t.dispose());
  vi.restoreAllMocks();
  vi.useRealTimers();
  setVisibility("visible");
  expect(dropped).toEqual([]);
});

it("every KNOWN_WRONG id names a case", () => {
  expect(Object.keys(KNOWN_WRONG).filter((id) => !CASES.has(id))).toEqual([]);
});

describe("a send", () => {
  cell("S1", "refused 500 with JSON is reported", async () => {
    server.postMode = { status: 500, body: { detail: "Internal error" } };
    tab.send(P1);
    await advance(10_000);
    expectScreen({
      error: '{"detail":"Internal error"}',
      notice: null,
      toasts: 0,
    });
  });

  cell(
    "S2",
    "refused by a proxy's HTML page is reported in words",
    async () => {
      server.postMode = { status: 502, body: "<html><body>502</body></html>" };
      tab.send(P1);
      await advance(10_000);
      expect(tab.screen().error).not.toMatch(/^</);
    },
  );

  cell("S3", "that never reached the server is reported", async () => {
    server.postMode = "lost-before-send";
    tab.send(P1);
    await advance(15_000);
    expectScreen({
      error: "Failed to fetch",
      notice: null,
      phase: "failed",
      toasts: 0,
    });
  });

  cell(
    "S4",
    "whose response was lost while its turn runs is followed",
    async () => {
      server.postMode = "lost-after-receive";
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(2_000);
      expectScreen({
        error: null,
        phase: "live",
        chat: [`user: ${P1}`, FIRST_BLOCK],
      });
    },
  );

  cell(
    "S5",
    "whose response was lost after its turn ended shows the answer",
    async () => {
      server.postMode = "lost-after-receive";
      tab.send(P1);
      await advance(0);
      server.publish();
      await advance(15_000);
      expectScreen({
        error: null,
        notice: null,
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "S6",
    "lost while hidden is followed into its running turn on return",
    async () => {
      server.postMode = "lost-after-receive";
      setVisibility("hidden");
      tab.send(P1);
      await advance(60_000);
      server.publish(9);
      setVisibility("visible");
      await advance(2_000);
      expectScreen({
        error: null,
        phase: "live",
        chat: [`user: ${P1}`, FIRST_BLOCK],
      });
    },
  );

  cell(
    "S7",
    "lost while hidden shows the answer its turn finished meanwhile",
    async () => {
      server.postMode = "lost-after-receive";
      setVisibility("hidden");
      tab.send(P1);
      await advance(30_000);
      server.publish();
      await advance(30_000);
      setVisibility("visible");
      await advance(15_000);
      expectScreen({
        error: null,
        notice: null,
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "S8",
    "on a frozen page whose socket died meanwhile shows the finished answer",
    async () => {
      server.postMode = "silent-after-receive";
      tab.send(P1);
      await advance(0);
      setVisibility("hidden");
      server.publish();
      await advance(60_000);
      server.connections[0].fail();
      setVisibility("visible");
      await advance(15_000);
      expectScreen({
        error: null,
        notice: null,
        phase: "idle",
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "S9",
    "on a frozen page is followed into its running turn on return",
    async () => {
      server.postMode = "silent-after-receive";
      tab.send(P1);
      await advance(0);
      setVisibility("hidden");
      server.publish(9);
      await advance(60_000);
      server.connections[0].fail();
      setVisibility("visible");
      await advance(2_000);
      expectScreen({
        error: null,
        phase: "live",
        chat: [`user: ${P1}`, FIRST_BLOCK],
      });
    },
  );

  cell(
    "S9b",
    "on a frozen page with its socket still open shows the answer on return",
    async () => {
      server.postMode = "silent-after-receive";
      tab.send(P1);
      await advance(0);
      setVisibility("hidden");
      server.publish();
      await advance(60_000);
      setVisibility("visible");
      await advance(3_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] });
    },
  );

  cell(
    "S10",
    "lost with its first session reads failing shows the answer and no notice",
    async () => {
      server.postMode = "lost-after-receive";
      server.failViews = 2;
      tab.send(P1);
      await advance(0);
      server.publish();
      await advance(30_000);
      expectScreen({
        error: null,
        notice: null,
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "S10h",
    "lost while hidden, first reads on return failing, shows the answer and no notice",
    async () => {
      server.postMode = "lost-after-receive";
      setVisibility("hidden");
      tab.send(P1);
      await advance(0);
      server.publish();
      await advance(60_000);
      server.failViews = 2;
      setVisibility("visible");
      await advance(30_000);
      expectScreen({
        error: null,
        notice: null,
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "S11",
    "that never arrived, first reads failing, is still reported",
    async () => {
      server.postMode = "lost-before-send";
      server.failViews = 2;
      tab.send(P1);
      await advance(30_000);
      expectScreen({ error: "Failed to fetch", notice: null, phase: "failed" });
    },
  );

  cell(
    "S11h",
    "that never arrived while hidden, first reads on return failing, is still reported",
    async () => {
      server.postMode = "lost-before-send";
      setVisibility("hidden");
      tab.send(P1);
      await advance(60_000);
      server.failViews = 2;
      setVisibility("visible");
      await advance(30_000);
      expectScreen({ error: "Failed to fetch", notice: null, phase: "failed" });
    },
  );

  cell(
    "S12",
    "lost while another read attaches its turn shows no error",
    async () => {
      tab.dispose();
      tab = openTab(300);
      server.postMode = "lost-after-receive";
      tab.send(P1);
      await advance(0);
      tab.runtime.observe(server.view());
      server.publish(9);
      await advance(20_000);
      expectScreen({ error: null, phase: "live" });
      server.publish();
      await advance(15_000);
      expectScreen({
        error: null,
        phase: "finished",
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "S12b",
    "sent just after an answer, lost, while that turn's settle read attaches it, shows no error",
    async () => {
      tab.dispose();
      tab = openTab(300);
      tab.send(P1);
      await advance(0);
      server.publish();
      // The finished turn's settle read starts 500 ms later and answers 300 ms after that.
      await advance(550);
      server.postMode = "lost-after-receive";
      tab.send(P2);
      await advance(400);
      server.publish(9);
      await advance(20_000);
      expectScreen({ error: null, phase: "live" });
      server.publish();
      await advance(15_000);
      expectScreen({
        error: null,
        chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, ANSWER],
      });
    },
  );

  cell(
    "S13",
    "lost, then sent again during the recovery, shows both turns",
    async () => {
      server.postMode = () =>
        server.turns.length === 0 ? "lost-after-receive" : "stream";
      tab.send(P1);
      await advance(0);
      tab.send(P2);
      await advance(0);
      server.publish();
      await advance(5_000);
      server.publish();
      await advance(30_000);
      expectScreen({
        error: null,
        chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, ANSWER],
      });
    },
  );

  cell(
    "S14",
    "answered by the route's own error frame is followed into its turn",
    async () => {
      server.postMode = "route-error-after-receive";
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(3_000);
      expectScreen({ error: null, phase: "live" });
    },
  );

  cell("S15", "while offline shows the offline notice", async () => {
    server.postMode = "lost-before-send";
    setOnline(false);
    tab.send(P1);
    await advance(10_000);
    expectScreen({ error: null, notice: "offline" });
  });

  cell(
    "S15b",
    "while offline is reported once the network returns",
    async () => {
      server.postMode = "lost-before-send";
      setOnline(false);
      tab.send(P1);
      await advance(10_000);
      setOnline(true);
      await advance(15_000);
      expectScreen({ error: "Failed to fetch", notice: null });
    },
  );

  cell(
    "S16",
    "queued at the running cap, its response lost, is followed into its turn",
    async () => {
      server.queueing = true;
      server.postMode = "lost-after-receive";
      tab.send(P1);
      await advance(10_000);
      server.startQueued();
      server.publish(9);
      await advance(25_000);
      expectScreen({
        error: null,
        phase: "live",
        chat: [`user: ${P1}`, FIRST_BLOCK],
      });
    },
  );

  cell(
    "S17",
    "while signed out reports it, without the reconnect loop",
    async () => {
      auth.signedIn = false;
      server.viewsDown = true;
      tab.send(P1);
      await advance(120_000);
      expectScreen({ error: SIGN_IN_AGAIN, notice: null, toasts: 0 });
    },
  );

  cell(
    "S18",
    "lost, then signed out before its recovery reads, reports it without the reconnect loop",
    async () => {
      server.postMode = "lost-before-send";
      server.viewsDown = true;
      tab.send(P1);
      await advance(0);
      auth.signedIn = false;
      await advance(120_000);
      expectScreen({ error: SIGN_IN_AGAIN, notice: null, toasts: 0 });
    },
  );

  cell(
    "S18s",
    "lost, then stopped while its sign-out check waits, stays stopped without the sign-in error",
    async () => {
      server.postMode = "lost-before-send";
      server.viewsDown = true;
      tab.send(P1);
      // `send` reads the token before its first await, so this lands after
      // the send's own lookup and before its recovery's.
      auth.signedIn = false;
      await advance(500);
      tab.runtime.stop();
      await advance(120_000);
      expectScreen({ error: null, notice: null, toasts: 0 });
    },
  );

  cell(
    "S18b",
    "lost while the token service is down keeps trying, and is reported once it is back",
    async () => {
      server.postMode = "lost-before-send";
      server.viewsDown = true;
      tab.send(P1);
      await advance(0);
      auth.tokenDown = true;
      await advance(20_000);
      auth.tokenDown = false;
      server.viewsDown = false;
      await advance(60_000);
      expectScreen({ error: "Failed to fetch", notice: null });
    },
  );

  cell(
    "S19",
    "whose request hangs without reaching the server is reported once it is given up",
    async () => {
      server.postMode = "hang-before-send";
      tab.send(P1);
      await advance(60_000);
      expectScreen({
        error: UNCONFIRMED,
        notice: null,
        phase: "failed",
        chat: [`user: ${P1}`],
      });
    },
  );

  cell(
    "S19s",
    "whose request hung, stopped while its checks run, shows no error",
    async () => {
      server.postMode = "hang-before-send";
      tab.send(P1);
      await advance(31_000);
      tab.runtime.stop();
      await advance(30_000);
      expectScreen({ error: null, notice: null });
    },
  );

  cell(
    "S19t",
    "whose request hung, stopped during its last check, shows no error",
    async () => {
      server.postMode = "hang-before-send";
      const readsBefore = server.viewReads;
      tab.send(P1);
      // The stop lands after the check's last read, before its verdict.
      let errorAtStop: Error | null | undefined;
      onRead(CONTINUATION_PROBES, () =>
        setTimeout(() => {
          errorAtStop = tab.runtime.getSnapshot().error;
          tab.runtime.stop();
        }, 0),
      );
      await advance(60_000);
      expect(errorAtStop).toBeNull();
      // Every read was the check's, and none followed the stop: a stop placed
      // a read early would leave one more read, as the check reads before it
      // sees the stop.
      expect(server.viewReads - readsBefore).toBe(CONTINUATION_PROBES);
      expectScreen({ error: null, notice: null });
    },
  );

  cell(
    "S19g",
    "whose request hung, once reported, is sent again under its first id and answered once",
    async () => {
      server.postMode = "hang-before-send";
      void tab.runtime.send(
        {
          parts: [
            { type: "text", text: P1 },
            {
              type: "file",
              mediaType: "application/pdf",
              filename: "report.pdf",
              url: "/api/proxy/api/workspace/files/file-1/download",
            },
            {
              type: "data-workspace-folder",
              data: { id: "folder-1", name: "Reports", fileCount: 2 },
            },
          ],
        },
        undefined,
      );
      await advance(40_000);
      expectScreen({ error: UNCONFIRMED });
      server.postMode = "stream";
      expect(tab.runtime.resendUnconfirmed(undefined)).not.toBeNull();
      await advance(0);
      server.publish();
      await advance(30_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] });
      // The same request again: its id, its file and its folder.
      const [first, again] = server.posts;
      expect(server.posts).toHaveLength(2);
      expect(again.message_id).toBe(first.message_id);
      expect(first.file_ids).toEqual(["file-1"]);
      expect(again.file_ids).toEqual(first.file_ids);
      expect(first.folder_ids).toEqual(["folder-1"]);
      expect(again.folder_ids).toEqual(first.folder_ids);
    },
  );

  cell(
    "S19p",
    "reported while a later send waits on its answer, is sent again once that send is answered",
    async () => {
      server.postMode = "hang-before-send";
      tab.send(P1);
      // A later send goes out while P1's checks run, and its answer is slow.
      await advance(30_500);
      server.postMode = "stream";
      server.respondAfterMs = 6_000;
      tab.send(P2);
      await advance(5_000);
      expectScreen({ error: UNCONFIRMED });
      expect(tab.runtime.resendUnconfirmed(undefined)).toBeNull();
      expect(tab.screen().chat).toEqual([`user: ${P1}`, `user: ${P2}`]);
      // P2's answer arrives and its turn ends; then P1 goes out again.
      await advance(2_000);
      server.publish();
      await advance(10_000);
      expect(tab.runtime.resendUnconfirmed(undefined)).not.toBeNull();
      await advance(0);
      server.publish();
      await advance(30_000);
      expectScreen({
        error: null,
        chat: [`user: ${P2}`, ANSWER, `user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "S19b",
    "sent again before its late first request reaches the server, starts one turn with one prompt row",
    async () => {
      server.postMode = "late-after-receive";
      tab.send(P1);
      await advance(38_000);
      expectScreen({ error: UNCONFIRMED });
      server.postMode = "stream";
      tab.runtime.resendUnconfirmed(undefined);
      await advance(0);
      // The first request reaches the server at 40 s, after the resend.
      await advance(5_000);
      server.publish();
      await advance(30_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] });
      expect(server.turns).toHaveLength(1);
      expect(server.history.filter((row) => row.content === P1)).toHaveLength(
        1,
      );
    },
  );

  cell(
    "S19l",
    "whose late request was answered after it was reported, no longer offers to send it again",
    async () => {
      server.postMode = "late-after-receive";
      tab.send(P1);
      await advance(38_000);
      expectScreen({ error: UNCONFIRMED });
      // The server took it at 40 s and answered; the page reads the chat.
      await advance(4_000);
      server.publish();
      await advance(1_000);
      tab.runtime.observe(server.view());
      await advance(1_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] });
      expect(tab.runtime.resendUnconfirmed(undefined)).toBeNull();
    },
  );

  cell(
    "S19u",
    "while its resend waits on an answer, refuses a new send rather than dropping it",
    async () => {
      server.postMode = "hang-before-send";
      tab.send(P1);
      await advance(40_000);
      expectScreen({ error: UNCONFIRMED });
      tab.runtime.resendUnconfirmed(undefined);
      await advance(0);
      await expect(tab.runtime.send({ text: P2 }, undefined)).rejects.toThrow(
        SendInFlightError,
      );
      await advance(0);
      // P1's two requests are all that went out: the refused send made none.
      expect(server.posts.map((post) => post.message)).toEqual([P1, P1]);
      expect(tab.screen().chat).toEqual([`user: ${P1}`]);
    },
  );

  cell(
    "S19a",
    "whose request reached the server only after it was reported, sent again, shows once with its answer",
    async () => {
      server.postMode = "late-after-receive";
      tab.send(P1);
      await advance(38_000);
      expectScreen({ error: UNCONFIRMED });
      // The server takes the first request at 40 s and starts its turn.
      await advance(4_000);
      server.postMode = "stream";
      expect(tab.runtime.resendUnconfirmed(undefined)).not.toBeNull();
      await advance(0);
      server.publish();
      await advance(30_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] });
      expect(server.posts).toHaveLength(2);
      expect(server.posts[1].message_id).toBe(server.posts[0].message_id);
    },
  );

  cell(
    "S19h",
    "whose request hung while hidden, never reaching the server, is reported on return",
    async () => {
      server.postMode = "hang-before-send";
      setVisibility("hidden");
      tab.send(P1);
      await advance(60_000);
      setVisibility("visible");
      await advance(15_000);
      expectScreen({ error: UNCONFIRMED, notice: null, phase: "failed" });
    },
  );

  cell(
    "S21",
    "lost while hidden, its prompt past the session's window on return, stays above its turn",
    async () => {
      server.postMode = "lost-after-receive";
      server.window = 2;
      setVisibility("hidden");
      tab.send(P1);
      await advance(60_000);
      server.publish(19);
      setVisibility("visible");
      await advance(2_000);
      expectScreen({ error: null });
      const chat = tab.screen().chat;
      expect(chat[0]).toBe(`user: ${P1}`);
      expect(chat.filter((line) => line === `user: ${P1}`)).toHaveLength(1);
    },
  );

  cell(
    "S21w",
    "lost while hidden, its row past the session's window under another tab's new turn on return, is not reported",
    async () => {
      const other = openTab();
      server.postMode = "lost-after-receive";
      server.window = 2;
      setVisibility("hidden");
      tab.send(P1);
      await advance(0);
      server.publish();
      await advance(30_000);
      server.postMode = "stream";
      other.send(P2);
      await advance(0);
      // The window shows the other tab's prompt, but not where P1's row is.
      setVisibility("visible");
      await advance(2_000);
      server.publish();
      await advance(30_000);
      expectScreen({ error: null });
    },
  );

  cell(
    "S23",
    "whose response was lost, read just before its prompt row landed, stays above its answer",
    async () => {
      server.postMode = "lost-after-receive";
      // The send's first check reads the turn but not yet the prompt row.
      server.lagRowsOnce = 1;
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(2_000);
      expectScreen({
        error: null,
        phase: "live",
        chat: [`user: ${P1}`, FIRST_BLOCK],
      });
      server.publish();
      await advance(30_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] });
    },
  );

  cell(
    "S23b",
    "a later message whose response was lost, read just before its prompt row landed, stays above its answer",
    async () => {
      tab.send(P1);
      await advance(0);
      server.publish();
      await advance(15_000);
      server.postMode = "lost-after-receive";
      // The read ends at the last reply, as if the new prompt were not written.
      server.lagRowsOnce = 1;
      tab.send(P2);
      await advance(0);
      server.publish(9);
      await advance(2_000);
      expectScreen({
        error: null,
        phase: "live",
        chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, FIRST_BLOCK],
      });
      server.publish();
      await advance(30_000);
      expectScreen({
        error: null,
        chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, ANSWER],
      });
    },
  );

  cell(
    "S24",
    "lost, its check's read failing, is still reported when its retry finds a later send's turn running",
    async () => {
      server.postMode = "lost-before-send";
      server.failViews = 1;
      tab.send(P1);
      await advance(0);
      server.postMode = "stream";
      tab.send(P2);
      await advance(0);
      server.publish(9);
      // P1's retry fires while P2's turn runs.
      await advance(2_000);
      server.publish();
      await advance(30_000);
      expectScreen({ error: "Failed to fetch" });
    },
  );

  cell(
    "S24c",
    "lost, its check's read failing, is still reported when a later send's dropped stream takes the retry",
    async () => {
      server.postMode = "lost-before-send";
      server.failViews = 1;
      tab.send(P1);
      await advance(0);
      server.postMode = "stream";
      tab.send(P2);
      await advance(0);
      server.publish(9);
      await advance(100);
      // P2's stream drops before P1's retry fires, and its own retry replaces it.
      server.connections[0].fail();
      await advance(5_000);
      server.publish();
      await advance(30_000);
      expectScreen({ error: "Failed to fetch" });
    },
  );

  cell(
    "S24w",
    "lost, its check's read failing, is still reported when the tab's return reconnects a later send's stream before the retry",
    async () => {
      server.postMode = "lost-before-send";
      server.failViews = 1;
      tab.send(P1);
      await advance(0);
      server.postMode = "stream";
      tab.send(P2);
      await advance(0);
      server.publish(9);
      await advance(100);
      // P2's stream drops while the tab is hidden, and the tab is back before
      // P1's retry fires: the reconnect replaces that retry.
      setVisibility("hidden");
      server.connections[0].fail();
      await advance(100);
      setVisibility("visible");
      await advance(5_000);
      server.publish();
      await advance(30_000);
      // P2 arrived and its answer follows it; P1 never did and has none.
      expectScreen({
        error: "Failed to fetch",
        chat: [`user: ${P1}`, `user: ${P2}`, ANSWER],
      });
    },
  );

  cell(
    "S15c",
    "two sends lost while offline are both checked once the network returns",
    async () => {
      server.postMode = "lost-before-send";
      setOnline(false);
      tab.send(P1);
      await advance(10_000);
      // The second reaches the server, but its response is lost too.
      server.postMode = "lost-after-receive";
      tab.send(P2);
      await advance(10_000);
      setOnline(true);
      await advance(5_000);
      server.publish();
      await advance(60_000);
      // P2 arrived and its answer follows it; P1 never did and has none.
      expectScreen({
        error: "Failed to fetch",
        chat: [`user: ${P1}`, `user: ${P2}`, ANSWER],
      });
    },
  );

  cell(
    "S19o",
    "lost while offline, then stopped, shows no error when the network returns",
    async () => {
      server.postMode = "lost-before-send";
      setOnline(false);
      tab.send(P1);
      await advance(10_000);
      tab.runtime.stop();
      await advance(1_000);
      setOnline(true);
      await advance(30_000);
      expectScreen({ error: null });
    },
  );

  cell(
    "S24r",
    "lost, its checks' reads failing for 25 minutes, is still reported when a check finds another's turn followed",
    async () => {
      server.postMode = "lost-before-send";
      server.viewsDown = true;
      tab.send(P1);
      await advance(25 * 60_000);
      // Another device's send starts a turn, and a page read the tab makes
      // during its next check shows it; that check's own read still fails.
      server.startTurn(P2);
      server.beforeViewOnce = () => {
        server.viewsDown = false;
        const view = server.view();
        server.viewsDown = true;
        tab.runtime.observe(view);
      };
      await advance(60_000);
      server.viewsDown = false;
      server.publish();
      await advance(60_000);
      expectScreen({ error: "Failed to fetch" });
    },
  );

  cell(
    "S24h",
    "lost while hidden, is still reported when the page showed another's turn before the tab came back",
    async () => {
      setVisibility("hidden");
      server.postMode = "lost-before-send";
      tab.send(P1);
      await advance(1_000);
      server.startTurn(P2);
      tab.runtime.observe(server.view());
      await advance(1_000);
      setVisibility("visible");
      await advance(2_000);
      server.publish();
      await advance(60_000);
      expectScreen({ error: "Failed to fetch" });
    },
  );

  cell(
    "S25",
    "the same text sent again after a turn that wrote no reply is its own message, its response lost",
    async () => {
      tab.send(P1);
      await advance(0);
      // The turn dies before it writes a reply: only its prompt row is kept.
      server.stopWithoutFinish();
      await advance(30_000);
      server.postMode = "lost-after-receive";
      tab.send(P1);
      await advance(0);
      server.publish();
      await advance(30_000);
      const chat = tab.screen().chat;
      expectScreen({ error: null });
      expect(chat.slice(-2)).toEqual([`user: ${P1}`, ANSWER]);
      expect(chat.filter((line) => line === `user: ${P1}`)).toHaveLength(2);
    },
  );

  cell(
    "S26",
    "a new chat's first message, sent before any view named its user, whose response was lost, is followed into its turn",
    async () => {
      server.namesUser = false;
      const fresh = openTab();
      server.postMode = "lost-after-receive";
      fresh.send(P1);
      // The page's next reads name the user; the send has no row id.
      server.namesUser = true;
      await advance(0);
      server.publish(9);
      await advance(2_000);
      expectScreen(
        { error: null, phase: "live", chat: [`user: ${P1}`, FIRST_BLOCK] },
        fresh,
      );
      server.publish();
      await advance(30_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] }, fresh);
    },
  );

  cell(
    "S20",
    "an expert's first message whose response was lost is followed into its turn",
    async () => {
      server.postMode = "lost-after-receive";
      tab.send(P1, KICKOFF);
      await advance(0);
      server.publish(9);
      await advance(2_000);
      expectScreen({
        error: null,
        phase: "live",
        chat: [`user: ${P1}`, FIRST_BLOCK],
      });
    },
  );
});

describe("a resume", () => {
  cell(
    "R1",
    "refused 401 ends with the sign-in error, without the connection toast",
    async () => {
      await streamFirstBlock();
      server.resumeMode = { status: 401, body: { detail: "Unauthorized" } };
      server.connections[0].fail();
      await advance(180_000);
      expectScreen({ error: SIGN_IN_AGAIN, toasts: 0 });
    },
  );

  cell("R2", "refused 404 ends, without the connection toast", async () => {
    await streamFirstBlock();
    server.resumeMode = { status: 404, body: { detail: "Session not found" } };
    server.connections[0].fail();
    await advance(180_000);
    expectScreen({ phase: "failed", toasts: 0 });
  });

  cell(
    "R3",
    "signed out mid-turn ends with the sign-in error, without the connection toast",
    async () => {
      await streamFirstBlock();
      auth.signedIn = false;
      server.viewsDown = true;
      server.connections[0].fail();
      await advance(180_000);
      expectScreen({ error: SIGN_IN_AGAIN, toasts: 0 });
    },
  );

  cell(
    "R4",
    "after a drop, while the turn ended meanwhile, shows the answer",
    async () => {
      await streamFirstBlock();
      server.connections[0].fail();
      server.publish();
      await advance(5_000);
      expectScreen({
        error: null,
        notice: null,
        toasts: 0,
        phase: "finished",
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell(
    "R5",
    "of a turn stopped elsewhere while hidden ends it inline",
    async () => {
      await streamFirstBlock();
      setVisibility("hidden");
      server.connections[0].fail();
      server.stopWithoutFinish();
      await advance(30_000);
      setVisibility("visible");
      await advance(30_000);
      expectScreen({ error: null, notice: null, toasts: 0, phase: "finished" });
    },
  );
});

describe("a turn this tab did not start", () => {
  cell(
    "F1",
    "an approval wake whose turn starts 6 s later is followed",
    async () => {
      await finishFirstTurn();
      tab.runtime.followBackendTurn();
      await advance(6_000);
      server.startTurn("(approval answered)");
      server.publish(9);
      await advance(10_000);
      expectScreen({ phase: "live" });
    },
  );

  cell(
    "F2",
    "started by the server after a failed send clears the send's error",
    async () => {
      server.postMode = "lost-before-send";
      tab.send(P1);
      await advance(15_000);
      tab.runtime.followBackendTurn();
      server.startTurn("(server-started)");
      server.publish(9);
      await advance(3_000);
      expectScreen({ phase: "live", error: null });
    },
  );

  cell(
    "F3",
    "started on another device while this tab is idle and visible is followed",
    async () => {
      await finishFirstTurn();
      server.startTurn(P2);
      server.publish(9);
      await advance(60_000);
      expectScreen({ phase: "live" });
    },
  );

  cell(
    "F3h",
    "started elsewhere while this tab was hidden is followed on return",
    async () => {
      await finishFirstTurn();
      setVisibility("hidden");
      await advance(60_000);
      server.startTurn(P2);
      server.publish(9);
      await advance(60_000);
      setVisibility("visible");
      await advance(5_000);
      expectScreen({ phase: "live" });
    },
  );
});

describe("two tabs on one chat", () => {
  cell(
    "M1",
    "a send from a stale tab while the other tab's turn runs keeps both prompts in order",
    async () => {
      const stale = openTab();
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      stale.send(P2);
      await advance(3_000);
      server.publish();
      await advance(5_000);
      server.publish();
      await advance(30_000);
      expectScreen(
        { error: null, chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, ANSWER] },
        stale,
      );
    },
  );

  cell(
    "M2",
    "a tab opened onto the other tab's running turn draws it once",
    async () => {
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      const second = openTab();
      await advance(100);
      server.publish();
      await advance(5_000);
      expectScreen({ error: null, chat: [`user: ${P1}`, ANSWER] }, second);
    },
  );

  cell(
    "M3",
    "a stale tab's send that never reached the server, while the other tab's turn runs, is reported below that turn",
    async () => {
      const stale = openTab();
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(3_000);
      server.publish();
      await advance(30_000);
      expectScreen(
        {
          error: "Failed to fetch",
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`],
        },
        stale,
      );
    },
  );

  cell(
    "M3q",
    "a stale tab's send that never reached the server is reported once the other tab's turn ends, though the page re-reads nothing",
    async () => {
      const stale = openTab(0, false);
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(3_000);
      server.publish();
      await advance(30_000);
      expectScreen(
        {
          error: "Failed to fetch",
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`],
        },
        stale,
      );
    },
  );

  cell(
    "M3f",
    "a stale tab's send that never reached the server is reported after the other tab's turn ends, though every end-of-turn read fails",
    async () => {
      const stale = openTab(0, false);
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(3_000);
      // The turn's end is seen, but no read succeeds until later.
      server.viewsDown = true;
      server.publish();
      await advance(5_000);
      server.viewsDown = false;
      await advance(60_000);
      expectScreen({ error: "Failed to fetch" }, stale);
    },
  );

  cell(
    "M3u",
    "two stale-tab sends behind the other tab's turn are both checked when their reads fail while the tab is hidden",
    async () => {
      const stale = openTab(300);
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(0);
      server.postMode = "lost-after-receive";
      stale.send(P3);
      await advance(3_000);
      // P1's turn ends while the tab is hidden and every read fails.
      setVisibility("hidden");
      server.viewsDown = true;
      server.publish();
      await advance(10_000);
      server.viewsDown = false;
      setVisibility("visible");
      await advance(5_000);
      server.publish();
      await advance(60_000);
      // P3 arrived and its answer follows it; P2 never did and has none.
      expectScreen(
        {
          error: "Failed to fetch",
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, `user: ${P3}`, ANSWER],
        },
        stale,
      );
    },
  );

  cell(
    "M3o",
    "two stale-tab sends behind the other tab's turn are both checked when their reads fail while offline",
    async () => {
      const stale = openTab(300);
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(0);
      server.postMode = "lost-after-receive";
      stale.send(P3);
      await advance(3_000);
      // P1's turn ends while the network is down.
      setOnline(false);
      server.viewsDown = true;
      server.publish();
      await advance(10_000);
      server.viewsDown = false;
      setOnline(true);
      await advance(5_000);
      server.publish();
      await advance(60_000);
      // P3 arrived and its answer follows it; P2 never did and has none.
      expectScreen(
        {
          error: "Failed to fetch",
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, `user: ${P3}`, ANSWER],
        },
        stale,
      );
    },
  );

  cell(
    "M3n",
    "a stale tab's lost send still reports its own failure after the message sent after it lands",
    async () => {
      const stale = openTab();
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(3_000);
      server.postMode = "stream";
      stale.send(P3);
      await advance(3_000);
      server.publish();
      await advance(5_000);
      server.publish();
      await advance(30_000);
      expectScreen({ error: "Failed to fetch" }, stale);
    },
  );

  cell(
    "M3d",
    "two stale-tab sends behind the other tab's turn are both checked once it ends",
    async () => {
      const stale = openTab(300);
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(0);
      // Sent before P2's first check reads the session; queued behind P1's turn.
      server.postMode = "lost-after-receive";
      stale.send(P3);
      await advance(3_000);
      server.publish();
      await advance(5_000);
      server.publish();
      await advance(30_000);
      // P3 arrived and its answer follows it; P2 never did and has none.
      expectScreen(
        {
          error: "Failed to fetch",
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, `user: ${P3}`, ANSWER],
        },
        stale,
      );
    },
  );

  cell(
    "M3r",
    "two stale-tab sends behind the other tab's turn are both checked when their reads fail at first",
    async () => {
      const stale = openTab(300);
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(0);
      server.postMode = "lost-after-receive";
      stale.send(P3);
      await advance(3_000);
      server.publish();
      await advance(5_000);
      server.publish();
      // The end-of-turn read is back, and both sends' checks are reading.
      await advance(900);
      server.viewsDown = true;
      await advance(500);
      server.viewsDown = false;
      await advance(30_000);
      // P3 arrived and its answer follows it; P2 never did and has none.
      expectScreen(
        {
          error: "Failed to fetch",
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, `user: ${P3}`, ANSWER],
        },
        stale,
      );
    },
  );

  cell(
    "M3s",
    "a stale tab's lost send is let go when the user stops the turn it waits on",
    async () => {
      const stale = openTab();
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(3_000);
      stale.runtime.stop();
      server.publish();
      await advance(1_000);
      stale.runtime.observe(server.view());
      await advance(30_000);
      expectScreen({ error: null }, stale);
    },
  );

  cell(
    "M3t",
    "a stale tab's lost send, stopped while its checks run, isn't reported when another tab's turn starts",
    async () => {
      const stale = openTab();
      server.postMode = "lost-before-send";
      stale.send(P2);
      await advance(1_000);
      stale.runtime.stop();
      server.postMode = "stream";
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(1_000);
      server.publish();
      await advance(30_000);
      expectScreen({ error: null }, stale);
    },
  );

  cell(
    "M4",
    "a stale tab's send that never reached the server doesn't take the other tab's prompt that starts with its words",
    async () => {
      const stale = openTab();
      const longer = `${P1} Keep it short.`;
      tab.send(longer);
      await advance(0);
      server.publish();
      await advance(3_000);
      server.postMode = "lost-before-send";
      stale.send(P1);
      await advance(15_000);
      expectScreen(
        {
          error: "Failed to fetch",
          chat: [`user: ${longer}`, ANSWER, `user: ${P1}`],
        },
        stale,
      );
    },
  );

  cell(
    "M5",
    "a stale tab's send queued behind the other tab's turn, its response lost, is answered in order",
    async () => {
      const stale = openTab();
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-after-receive";
      stale.send(P2);
      // Longer than the send's own checks: only the turn's end can settle it.
      await advance(10_000);
      expectScreen({ error: null }, stale);
      server.publish();
      await advance(5_000);
      server.publish();
      await advance(30_000);
      expectScreen(
        {
          error: null,
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, ANSWER],
        },
        stale,
      );
    },
  );

  cell(
    "M5p",
    "a stale tab's queued send gets no error when the page attaches the other tab's turn before the send's own check",
    async () => {
      const stale = openTab(300);
      tab.send(P1);
      await advance(0);
      server.publish(9);
      await advance(100);
      server.postMode = "lost-after-receive";
      stale.send(P2);
      await advance(0);
      stale.runtime.observe(server.view());
      await advance(10_000);
      expectScreen({ error: null }, stale);
      server.publish();
      await advance(5_000);
      server.publish();
      await advance(30_000);
      expectScreen(
        {
          error: null,
          chat: [`user: ${P1}`, ANSWER, `user: ${P2}`, ANSWER],
        },
        stale,
      );
    },
  );
});

describe("the page lifecycle", () => {
  cell(
    "L1",
    "a back/forward-cache restore after the stream died shows the answer",
    async () => {
      await streamFirstBlock();
      setVisibility("hidden");
      server.connections[0].fail();
      server.publish();
      await advance(120_000);
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        configurable: true,
      });
      window.dispatchEvent(new Event("pageshow"));
      await advance(5_000);
      expectScreen({
        error: null,
        toasts: 0,
        phase: "finished",
        chat: [`user: ${P1}`, ANSWER],
      });
    },
  );

  cell("L2", "a network switch mid-turn recovers silently", async () => {
    await streamFirstBlock();
    server.resumeMode = "network-error";
    server.connections[0].fail();
    await advance(3_000);
    server.resumeMode = "stream";
    await advance(10_000);
    server.publish();
    await advance(5_000);
    expectScreen({ error: null, notice: null, toasts: 0, phase: "finished" });
  });

  cell(
    "L3",
    "a send whose page froze for 20 s shows the answer soon after return",
    async () => {
      server.postMode = "silent-after-receive";
      tab.send(P1);
      await advance(0);
      setVisibility("hidden");
      server.publish();
      await advance(20_000);
      setVisibility("visible");
      await advance(35_000);
      expectScreen({ chat: [`user: ${P1}`, ANSWER] });
    },
  );
});

function cell(id: string, title: string, run: () => Promise<void>) {
  CASES.add(id);
  const wrong = KNOWN_WRONG[id];
  if (wrong) it.fails(`${id}: ${title} (known wrong: ${wrong})`, run);
  else it(`${id}: ${title}`, run);
}

function expectScreen(should: Partial<Screen>, on: Tab = tab) {
  const screen = on.screen();
  const seen = Object.fromEntries(
    Object.keys(should).map((key) => [key, screen[key as keyof Screen]]),
  );
  expect(seen).toEqual(should);
}

/** Runs `fn` as the `count`-th session read from now starts. */
function onRead(count: number, fn: () => void) {
  server.beforeViewOnce = () => {
    if (count <= 1) fn();
    else onRead(count - 1, fn);
  };
}

function openTab(viewLatencyMs = 0, echoes = true) {
  const opened = new Tab(server, viewLatencyMs, echoes);
  tabs.push(opened);
  return opened;
}

async function streamFirstBlock() {
  tab.send(P1);
  await advance(0);
  server.publish(9);
  await advance(100);
}

async function finishFirstTurn() {
  tab.send(P1);
  await advance(0);
  server.publish();
  await advance(3_000);
}

async function advance(ms: number) {
  await vi.advanceTimersByTimeAsync(ms);
}

function setVisibility(state: "hidden" | "visible") {
  Object.defineProperty(document, "visibilityState", {
    value: state,
    configurable: true,
  });
  document.dispatchEvent(new Event("visibilitychange"));
}

function setOnline(online: boolean) {
  vi.spyOn(navigator, "onLine", "get").mockReturnValue(online);
  window.dispatchEvent(new Event(online ? "online" : "offline"));
}
