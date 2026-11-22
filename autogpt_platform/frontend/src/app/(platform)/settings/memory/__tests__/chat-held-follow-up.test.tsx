import {
  getGetV2GetPendingMessagesMockHandler200,
  getPostV2CreateSessionMockHandler200,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetV2GetPendingReviewsForChatSessionMockHandler } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  createStreamHold,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";
import { screen, waitFor, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { UIMessageChunk } from "ai";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { MEMORY_CHAT_PROMPTS } from "../helpers";
import { renderMemoryPage, setupMemoryChat } from "./chat-test-helpers";
// After the helpers, whose mocks must be in place before the copilot modules load.
import { getOrCreateCopilotChatRuntime } from "@/app/(platform)/copilot/copilotChatRegistry";

// A follow-up sent while Otto's answer is still drawing, after the backend's
// turn has ended, is refused by `/messages/pending` with a 409. Sending it at
// once starts a second turn on top of the answer still on screen (#17250).
// The panel must hold it as a queued message and send it once the answer has
// settled. Every open starts a new session, so one left behind comes back in
// the composer of the next chat about the same memory.

const ANSWER_OPENING = "You run a candle brand called Emberline,";
const ANSWER_CLOSING = "and you send a summary email every Monday.";
const FULL_ANSWER = `${ANSWER_OPENING} ${ANSWER_CLOSING}`;
const FOLLOW_UP = "What else do you remember?";
const FOLLOW_UP_ANSWER = "That is everything I remember.";
const SUMMARY_PROMPT = MEMORY_CHAT_PROMPTS.summary;
const FIRST_CHAT = "memory-chat-1";
// Keyed by the memory, not the session: no session opens twice.
const HELD_COPY_KEY = "copilot-held-follow-ups:memory-chat:personal";

/** The first answer is parked before this chunk: its closing words. */
const MID_ANSWER = 4;

function summaryChunks(): UIMessageChunk[] {
  return [
    { type: "start", messageId: "summary-reply" },
    { type: "start-step" },
    { type: "text-start", id: "summary-text" },
    // Ends on a space: the typewriter smoothing holds back a partial word.
    { type: "text-delta", id: "summary-text", delta: `${ANSWER_OPENING} ` },
    { type: "text-delta", id: "summary-text", delta: ANSWER_CLOSING },
    { type: "text-end", id: "summary-text" },
    { type: "finish-step" },
    { type: "finish" },
  ];
}

interface Turn {
  sessionId: string;
  message: string;
}

let releaseAfterTest = () => {};

/**
 * Every open creates a session of its own, as the backend does. The first
 * answer holds mid-sentence until `finishAnswer`, and the queue refuses every
 * follow-up: the turn it would have joined has ended.
 */
function serveMemoryChat() {
  const answer = createStreamHold(MID_ANSWER);
  // A test that fails mid-answer must not leave the stream open.
  releaseAfterTest = answer.release;
  const turns: Turn[] = [];
  const queued: string[] = [];
  let created = 0;
  server.use(
    getPostV2CreateSessionMockHandler200(() =>
      getPostV2CreateSessionResponseMock200({ id: `memory-chat-${++created}` }),
    ),
    getGetV2GetPendingMessagesMockHandler200({ count: 0, messages: [] }),
    getGetV2GetPendingReviewsForChatSessionMockHandler([]),
    http.post(
      "*/api/chat/sessions/:sessionId/stream",
      async ({ params, request }) => {
        const { message } = (await request.json()) as { message: string };
        turns.push({ sessionId: String(params.sessionId), message });
        const chunks =
          message === FOLLOW_UP
            ? assistantTextChunks(FOLLOW_UP_ANSWER, {
                messageId: `follow-up-reply-${turns.length}`,
              })
            : summaryChunks();
        return streamSseResponse(chunks, {
          hold: turns.length === 1 ? answer.hold : undefined,
          abortSignal: request.signal,
        });
      },
    ),
    http.post(
      "*/api/chat/sessions/:sessionId/messages/pending",
      async ({ request }) => {
        const { message } = (await request.json()) as { message: string };
        queued.push(message);
        return HttpResponse.json(
          {
            detail:
              "Session has no active turn. Start a new turn with POST /stream.",
          },
          { status: 409 },
        );
      },
    ),
  );
  return { turns, queued, finishAnswer: answer.release };
}

beforeEach(setupMemoryChat);

afterEach(() => {
  releaseAfterTest();
  window.sessionStorage.clear();
});

function memoryChatPanel() {
  return screen.findByRole("complementary", { name: "Memory chat panel" });
}

function composer(panel: HTMLElement) {
  return within(panel).getByLabelText<HTMLTextAreaElement>(
    "Chat message input",
  );
}

/** The text shows as one of the panel's queued messages. */
function isQueued(panel: HTMLElement, text: string) {
  return within(panel)
    .queryAllByText("Queued")
    .some((label) => label.parentElement?.textContent?.includes(text));
}

/**
 * The copy the panel keeps of a follow-up it holds, for the next chat about
 * the same memory. Written once the 409 has been handled, unlike the chip,
 * which goes up before the queue request does.
 */
function heldCopy() {
  return window.sessionStorage.getItem(HELD_COPY_KEY);
}

async function expectHeld(panel: HTMLElement) {
  await waitFor(() => expect(heldCopy()).toBe(JSON.stringify([FOLLOW_UP])));
  expect(isQueued(panel, FOLLOW_UP)).toBe(true);
}

/** Opens the summary chat and sends a follow-up while its answer draws. */
async function sendFollowUpMidAnswer() {
  const user = userEvent.setup();
  await user.click(
    await screen.findByRole("button", { name: "View my summary" }),
  );
  const panel = await memoryChatPanel();
  await within(panel).findByText(ANSWER_OPENING);
  await user.type(composer(panel), FOLLOW_UP);
  await user.click(
    within(panel).getByRole("button", { name: "Queue message" }),
  );
  return { user, panel };
}

/** The first chat's answer has streamed to its end, on screen or not. */
async function waitForFirstAnswerToEnd() {
  await waitFor(() =>
    expect(getOrCreateCopilotChatRuntime(FIRST_CHAT).chat.status).toBe("ready"),
  );
}

async function selectMemory(name: RegExp) {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Memory scope" }));
  await user.click(await screen.findByRole("menuitem", { name }));
}

/** Back in the composer, with one toast that says why. */
async function expectBackInComposer(panel: HTMLElement) {
  await waitFor(() => expect(composer(panel).value).toBe(FOLLOW_UP));
  expect(await screen.findByText("Follow-up not sent")).toBeDefined();
  expect(
    screen.getByText(
      "You left the chat before it could go out. It's back in the composer.",
    ),
  ).toBeDefined();
}

describe("Memory chat — a follow-up the queue refuses mid-answer", () => {
  it("waits as a queued message until Otto's answer has finished, then goes out once", async () => {
    const backend = serveMemoryChat();
    renderMemoryPage();

    const { panel } = await sendFollowUpMidAnswer();
    // The panel has handled the 409 one way or the other: it holds the
    // follow-up, or it started a turn of its own on top of the answer.
    await waitFor(() =>
      expect(heldCopy() !== null || backend.turns.length > 1).toBe(true),
    );
    expect(backend.queued).toEqual([FOLLOW_UP]);
    expect(backend.turns.map((turn) => turn.message)).toEqual([SUMMARY_PROMPT]);
    expect(isQueued(panel, FOLLOW_UP)).toBe(true);

    backend.finishAnswer();
    // Past the stream's end, the post-finish probe waits half a second.
    await within(panel).findByText(FOLLOW_UP_ANSWER, undefined, {
      timeout: 10_000,
    });

    expect(backend.turns).toEqual([
      { sessionId: FIRST_CHAT, message: SUMMARY_PROMPT },
      { sessionId: FIRST_CHAT, message: FOLLOW_UP },
    ]);
    // The answer drew to its end, and shows once.
    expect(within(panel).getAllByText(FULL_ANSWER)).toHaveLength(1);
    expect(within(panel).getAllByText(FOLLOW_UP)).toHaveLength(1);
    expect(isQueued(panel, FOLLOW_UP)).toBe(false);
    expect(backend.queued).toEqual([FOLLOW_UP]);
    expect(heldCopy()).toBeNull();
  });

  it("keeps it when you leave memory settings first, and puts it back in the composer of the next memory chat", async () => {
    const backend = serveMemoryChat();
    const firstVisit = renderMemoryPage();
    const { panel } = await sendFollowUpMidAnswer();
    await expectHeld(panel);

    firstVisit.unmount();
    backend.finishAnswer();
    await waitForFirstAnswerToEnd();
    expect(heldCopy()).toBe(JSON.stringify([FOLLOW_UP]));

    renderMemoryPage();
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "View my summary" }));
    const nextChat = await memoryChatPanel();
    await expectBackInComposer(nextChat);
    expect(heldCopy()).toBeNull();
    await within(nextChat).findByText(FULL_ANSWER);
    expect(backend.turns).toEqual([
      { sessionId: FIRST_CHAT, message: SUMMARY_PROMPT },
      { sessionId: "memory-chat-2", message: SUMMARY_PROMPT },
    ]);
  });

  it("keeps it for its own memory when you switch to another one first", async () => {
    const backend = serveMemoryChat();
    renderMemoryPage();
    const { user, panel } = await sendFollowUpMidAnswer();
    await expectHeld(panel);

    await selectMemory(/Growth Marketer/);
    await waitFor(() =>
      expect(
        screen.queryByRole("complementary", { name: "Memory chat panel" }),
      ).toBeNull(),
    );
    backend.finishAnswer();
    await waitForFirstAnswerToEnd();

    await user.click(
      await screen.findByRole("button", { name: "View Maria's summary" }),
    );
    const mariasChat = await memoryChatPanel();
    await within(mariasChat).findByText(FULL_ANSWER);
    expect(composer(mariasChat).value).toBe("");

    await selectMemory(/Your account memory/);
    await user.click(
      await screen.findByRole("button", { name: "View my summary" }),
    );
    await expectBackInComposer(await memoryChatPanel());
    expect(backend.turns.map((turn) => turn.message)).toEqual([
      SUMMARY_PROMPT,
      SUMMARY_PROMPT,
      SUMMARY_PROMPT,
    ]);
  });

  it("puts it back in the composer when you start a new chat about the same memory first", async () => {
    const backend = serveMemoryChat();
    renderMemoryPage();
    const { user, panel } = await sendFollowUpMidAnswer();
    await expectHeld(panel);

    await user.click(screen.getByRole("button", { name: "View my summary" }));
    const newChat = await memoryChatPanel();
    await expectBackInComposer(newChat);
    expect(isQueued(newChat, FOLLOW_UP)).toBe(false);

    backend.finishAnswer();
    await waitForFirstAnswerToEnd();
    await within(newChat).findByText(FULL_ANSWER);
    expect(backend.turns).toEqual([
      { sessionId: FIRST_CHAT, message: SUMMARY_PROMPT },
      { sessionId: "memory-chat-2", message: SUMMARY_PROMPT },
    ]);
  });
});
