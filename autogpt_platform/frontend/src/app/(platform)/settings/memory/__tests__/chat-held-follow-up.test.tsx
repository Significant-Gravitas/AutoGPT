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
// After the helpers, whose mocks must be in place before these load.
import { getOrCreateCopilotChatRuntime } from "@/app/(platform)/copilot/copilotChatRegistry";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { ImpersonationState } from "@/lib/impersonation";

// A follow-up sent while Otto's answer is still drawing, after the backend's
// turn has ended, is refused by `/messages/pending` with a 409. Sending it at
// once starts a second turn on top of the answer still on screen (#17250).
// The panel must hold it as a queued message and send it once the answer has
// settled. Every open starts a new session, so one left behind comes back in
// the composer of the next chat about the same memory, for the same account
// only.

const ANSWER_OPENING = "You run a candle brand called Emberline,";
const ANSWER_CLOSING = "and you send a summary email every Monday.";
const FULL_ANSWER = `${ANSWER_OPENING} ${ANSWER_CLOSING}`;
const FOLLOW_UP = "What else do you remember?";
const FOLLOW_UP_ANSWER = "That is everything I remember.";
const SUMMARY_PROMPT = MEMORY_CHAT_PROMPTS.summary;
const FIRST_CHAT = "memory-chat-1";

/** The first answer is parked before this chunk: its closing words. */
const MID_ANSWER = 4;

interface Identity {
  user: User;
  /** The user an admin is acting as. */
  actingAs?: string;
}

const USER_A: Identity = { user: { id: "user-a" } as User };
const USER_B: Identity = { user: { id: "user-b" } as User };
const ADMIN = { id: "admin-1", role: "admin" } as User;
const ADMIN_AS_A: Identity = { user: ADMIN, actingAs: "user-a" };

function signInAs({ user, actingAs }: Identity) {
  ImpersonationState.clear();
  if (actingAs) ImpersonationState.set(actingAs, user.id);
  useAuthStore.setState({ user });
}

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

// A test that fails midway must not leave a request open.
const releaseAfterTest: Array<() => void> = [];

/**
 * Every open creates a session of its own, as the backend does. The first
 * answer holds mid-sentence until `finishAnswer`, and the queue refuses every
 * follow-up: the turn it would have joined has ended. With `holdRefusal`, the
 * refusal waits for `refuse`.
 */
function serveMemoryChat({ holdRefusal = false } = {}) {
  const answer = createStreamHold(MID_ANSWER);
  const refusal = Promise.withResolvers<void>();
  if (!holdRefusal) refusal.resolve();
  releaseAfterTest.push(answer.release, refusal.resolve);
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
        await refusal.promise;
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
  return {
    turns,
    queued,
    finishAnswer: answer.release,
    refuse: refusal.resolve,
  };
}

beforeEach(() => {
  setupMemoryChat();
  signInAs(USER_A);
});

afterEach(() => {
  releaseAfterTest.splice(0).forEach((release) => release());
  ImpersonationState.clear();
  useAuthStore.setState({ user: null });
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
 * The copies the panel keeps of follow-ups it holds, for the next chat about
 * the same memory, whatever their keys. Written once the 409 has been
 * handled, unlike the chip, which goes up before the queue request does.
 */
function heldCopies() {
  const copies: string[] = [];
  for (let i = 0; i < window.sessionStorage.length; i++) {
    const key = window.sessionStorage.key(i);
    if (!key?.startsWith("copilot-held-follow-ups:")) continue;
    copies.push(window.sessionStorage.getItem(key) ?? "");
  }
  return copies;
}

const KEPT = [JSON.stringify([FOLLOW_UP])];

async function expectHeld(panel: HTMLElement) {
  await waitFor(() => expect(heldCopies()).toEqual(KEPT));
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

/** Leaves memory settings while a refused follow-up waits for the answer. */
async function leaveWithFollowUpHeld() {
  const backend = serveMemoryChat();
  const visit = renderMemoryPage();
  const { panel } = await sendFollowUpMidAnswer();
  await expectHeld(panel);
  visit.unmount();
  backend.finishAnswer();
  await waitForFirstAnswerToEnd();
  return backend;
}

/** Comes back to memory settings and opens the summary chat. */
async function openMemoryChatAgain() {
  renderMemoryPage();
  await userEvent
    .setup()
    .click(await screen.findByRole("button", { name: "View my summary" }));
  return memoryChatPanel();
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
      expect(heldCopies().length > 0 || backend.turns.length > 1).toBe(true),
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
    expect(heldCopies()).toEqual([]);
  });

  it("keeps it when you leave memory settings first, and puts it back in the composer of the next memory chat", async () => {
    const backend = await leaveWithFollowUpHeld();
    expect(heldCopies()).toEqual(KEPT);

    const nextChat = await openMemoryChatAgain();
    await expectBackInComposer(nextChat);
    expect(heldCopies()).toEqual([]);
    await within(nextChat).findByText(FULL_ANSWER);
    expect(backend.turns).toEqual([
      { sessionId: FIRST_CHAT, message: SUMMARY_PROMPT },
      { sessionId: "memory-chat-2", message: SUMMARY_PROMPT },
    ]);
  });

  it("keeps one refused after you closed the chat, and puts it back in the composer of the next memory chat", async () => {
    const backend = serveMemoryChat({ holdRefusal: true });
    renderMemoryPage();
    const { user, panel } = await sendFollowUpMidAnswer();
    await waitFor(() => expect(backend.queued).toEqual([FOLLOW_UP]));
    backend.finishAnswer();
    // Settled, so a follow-up held for this chat from now on goes out at once.
    await within(panel).findByText("Reply finished", undefined, {
      timeout: 10_000,
    });

    await user.click(
      within(panel).getByRole("button", { name: "Close memory chat" }),
    );
    backend.refuse();
    // Kept for the next chat: sent into the closed one, it would be gone.
    await waitFor(() => expect(heldCopies()).toEqual(KEPT));
    expect(backend.turns.map((turn) => turn.message)).toEqual([SUMMARY_PROMPT]);

    await user.click(screen.getByRole("button", { name: "View my summary" }));
    const nextChat = await memoryChatPanel();
    await expectBackInComposer(nextChat);
    expect(heldCopies()).toEqual([]);
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

    await selectMemory(/Growth Marketer$/);
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

    await selectMemory(/Your account memory — everything you do together$/);
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

  it("puts one refused only after a new chat about the same memory opened in that chat's composer", async () => {
    const backend = serveMemoryChat({ holdRefusal: true });
    renderMemoryPage();
    const { user } = await sendFollowUpMidAnswer();
    await waitFor(() => expect(backend.queued).toEqual([FOLLOW_UP]));

    await user.click(screen.getByRole("button", { name: "View my summary" }));
    const newChat = await memoryChatPanel();
    // Open and answered: it has long since taken what was waiting for it.
    await within(newChat).findByText(FULL_ANSWER);
    expect(composer(newChat).value).toBe("");

    backend.refuse();
    await expectBackInComposer(newChat);
    expect(heldCopies()).toEqual([]);

    backend.finishAnswer();
    await waitForFirstAnswerToEnd();
    expect(backend.turns.map((turn) => turn.message)).toEqual([
      SUMMARY_PROMPT,
      SUMMARY_PROMPT,
    ]);
  });

  it("puts one refused only after you left and opened a new chat about the same memory in that chat's composer", async () => {
    const backend = serveMemoryChat({ holdRefusal: true });
    const visit = renderMemoryPage();
    await sendFollowUpMidAnswer();
    await waitFor(() => expect(backend.queued).toEqual([FOLLOW_UP]));
    visit.unmount();

    const newChat = await openMemoryChatAgain();
    // Open and answered: it has long since taken what was waiting for it.
    await within(newChat).findByText(FULL_ANSWER);
    expect(composer(newChat).value).toBe("");

    backend.refuse();
    await expectBackInComposer(newChat);
    expect(heldCopies()).toEqual([]);

    backend.finishAnswer();
    await waitForFirstAnswerToEnd();
    expect(backend.turns.map((turn) => turn.message)).toEqual([
      SUMMARY_PROMPT,
      SUMMARY_PROMPT,
    ]);
  });
});

describe("Memory chat — a held follow-up stays with its account", () => {
  it.each([
    {
      when: "another account signs in on the tab",
      typedBy: USER_A,
      nextUp: USER_B,
    },
    {
      when: "the admin who acted as the user is back as themselves",
      typedBy: ADMIN_AS_A,
      nextUp: { user: ADMIN },
    },
    {
      when: "the user signs in on the tab where an admin acted as them",
      typedBy: ADMIN_AS_A,
      nextUp: USER_A,
    },
  ])(
    "doesn't come back in the next memory chat when $when",
    async ({ typedBy, nextUp }) => {
      signInAs(typedBy);
      const backend = await leaveWithFollowUpHeld();

      signInAs(nextUp);
      const nextChat = await openMemoryChatAgain();
      await within(nextChat).findByText(FULL_ANSWER);
      expect(composer(nextChat).value).toBe("");
      expect(screen.queryByText("Follow-up not sent")).toBeNull();
      // Still waiting for whoever typed it.
      expect(heldCopies()).toEqual(KEPT);
      expect(backend.turns.map((turn) => turn.message)).toEqual([
        SUMMARY_PROMPT,
        SUMMARY_PROMPT,
      ]);
    },
  );

  it("comes back when the admin acts as the same user again", async () => {
    signInAs(ADMIN_AS_A);
    await leaveWithFollowUpHeld();

    signInAs(ADMIN_AS_A);
    await expectBackInComposer(await openMemoryChatAgain());
    expect(heldCopies()).toEqual([]);
  });
});
