import type { Page } from "@playwright/test";
import {
  currentSessionID,
  firstWords,
  recordChatStreams,
  runningTurnID,
  startChat,
} from "./copilot-chat";
import {
  answerWordsIn,
  clickStop,
  errorCards,
  expectOnceEachInOrder,
  savedConversation,
  sendAsNewMessage,
  sendOutcome,
  watchCancel,
} from "./copilot-stop";
import { expect, test } from "./fixtures";
import { pinKnownFailure } from "./known-failure";
import afterStopScript from "./model-scripts/after-stop.json";
import stopDuringToolScript from "./model-scripts/stop-during-tool.json";
import stopMidReplyScript from "./model-scripts/stop-mid-reply.json";
import {
  expectEachTurnOnce,
  type JournalEntry,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

// A long answer, a word every WORD_DELAY_MS: about half a minute in full, so
// the Stop lands mid-reply, and a CLI left running after it would still be
// reading the answer when the next turn starts.
const STOPPED: ModelScript = stopMidReplyScript;
const ANSWER = STOPPED.turns[0].text ?? "";
const ANSWER_WORDS = ANSWER.split(" ").length;
const WORD_DELAY_MS = stopMidReplyScript.turns[0].stream.delay_ms;
const STREAMING_WORDS = 6;

// What the user sends after a Stop. It carries the after-stop script's own
// marker: a turn after a stopped one may resume the chat or get it as text.
const AFTER_STOP: ModelScript = afterStopScript;
const CORRECTION = AFTER_STOP.turns[0].expect_user_text ?? "";
const CORRECTION_ANSWER = AFTER_STOP.turns[0].text ?? "";

// A Countdown Timer block the model runs, set far longer than any test
// waits, so the Stop always lands while it runs. Its second turn answers the
// timer's result, which a stopped turn must never ask for.
const TOOL: ModelScript = stopDuringToolScript;
const TOOL_INTRO = TOOL.turns[0].text ?? "";
const TOOL_ANSWER = TOOL.turns[1].text ?? "";

const STOPPED_CARD = "Task stopped";

// wall-clock: three of the cancel route's 5 s windows
// (_CANCEL_CONFIRM_TIMEOUT_SECONDS in backend/api/features/chat/routes.py).
// After one, the route completes a stopped turn itself, so only a hang lasts
// this long.
const STOPPED_TURN_ENDS_WITHIN_MS = 15_000;
// wall-clock: two of the cancel route's 5 s windows. The executor confirms a
// Stop inside one, and the scripted model sees the dropped stream at its next
// word, a tenth of a second later; a CLI still reading is past both. The
// saved reply may hold what the model streamed in that time, counted in its
// words rather than on this runner's clock: on dev it kept 4-9 words past
// what the chat showed, and the slowest Stop took 5.5 s.
const STOPPED_REPLY_CUT_WITHIN_MS = 10_000;
const WORDS_AFTER_STOP = STOPPED_REPLY_CUT_WITHIN_MS / WORD_DELAY_MS;
// wall-clock: a new turn's CLI start and a one-line reply take a few
// seconds on CI; an unanswered message fails here, well inside the test's
// 90 s.
const ANSWERED_WITHIN_MS = 30_000;
// wall-clock: on dev, a stopped turn's CLI asks the model for its next step
// about 1.5 s after the Stop; this is ten times that.
const STOPPED_TURN_QUIET_FOR_MS = 15_000;
// The reloaded chat has drawn the stopped card below the timer's row by
// then; this only gives a row that shouldn't be there time to go.
const STOPPED_TOOL_CHECK_MS = 5_000;

test.describe("copilot stop journey", () => {
  test.beforeAll(requireScriptedModel);

  test(
    "a correction sent right after Stop is answered, after the reply as it stopped",
    {
      annotation: [
        {
          type: "issue",
          description:
            "https://github.com/Significant-Gravitas/AutoGPT/pull/16673",
        },
        {
          type: "issue",
          description:
            "https://github.com/Significant-Gravitas/AutoGPT/pull/16729",
        },
      ],
    },
    async ({ page, journeyUser: _user }) => {
      const run = scriptedRun("stop-mid-reply");
      const after = scriptedRun("after-stop");
      const firstMessage = `Tell me the whole story of the garden. ${run.marker}`;
      const correction = `${CORRECTION} ${after.marker}`;
      const transcript = page.getByRole("log");
      await recordChatStreams(page);

      await startChat(page, firstMessage);
      await expect(
        transcript.getByText(firstWords(ANSWER, STREAMING_WORDS)),
        "the answer should start streaming",
      ).toBeVisible({ timeout: 60_000 });
      const sessionID = await currentSessionID(page);
      const stoppedTurn = await runningTurnID(page, sessionID);
      expect(stoppedTurn, "the answer's turn should be running").toBeTruthy();

      const cancel = watchCancel(page, STOPPED_TURN_ENDS_WITHIN_MS);
      const stoppedAt = await clickStop(page);
      const wordsAtStop = answerWordsIn(await transcript.innerText(), ANSWER);
      const composerReadyAt = await sendAsNewMessage(page, correction);

      await expect(
        transcript.getByText(STOPPED_CARD, { exact: true }).first(),
        "the reply should show it was stopped",
      ).toBeVisible();
      expect(
        wordsAtStop,
        "the Stop should land mid-reply",
      ).toBeGreaterThanOrEqual(STREAMING_WORDS);
      expect(
        wordsAtStop + WORDS_AFTER_STOP,
        "the Stop should land with most of the reply still to come",
      ).toBeLessThan(ANSWER_WORDS);
      const turnEndedAt = await expectStoppedTurnEnds(
        page,
        sessionID,
        stoppedTurn,
        stoppedAt,
      );
      const cancelled = await cancel;
      expect(cancelled.status, "the Stop should reach the server").toBe(200);

      const wordsSaved = await expectReplySavedAsStopped(
        page,
        sessionID,
        wordsAtStop,
      );
      // Live, the chat keeps the reply as it stood at the Stop, or catches up
      // with the saved one if it reads the chat before the next turn starts.
      // The reload below shows the saved one.
      const wordsShown = answerWordsIn(await transcript.innerText(), ANSWER);
      expect(
        wordsShown,
        "the chat should keep every word of the reply it showed at the Stop",
      ).toBeGreaterThanOrEqual(wordsAtStop);
      expect(
        wordsShown,
        "the chat should show no more of the reply than was saved",
      ).toBeLessThanOrEqual(wordsSaved);
      expect(
        modelTurns(await readJournal(run.nonce)),
        "the answer should be asked for once, by the SDK engine's CLI",
      ).toEqual([[1, "anthropic"]]);
      const answered = await transcript
        .getByText(CORRECTION_ANSWER, { exact: true })
        .first()
        .waitFor({ timeout: ANSWERED_WITHIN_MS })
        .then(
          () => true,
          () => false,
        );
      const replyEnd = await stoppedReplyEnd(run.nonce, stoppedAt);
      const errorCardsShown = await errorCards(page).count();
      test.info().annotations.push({
        type: "stop timing",
        description: [
          `composer took a new message ${composerReadyAt - stoppedAt} ms after Stop`,
          `server let go of the stopped turn ${turnEndedAt - stoppedAt} ms after Stop`,
          `cancel answered ${cancelled.at - stoppedAt} ms after Stop (${cancelled.reason ?? "confirmed"})`,
          `${wordsAtStop} words shown at Stop, ${wordsSaved} saved, ${wordsShown} shown once the turn ended`,
          `correction: ${await sendOutcome(page, 1)}, ${answered ? "answered" : "not answered"}, ${errorCardsShown} error card(s)`,
          `stopped reply: ${replyEnd}`,
        ].join("; "),
      });

      // Which bug shows depends on how fast the stopped turn ends. A slow
      // Stop leaves it running when the correction arrives: the correction is
      // queued behind it and deleted with its queue (#16673). A fast one cuts
      // its CLI's close short, so the CLI keeps reading the stopped reply
      // (#16729), and the correction's turn can be dropped while the
      // executor still holds the chat, under an error card ("This reply
      // could not start", #16673). So the error card check is part of the
      // pinned one: it runs on dev, where it is one of the bugs' own marks.
      const pinned = pinKnownFailure(
        "#16673, #16729: a correction sent right after Stop is never answered or can't start, or a CLI keeps reading the stopped reply",
        "the correction should be answered with no error card, and the stopped reply's model stream ended",
      );
      expect(
        [
          answered ? null : "the correction was never answered",
          errorCardsShown === 0 ? null : `${errorCardsShown} error card(s)`,
          replyEnd === "cut" ? null : `the stopped reply was ${replyEnd}`,
        ].filter(Boolean),
        pinned,
      ).toEqual([]);
      await expect
        .poll(() => savedConversation(page, sessionID), {
          message:
            "the chat should save the stopped reply, the correction and its answer, once each and in order",
        })
        .toEqual([
          ["user", firstMessage],
          ["assistant", firstWords(ANSWER, wordsSaved)],
          ["user", correction],
          ["assistant", CORRECTION_ANSWER],
        ]);
      await page.reload();
      await expect(
        page
          .getByRole("log")
          .getByText(CORRECTION_ANSWER, { exact: true })
          .first(),
      ).toBeVisible();
      await expectOnceEachInOrder(
        page,
        [
          firstWords(ANSWER, wordsSaved),
          STOPPED_CARD,
          CORRECTION,
          CORRECTION_ANSWER,
        ],
        "the reloaded chat should show the stopped reply, the correction and its answer",
      );
      expect(
        answerWordsIn(await page.getByRole("log").innerText(), ANSWER),
        "the reloaded chat should show the reply as it stopped",
      ).toBe(wordsSaved);
      expectEachTurnOnce(await readJournal(after.nonce), AFTER_STOP);
    },
  );

  test(
    "Stop while a tool runs ends the turn there, and the next message is answered",
    {
      annotation: [
        {
          type: "issue",
          description:
            "https://github.com/Significant-Gravitas/AutoGPT/pull/16729",
        },
      ],
    },
    async ({ page, journeyUser: _user }) => {
      const run = scriptedRun("stop-during-tool");
      const after = scriptedRun("after-stop");
      const firstMessage = `Set a timer, then tell me when it goes off. ${run.marker}`;
      const nextMessage = `${CORRECTION} ${after.marker}`;
      const transcript = page.getByRole("log");

      await startTheTimer(page, firstMessage, run.nonce);
      const sessionID = await currentSessionID(page);
      const stoppedTurn = await runningTurnID(page, sessionID);
      expect(stoppedTurn, "the timer's turn should be running").toBeTruthy();

      const cancel = watchCancel(page, STOPPED_TURN_ENDS_WITHIN_MS);
      const stoppedAt = await clickStop(page);
      await expect(
        transcript.getByText(STOPPED_CARD, { exact: true }).first(),
        "the turn should show it was stopped",
      ).toBeVisible();
      const turnEndedAt = await expectStoppedTurnEnds(
        page,
        sessionID,
        stoppedTurn,
        stoppedAt,
      );
      const cancelled = await cancel;
      expect(cancelled.status, "the Stop should reach the server").toBe(200);
      test.info().annotations.push({
        type: "stop timing",
        description: [
          `server let go of the stopped turn ${turnEndedAt - stoppedAt} ms after Stop`,
          `cancel answered ${cancelled.at - stoppedAt} ms after Stop (${cancelled.reason ?? "confirmed"})`,
        ].join("; "),
      });

      await sendAsNewMessage(page, nextMessage);
      await expect(
        transcript.getByText(CORRECTION_ANSWER, { exact: true }).first(),
        "the message after the Stop should be answered",
      ).toBeVisible({ timeout: ANSWERED_WITHIN_MS });
      await expect(errorCards(page), "no error card should show").toHaveCount(
        0,
      );

      // The timer never goes off in a test. What a stopped turn that carries
      // on does is ask the model for its next step, soon after the Stop.
      const watchUntil = stoppedAt + STOPPED_TURN_QUIET_FOR_MS;
      let added = "watching";
      await expect
        .poll(
          async () => {
            added =
              (await whatTheStoppedTurnAdded(page, run.nonce)) ??
              (Date.now() >= watchUntil ? "nothing" : "watching");
            return added;
          },
          {
            message: "the watch on the stopped turn should end",
            timeout: Math.max(0, watchUntil - Date.now()) + 10_000,
            intervals: [1_000],
          },
        )
        .not.toBe("watching");
      // The stopped turn's CLI outlives the Stop and carries on from the
      // cancelled timer: it asks the model for the next step (#16729).
      const pinned = pinKnownFailure(
        "#16729: a turn stopped during a tool call carries on, its CLI asking the model for the next step",
        "nothing of the stopped turn should arrive after the Stop",
      );
      expect(added, pinned).toBe("nothing");
      await expect
        .poll(() => savedConversation(page, sessionID), {
          message:
            "the chat should save the stopped turn, the next message and its answer, once each and in order",
        })
        .toEqual([
          ["user", firstMessage],
          ["assistant", TOOL_INTRO],
          ["user", nextMessage],
          ["assistant", CORRECTION_ANSWER],
        ]);
      await page.reload();
      await expect(
        page
          .getByRole("log")
          .getByText(CORRECTION_ANSWER, { exact: true })
          .first(),
      ).toBeVisible();
      await expectOnceEachInOrder(
        page,
        [TOOL_INTRO, STOPPED_CARD, CORRECTION, CORRECTION_ANSWER],
        "the reloaded chat should show the stopped turn, the next message and its answer",
      );
      expect(
        modelTurns(await readJournal(run.nonce)),
        "the timer's turn should be asked for once, by the SDK engine's CLI",
      ).toEqual([[1, "anthropic"]]);
      expectEachTurnOnce(await readJournal(after.nonce), AFTER_STOP);
    },
  );

  test(
    "a tool stopped while it runs shows as stopped, not as run, after a reload",
    {
      annotation: [
        {
          type: "issue",
          description:
            "https://github.com/Significant-Gravitas/AutoGPT/issues/17294",
        },
      ],
    },
    async ({ page, journeyUser: _user }) => {
      const run = scriptedRun("stop-during-tool");
      await startTheTimer(
        page,
        `Set a timer, then tell me when it goes off. ${run.marker}`,
        run.nonce,
      );
      const sessionID = await currentSessionID(page);
      const stoppedTurn = await runningTurnID(page, sessionID);
      expect(stoppedTurn, "the timer's turn should be running").toBeTruthy();
      const cancel = watchCancel(page, STOPPED_TURN_ENDS_WITHIN_MS);
      const stoppedAt = await clickStop(page);
      await expectStoppedTurnEnds(page, sessionID, stoppedTurn, stoppedAt);
      // A reload would cut off the Stop's own request.
      expect((await cancel).status, "the Stop should reach the server").toBe(
        200,
      );

      await page.reload();
      const transcript = page.getByRole("log");
      await expect(
        transcript.getByText(STOPPED_CARD, { exact: true }).first(),
        "the reloaded chat should show the turn was stopped",
      ).toBeVisible();
      // The stopped call is saved with an empty result, so its row reads
      // `Ran "Countdown Timer"` and the chain's heading "Ran blocks" (#17294).
      const pinned = pinKnownFailure(
        "#17294: a tool stopped by Stop is saved as an empty successful result and shows as Ran",
        "the stopped timer should not show as run after a reload",
      );
      await expect(
        transcript.getByText(/^Ran( blocks$| ")/),
        pinned,
      ).toHaveCount(0, { timeout: STOPPED_TOOL_CHECK_MS });
    },
  );
});

/**
 * Waits for the stopped turn's reply to be saved and checks it: the start of
 * the answer, every word the chat showed at the Stop, and none the model
 * streamed more than STOPPED_REPLY_CUT_WITHIN_MS after it. Returns how many
 * words it holds.
 */
async function expectReplySavedAsStopped(
  page: Page,
  sessionID: string,
  wordsAtStop: number,
) {
  let reply = "";
  await expect
    .poll(
      async () => {
        reply = (await savedConversation(page, sessionID))[1]?.[1] ?? "";
        return answerWordsIn(reply, ANSWER);
      },
      { message: "the stopped reply should be saved with every word shown" },
    )
    .toBeGreaterThanOrEqual(wordsAtStop);
  const words = answerWordsIn(reply, ANSWER);
  expect(reply, "the saved reply should be the start of the answer").toBe(
    firstWords(ANSWER, words),
  );
  expect(
    words,
    `the reply should stop growing within ${STOPPED_REPLY_CUT_WITHIN_MS} ms of the Stop`,
  ).toBeLessThanOrEqual(wordsAtStop + WORDS_AFTER_STOP);
  return words;
}

/**
 * Waits for the server to stop running `turn`, which must happen within
 * STOPPED_TURN_ENDS_WITHIN_MS of the Stop at `stoppedAt`, however long the
 * steps since the Stop took. Returns when it saw the turn end.
 */
async function expectStoppedTurnEnds(
  page: Page,
  sessionID: string,
  turn: string | null,
  stoppedAt: number,
) {
  const deadline = stoppedAt + STOPPED_TURN_ENDS_WITHIN_MS;
  await expect
    .poll(() => runningTurnID(page, sessionID), {
      message: "the server should end the stopped turn",
      timeout: Math.max(1, deadline - Date.now()),
      intervals: [100],
    })
    .not.toBe(turn);
  const endedAt = Date.now();
  expect(
    endedAt,
    `the server should end the stopped turn within ${STOPPED_TURN_ENDS_WITHIN_MS} ms of the Stop`,
  ).toBeLessThanOrEqual(deadline);
  return endedAt;
}

/**
 * Starts a chat whose model runs the Countdown Timer, and returns once the
 * timer is running: the model has finished asking for it and its card says
 * so.
 */
async function startTheTimer(page: Page, message: string, nonce: string) {
  const transcript = page.getByRole("log");
  await startChat(page, message);
  await expect(
    transcript.getByText(TOOL_INTRO, { exact: true }),
    "Otto should say it starts a timer",
  ).toBeVisible({ timeout: 60_000 });
  await expect
    .poll(async () => firstReplyCompleted(await readJournal(nonce)), {
      message: "the model should finish asking for the timer",
    })
    .toBe(true);
  await expect(
    transcript.getByText(/^Running "/).first(),
    "the timer should show as running",
  ).toBeVisible();
}

/** The model requests a run made, as [turn, dialect], side calls left out. */
function modelTurns(journal: JournalEntry[]) {
  return journal
    .filter((entry) => entry.kind !== "side_call")
    .map((entry) => [entry.turn ?? entry.kind, entry.dialect]);
}

/**
 * How the stopped reply's model stream ended: "cut" when the platform dropped
 * it within STOPPED_REPLY_CUT_WITHIN_MS of the Stop at `stoppedAt`, else what
 * happened instead. The scripted model sees a dropped stream at its next word
 * and records when; it runs on this machine's clock, as the stack does. That
 * time decides, so a late look can't pass a late cut.
 */
async function stoppedReplyEnd(nonce: string, stoppedAt: number) {
  const deadline = stoppedAt + STOPPED_REPLY_CUT_WITHIN_MS;
  let reply: JournalEntry | undefined;
  await expect
    .poll(
      async () => {
        reply = firstReply(await readJournal(nonce));
        return reply?.completed != null || Date.now() >= deadline;
      },
      {
        message: "the stopped reply's model stream should end or be checked",
        timeout: Math.max(0, deadline - Date.now()) + 10_000,
        intervals: [250],
      },
    )
    .toBe(true);
  if (reply?.completed === true) return "read to its end";
  if (reply?.completed !== false) return "still being read";
  const cutAfter = (reply.finished_at ?? Infinity) * 1000 - stoppedAt;
  return cutAfter <= STOPPED_REPLY_CUT_WITHIN_MS
    ? "cut"
    : `cut ${Math.round(cutAfter)} ms after the Stop`;
}

/** The run's first model reply, as the scripted model journaled it. */
function firstReply(journal: JournalEntry[]) {
  return journal.find((entry) => entry.kind === "turn");
}

/** Whether the run's first model reply reached the platform in full; null
 *  while it is still being sent. */
function firstReplyCompleted(journal: JournalEntry[]) {
  return firstReply(journal)?.completed ?? null;
}

/** Anything the stopped timer's turn added after the Stop, or null. */
async function whatTheStoppedTurnAdded(page: Page, nonce: string) {
  const asked = modelTurns(await readJournal(nonce));
  if (asked.length !== 1) {
    return `the model was asked again: ${JSON.stringify(asked)}`;
  }
  const shown = await page.getByRole("log").innerText();
  if (shown.includes(TOOL_ANSWER)) {
    return "the chat showed an answer to the timer";
  }
  return null;
}
