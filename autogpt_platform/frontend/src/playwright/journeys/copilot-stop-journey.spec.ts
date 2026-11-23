import type { Page } from "@playwright/test";
import {
  currentSessionID,
  firstWords,
  recordChatStreams,
  runningTurnID,
  savedMessages,
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

// A Countdown Timer block the model runs; its second turn answers the
// timer's result, which a stopped turn must never ask for.
const TOOL: ModelScript = stopDuringToolScript;
const TOOL_INTRO = TOOL.turns[0].text ?? "";
const TOOL_ANSWER = TOOL.turns[1].text ?? "";
const TIMER = stopDuringToolScript.turns[0].tool_calls?.[0]?.input.input;
if (!TIMER) throw new Error("stop-during-tool.json should run a timer");
const TIMER_MS = TIMER.seconds * 1000;
const TIMER_MESSAGE = TIMER.input_message;

const STOPPED_CARD = "Task stopped";

// wall-clock: the cancel route's own window (_CANCEL_CONFIRM_TIMEOUT_SECONDS
// in backend/api/features/chat/routes.py). A turn still taking in the reply
// that long after the Stop wasn't stopped. Counted in words the scripted
// model streams, not on this runner's clock.
const STOP_REACHES_TURN_WITHIN_MS = 5_000;
const WORDS_AFTER_STOP = STOP_REACHES_TURN_WITHIN_MS / WORD_DELAY_MS;
// wall-clock: three of the cancel route's 5 s windows. After one, the route
// completes a stopped turn itself, so only a hang lasts this long.
const STOPPED_TURN_ENDS_WITHIN_MS = 15_000;
// wall-clock: a new turn's CLI start and a one-line reply take a few
// seconds on CI; an unanswered message fails here, well inside the test's
// 90 s.
const ANSWERED_WITHIN_MS = 30_000;
// wall-clock: once the timer would have gone off, time for its result to
// reach the model and an answer to be saved, had the Stop not ended it.
const TIMER_RESULT_LANDS_WITHIN_MS = 8_000;

test.describe("copilot stop journey", () => {
  test.beforeAll(requireScriptedModel);

  test("a correction sent right after Stop is answered, after the reply as it stopped", async ({
    page,
    journeyUser: _user,
  }) => {
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
    await expect
      .poll(() => runningTurnID(page, sessionID), {
        message: "the server should end the stopped turn",
        timeout: STOPPED_TURN_ENDS_WITHIN_MS,
        intervals: [100],
      })
      .not.toBe(stoppedTurn);
    const turnEndedAt = Date.now();
    const cancelled = await cancel;
    expect(cancelled.status, "the Stop should reach the server").toBe(200);

    const wordsSaved = await expectReplySavedAsStopped(
      page,
      sessionID,
      wordsAtStop,
    );
    await expect
      .poll(async () => answerWordsIn(await transcript.innerText(), ANSWER), {
        message: "the chat should show the reply as it was saved",
      })
      .toBe(wordsSaved);
    expect(
      modelTurns(await readJournal(run.nonce)),
      "the answer should be asked for once, by the SDK engine's CLI",
    ).toEqual([[1, "anthropic"]]);
    test.info().annotations.push({
      type: "stop timing",
      description: [
        `composer took a new message ${composerReadyAt - stoppedAt} ms after Stop`,
        `server let go of the stopped turn ${turnEndedAt - stoppedAt} ms after Stop`,
        `cancel answered ${cancelled.at - stoppedAt} ms after Stop (${cancelled.reason ?? "confirmed"})`,
        `${wordsAtStop} words shown at Stop, ${wordsSaved} saved`,
        `correction: ${await sendOutcome(page, 1)}`,
      ].join("; "),
    });

    // A message sent while the Stop settles is queued behind the stopped
    // turn and deleted with its queue (#16673); one that starts its own turn
    // finds the stopped turn's CLI still holding the session (#16729).
    const pinned = pinKnownFailure(
      "#16673, #16729: a message sent right after Stop is never answered",
      "the correction sent right after Stop should be answered",
    );
    await expect(
      transcript.getByText(CORRECTION_ANSWER, { exact: true }).first(),
      pinned,
    ).toBeVisible({ timeout: ANSWERED_WITHIN_MS });
    await expect(errorCards(page), "no error card should show").toHaveCount(0);
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
    await expect
      .poll(async () => firstReplyCompleted(await readJournal(run.nonce)), {
        message:
          "the Stop should end the model's reply, not leave a CLI reading it",
      })
      .toBe(false);
    expect(
      cancelled.reason,
      "the executor should confirm the Stop inside the cancel route's 5 s",
    ).not.toBe("cancel_published_not_confirmed");
  });

  test("Stop while a tool runs ends the turn there, and the next message is answered", async ({
    page,
    journeyUser: _user,
  }) => {
    const run = scriptedRun("stop-during-tool");
    const after = scriptedRun("after-stop");
    const firstMessage = `Set a timer, then tell me when it goes off. ${run.marker}`;
    const nextMessage = `${CORRECTION} ${after.marker}`;
    const transcript = page.getByRole("log");

    await startChat(page, firstMessage);
    await expect(
      transcript.getByText(TOOL_INTRO, { exact: true }),
      "Otto should say it starts a timer",
    ).toBeVisible({ timeout: 60_000 });
    // The model's turn ends in the timer's call, so the timer runs next.
    await expect
      .poll(async () => firstReplyCompleted(await readJournal(run.nonce)), {
        message: "the model should finish asking for the timer",
      })
      .toBe(true);
    await expect(
      transcript.getByText(/^Running "/).first(),
      "the timer should show as running",
    ).toBeVisible();
    const timerStartedBy = Date.now();
    const sessionID = await currentSessionID(page);
    const stoppedTurn = await runningTurnID(page, sessionID);
    expect(stoppedTurn, "the timer's turn should be running").toBeTruthy();

    const cancel = watchCancel(page, STOPPED_TURN_ENDS_WITHIN_MS);
    const stoppedAt = await clickStop(page);
    await expect(
      transcript.getByText(STOPPED_CARD, { exact: true }).first(),
      "the turn should show it was stopped",
    ).toBeVisible();
    await expect
      .poll(() => runningTurnID(page, sessionID), {
        message: "the server should end the stopped turn",
        timeout: STOPPED_TURN_ENDS_WITHIN_MS,
        intervals: [100],
      })
      .not.toBe(stoppedTurn);
    const turnEndedAt = Date.now();
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
    await expect(errorCards(page), "no error card should show").toHaveCount(0);

    // Watched until the timer's result, had the Stop not ended its turn,
    // would have reached the model and its answer the chat.
    const watchUntil = timerStartedBy + TIMER_MS + TIMER_RESULT_LANDS_WITHIN_MS;
    let added = "watching";
    await expect
      .poll(
        async () => {
          added =
            (await whatTheStoppedTurnAdded(page, sessionID, run.nonce)) ??
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
      "nothing of the stopped turn should arrive by the time its timer would have gone off",
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
    expect(
      cancelled.reason,
      "the executor should confirm the Stop inside the cancel route's 5 s",
    ).not.toBe("cancel_published_not_confirmed");
  });
});

/**
 * Waits for the stopped turn's reply to be saved and checks it: the start of
 * the answer, every word the chat showed at the Stop, and none the model
 * streamed more than STOP_REACHES_TURN_WITHIN_MS after it. Returns how many
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
    `the reply should stop growing within ${STOP_REACHES_TURN_WITHIN_MS} ms of the Stop`,
  ).toBeLessThanOrEqual(wordsAtStop + WORDS_AFTER_STOP);
  return words;
}

/** The model requests a run made, as [turn, dialect], side calls left out. */
function modelTurns(journal: JournalEntry[]) {
  return journal
    .filter((entry) => entry.kind !== "side_call")
    .map((entry) => [entry.turn ?? entry.kind, entry.dialect]);
}

/** Whether the run's first model reply reached the platform in full; null
 *  while it is still being sent. */
function firstReplyCompleted(journal: JournalEntry[]) {
  return journal.find((entry) => entry.kind === "turn")?.completed ?? null;
}

/** Anything the stopped timer's turn added after the Stop, or null. */
async function whatTheStoppedTurnAdded(
  page: Page,
  sessionID: string,
  nonce: string,
) {
  const asked = modelTurns(await readJournal(nonce));
  if (asked.length !== 1) {
    return `the model was asked again: ${JSON.stringify(asked)}`;
  }
  // The call's own arguments name the message too; only a result says it
  // went off.
  const results = (await savedMessages(page, sessionID)).filter(
    (message) => message.role === "tool",
  );
  if (results.some((result) => result.content?.includes(TIMER_MESSAGE))) {
    return "the timer's result was saved";
  }
  const shown = await page.getByRole("log").innerText();
  if (shown.includes(TOOL_ANSWER))
    return "the chat showed an answer to the timer";
  if (/Ran "/.test(shown)) return "the chat showed the timer as finished";
  return null;
}
